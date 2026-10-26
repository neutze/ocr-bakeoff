"""Page images: rendering, the simulated scan, clean-up, crops and blank-row cuts.

Rendering follows the same author's document pipeline (MIT): 300 DPI, CropBox widened to
the MediaBox so scanner-clipped headers and footers survive, Tesseract's orientation
check before the page is used. The rest is OpenCV, so every lane but Apple Vision runs
on any OS.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

from .config import MAX_PAGE_PIXELS, RENDER_DPI, TESSDATA


def render_scale(width_points: float, height_points: float, dpi: int = RENDER_DPI) -> float:
    scale = max(dpi, 1) / 72.0
    longest = max(width_points, height_points)
    if MAX_PAGE_PIXELS > 0 and longest > 0 and longest * scale > MAX_PAGE_PIXELS:
        scale = MAX_PAGE_PIXELS / longest
    return scale


def detect_rotation(png: bytes) -> int:
    """Clockwise degrees to read upright, from Tesseract OSD. 0 whenever it is unsure."""
    if not (TESSDATA / "osd.traineddata").exists():
        return 0
    try:
        done = subprocess.run(
            ["tesseract", "stdin", "stdout", "--psm", "0", "--tessdata-dir", str(TESSDATA)],
            input=png,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    if done.returncode != 0:
        return 0
    for line in done.stdout.decode("utf-8", "replace").splitlines():
        if line.startswith("Rotate:"):
            try:
                return int(line.split(":", 1)[1].strip()) % 360
            except ValueError:
                return 0
    return 0


def render_pdf_page(path: Path, page_index: int, *, auto_orient: bool = True) -> bytes:
    with fitz.open(path) as pdf:
        page = pdf.load_page(page_index)
        try:
            if page.cropbox != page.mediabox:
                page.set_cropbox(page.mediabox)
        except ValueError:
            # PyMuPDF refuses a CropBox it cannot place inside the MediaBox, which happens when
            # the MediaBox does not start at the origin. Such a page renders with its CropBox.
            pass
        scale = render_scale(page.rect.width, page.rect.height)
        png = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png")
        if auto_orient:
            rotation = detect_rotation(png)
            if rotation:
                png = page.get_pixmap(matrix=fitz.Matrix(scale, scale).prerotate(rotation), alpha=False).tobytes("png")
        return png


def image_as_png(path: Path) -> bytes:
    """An image file's first frame, as PNG, pixels unchanged."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image {path.name}")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"cannot encode {path.name}")
    return encoded.tobytes()


def decode_gray(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("cannot decode image")
    return image


def degrade_transform(seed: int, width: int, height: int) -> tuple[np.random.Generator, np.ndarray, int, int]:
    """The simulated scan's geometry: scale to 2/3, then a seeded 0.5-1.5 deg rotation.

    Exposed so a reference word box on the clean page can be carried onto the degraded one,
    which is what lets word positions be scored on a scan-like image. Returns the generator
    in the state degrade() continues from, so its output stays byte-identical."""
    rng = np.random.default_rng(seed)
    w2, h2 = round(width * 2 / 3), round(height * 2 / 3)
    degrees = (0.5 + rng.random()) * (1 if rng.random() < 0.5 else -1)
    return rng, cv2.getRotationMatrix2D((w2 / 2, h2 / 2), degrees, 1.0), w2, h2


def map_box(
    box: tuple[float, float, float, float], seed: int, width: int, height: int
) -> tuple[float, float, float, float]:
    """A normalised clean-page box, as the axis-aligned normalised box it becomes on the degraded page."""
    _, matrix, w2, h2 = degrade_transform(seed, width, height)
    x0, y0, x1, y1 = box[0] * w2, box[1] * h2, box[2] * w2, box[3] * h2
    corners = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]]) @ matrix.T
    return (corners[:, 0].min() / w2, corners[:, 1].min() / h2, corners[:, 0].max() / w2, corners[:, 1].max() / h2)


def degrade(png: bytes, seed: int) -> bytes:
    """A deterministic simulated scan: 200 DPI, 0.5-1.5 deg skew, grey, flatter, noisy, JPEG q60."""
    image = decode_gray(png)
    h, w = image.shape
    rng, matrix, w2, h2 = degrade_transform(seed, w, h)
    image = cv2.resize(image, (w2, h2), interpolation=cv2.INTER_AREA)
    image = cv2.warpAffine(image, matrix, (w2, h2), flags=cv2.INTER_LINEAR, borderValue=255)
    flat = (image.astype(np.float32) - 128.0) * 0.85 + 128.0 + 0.03 * 255
    flat += rng.normal(0.0, 8.0, size=flat.shape).astype(np.float32)
    ok, encoded = cv2.imencode(".jpg", np.clip(flat, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 60])
    if not ok:
        raise ValueError("cannot encode degraded page")
    return encoded.tobytes()


def estimate_skew(gray: np.ndarray) -> float:
    """Degrees (counter-clockwise) that make text rows horizontal, by projection profile.

    Tries -5..5 degrees on a binarised thumbnail and keeps the angle whose row sums
    vary most: straight rows give sharp peaks and empty gaps. Outside that range the
    page is laid out that way, not skewed.
    """
    scale = 800 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    _, ink = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if ink.sum() < 50:
        return 0.0
    h, w = ink.shape
    best, best_score = 0.0, -1.0
    for angle in np.arange(-5.0, 5.01, 0.25):
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
        rotated = cv2.warpAffine(ink.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best, best_score = float(angle), score
    return best


def clean(data: bytes) -> tuple[bytes, float]:
    """Deskew, grey, a little contrast and sharpening: what OCRmyPDF feeds Tesseract."""
    gray = decode_gray(data)
    skew = estimate_skew(gray)
    if skew:
        h, w = gray.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
        gray = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    flat = np.clip((gray.astype(np.float32) - 128.0) * 1.15 + 128.0, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(flat, (0, 0), 1.2)
    sharp = cv2.addWeighted(flat, 1.6, blurred, -0.6, 0)
    ok, encoded = cv2.imencode(".png", sharp)
    if not ok:
        raise ValueError("cannot encode cleaned page")
    return encoded.tobytes(), skew


def crop(data: bytes, box: tuple[float, float, float, float]) -> bytes:
    """PNG of a normalised top-left box [x0, y0, x1, y1], native resolution."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    part = image[max(0, int(y0 * h)) : min(h, int(round(y1 * h))), max(0, int(x0 * w)) : min(w, int(round(x1 * w)))]
    ok, encoded = cv2.imencode(".png", part)
    if not ok:
        raise ValueError("cannot encode crop")
    return encoded.tobytes()


def blank_row_fraction(data: bytes, target: float, window: float = 0.08) -> float:
    """The row nearest `target` (a page fraction) with the least ink, within +-window.

    A header crop cut at exactly 20% slices through any line that straddles it, and a
    half-line becomes garbage tokens in the gold. Cutting at the emptiest nearby row
    keeps every line whole.
    """
    gray = decode_gray(data)
    h = gray.shape[0]
    lo, hi = max(1, int((target - window) * h)), min(h - 1, int((target + window) * h))
    ink = (gray[lo:hi] < 160).sum(axis=1)
    least = ink.min()
    candidates = np.flatnonzero(ink == least) + lo
    row = int(candidates[np.argmin(np.abs(candidates - target * h))])
    return row / h


def mime_for(path: Path) -> str:
    return "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"


def page_size(data: bytes) -> tuple[int, int]:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    return image.shape[1], image.shape[0]
