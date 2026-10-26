"""The test folder: which pages, what kind, born-digital or scan, and their images.

Layout convention:

    testfiles/
      letters/          <- the subfolder name is the document kind
        a.pdf
        a.gold.txt      <- optional typed reference for a.pdf page 1 (a.p2.gold.txt for page 2)
      presentations/
      scans/
      x.pdf             <- files at the top level are kind "unsorted"

A PDF page whose text layer is substantial and was not written by an OCR tool is
born-digital: its own text layer is the reference, and it is scored twice, as rendered
and as a simulated scan. Everything else is a scan.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from html import unescape
from pathlib import Path
from typing import Any

import pymupdf as fitz

from .config import (
    FOOTER_FRACTION,
    HEADER_FRACTION,
    OCR_PRODUCER_RE,
    SOURCE_SUFFIXES,
    SPARSE_MAX_PICTURE_SHARE,
    SPARSE_MIN_CHARS,
    TEXT_LAYER_MIN_CHARS,
    VISIBILITY_DPI,
    VISIBILITY_MIN_STD,
)
from .render import degrade, image_as_png, render_pdf_page
from .text import run_text

RESULTS_DIRNAME = ".ocr-bakeoff"
_WORD_RE = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>')
_PAGE_RE = re.compile(r'<page width="([\d.]+)" height="([\d.]+)">')


@dataclass
class Page:
    key: str
    rel: str
    source: str
    page: int
    kind: str
    tier: str  # "born-digital" | "scan"
    variants: dict[str, str] = field(default_factory=dict)
    typed_gold: str | None = None

    @property
    def path(self) -> Path:
        return Path(self.source)

    @property
    def seed(self) -> int:
        """The simulated scan's seed: fixed per file, so its degraded image never changes."""
        return int(self.key[:8], 16)


def reference_path(workdir: Path, image_key: str) -> Path:
    """Where an external reference (Google Drive's reading) for one page image is cached."""
    return workdir / "reference" / f"{image_key}.json"


def file_digest(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def kind_of(testdir: Path, path: Path) -> str:
    parts = path.relative_to(testdir).parts
    return parts[0].strip().lower().replace(" ", "-") if len(parts) > 1 else "unsorted"


def typed_gold_path(path: Path, page: int) -> Path | None:
    names = [f"{path.stem}.p{page}.gold.txt"] + ([f"{path.stem}.gold.txt"] if page == 1 else [])
    return next((path.with_name(n) for n in names if path.with_name(n).exists()), None)


def written_by_ocr_tool(pdf: fitz.Document) -> bool:
    """The file's producer or creator names an OCR tool or a scanner: its layer, if any, is
    an old OCR pass with the right length and garbage content."""
    producer = " ".join(filter(None, [pdf.metadata.get("producer"), pdf.metadata.get("creator")]))
    return bool(OCR_PRODUCER_RE.search(producer))


def picture_rects(page: fitz.Page) -> list[fitz.Rect]:
    """Every picture on the page, clipped to it; pictures entirely off the page are left out."""
    return [r for r in (fitz.Rect(i["bbox"]) & page.rect for i in page.get_image_info()) if not r.is_empty]


def text_layer_ok(pdf: fitz.Document, index: int) -> bool:
    """Born-digital: a text layer not written by an OCR tool, and either substantial, or
    sparse on a page that is not mostly picture. A slide with a title and four bullets has
    far under 200 characters and is still born-digital; a scan never has its page mostly
    covered by vector text."""
    page = pdf[index]
    chars = len(page.get_text("text").strip())
    if written_by_ocr_tool(pdf):
        return False
    if chars >= TEXT_LAYER_MIN_CHARS:
        return True
    area = page.rect.width * page.rect.height
    pictures = sum(r.get_area() for r in picture_rects(page))
    return chars >= SPARSE_MIN_CHARS and pictures < SPARSE_MAX_PICTURE_SHARE * area


def discover(testdir: Path, max_pages: int = 1) -> list[Page]:
    pages: list[Page] = []
    for path in sorted(testdir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if any(part.startswith(".") for part in path.relative_to(testdir).parts):
            continue  # hidden folders, including the results folder
        digest, kind = file_digest(path), kind_of(testdir, path)
        rel = str(path.relative_to(testdir))
        if path.suffix.lower() == ".pdf":
            with fitz.open(path) as pdf:
                for index in range(min(len(pdf), max_pages)):
                    tier = "born-digital" if text_layer_ok(pdf, index) else "scan"
                    gold = typed_gold_path(path, index + 1)
                    pages.append(
                        Page(
                            f"{digest}-p{index + 1}",
                            rel,
                            str(path),
                            index + 1,
                            kind,
                            tier,
                            typed_gold=str(gold) if gold else None,
                        )
                    )
        else:
            gold = typed_gold_path(path, 1)
            pages.append(Page(f"{digest}-p1", rel, str(path), 1, kind, "scan", typed_gold=str(gold) if gold else None))
    return pages


_LINE_RE = re.compile(r"<line[^>]*>(.*?)</line>", re.DOTALL)


def visibility_map(path: Path, page: int, width: float, height: float):
    """A function telling whether a word box (in pdftotext's points) shows any ink.

    Real text layers carry text nobody can see: invisible text under a picture, white on
    white, template leftovers, words outside the page. An engine that rightly does not read
    them must not be charged for it, so the reference keeps only words whose box has
    contrast on the rendered page."""
    import cv2
    import numpy as np

    with fitz.open(path) as pdf:
        pix = pdf[page - 1].get_pixmap(dpi=VISIBILITY_DPI, alpha=False)
    image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else image[:, :, 0]
    sx, sy = pix.width / max(width, 1.0), pix.height / max(height, 1.0)

    def visible(x0: float, y0: float, x1: float, y1: float) -> bool:
        a, b = max(0, int(x0 * sx)), max(0, int(y0 * sy))
        c, d = min(pix.width, int(round(x1 * sx + 0.5))), min(pix.height, int(round(y1 * sy + 0.5)))
        if c <= a or d <= b:
            return False
        return float(gray[b:d, a:c].std()) >= VISIBILITY_MIN_STD

    return visible


def text_layer_gold(path: Path, page: int) -> dict[str, Any]:
    """Visible text only: full text by line, plus header and footer words by position."""
    bbox = run_text(
        ["pdftotext", "-f", str(page), "-l", str(page), "-enc", "UTF-8", "-bbox-layout", str(path), "-"],
        check=True,
    ).stdout
    size = _PAGE_RE.search(bbox)
    width, height = (float(size.group(1)), float(size.group(2))) if size else (0.0, 0.0)
    visible = visibility_map(path, page, width, height)
    lines, header, footer, hidden, words = [], [], [], 0, []
    for line in _LINE_RE.findall(bbox):
        kept = []
        for x0, y0, x1, y1, word in _WORD_RE.findall(line):
            x0, y0, x1, y1 = map(float, (x0, y0, x1, y1))
            if not visible(x0, y0, x1, y1):
                hidden += 1
                continue
            text = unescape(word)
            kept.append(text)
            if width and height:
                words.append(
                    [text, round(x0 / width, 5), round(y0 / height, 5), round(x1 / width, 5), round(y1 / height, 5)]
                )
            if height and y0 < HEADER_FRACTION * height:
                header.append(text)
            if height and y1 > (1 - FOOTER_FRACTION) * height:
                footer.append(text)
        if kept:
            lines.append(" ".join(kept))
    # pdftotext measures within the CropBox, the rendered page is the MediaBox: positions only
    # line up when the two are the same, which is most PDFs.
    with fitz.open(path) as pdf:
        geometry_ok = pdf[page - 1].cropbox == pdf[page - 1].mediabox
    return {
        "full_text": "\n".join(lines),
        "header_text": " ".join(header),
        "footer_text": " ".join(footer),
        "hidden_words": hidden,
        "words": words,
        "geometry_ok": geometry_ok,
    }


def prepare(pages: list[Page], workdir: Path) -> None:
    """Render every variant once and extract text-layer gold. Idempotent: existing files are kept."""
    images, gold = workdir / "pages", workdir / "gold"
    images.mkdir(parents=True, exist_ok=True)
    gold.mkdir(parents=True, exist_ok=True)
    for page in pages:
        is_pdf = page.path.suffix.lower() == ".pdf"
        if page.tier == "born-digital":
            clean, degraded = images / f"{page.key}.clean.png", images / f"{page.key}.degraded.jpg"
            if not clean.exists():
                clean.write_bytes(render_pdf_page(page.path, page.page - 1))
            if not degraded.exists():
                degraded.write_bytes(degrade(clean.read_bytes(), seed=page.seed))
            page.variants = {"clean": clean.name, "degraded": degraded.name}
            layer = gold / f"{page.key}.textlayer.json"
            if not layer.exists():
                layer.write_text(
                    json.dumps(text_layer_gold(page.path, page.page), ensure_ascii=False), encoding="utf-8"
                )
        else:
            scan = images / f"{page.key}.scan.png"
            if not scan.exists():
                scan.write_bytes(render_pdf_page(page.path, page.page - 1) if is_pdf else image_as_png(page.path))
            page.variants = {"scan": scan.name}
    (workdir / "pages.json").write_text(
        json.dumps([asdict(p) for p in pages], ensure_ascii=False, indent=1), encoding="utf-8"
    )
