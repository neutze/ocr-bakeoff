"""Profile a folder's documents by structure only, so look-alikes can be generated.

Reads each sampled PDF locally and records *shape*, never content: page format, born-digital
or scan, which kind of application produced it, whether header and footer are text, image
or absent, columns, table rules, font sizes, share of digits, a language guess. The
terminal gets counts. The per-file profiles (with paths) go to a private JSON in the cache,
never to stdout, so a session driving this tool sees no file names and no text.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf as fitz

from .config import CACHE, FOOTER_FRACTION, HEADER_FRACTION, OCR_PRODUCER_RE, TEXT_LAYER_MIN_CHARS
from .corpus import picture_rects, written_by_ocr_tool

PRODUCERS = [
    ("presentation", r"powerpoint|keynote|impress|google slides"),
    ("word-processor", r"word|pages|writer|libreoffice|openoffice|google docs"),
    ("latex", r"tex|pdfetex|xetex|luatex"),
    ("browser/web", r"chrome|chromium|skia|safari|firefox|webkit|wkhtml|puppeteer"),
    ("scanner/ocr", OCR_PRODUCER_RE.pattern),
    ("invoicing/erp", r"sap|datev|lexware|sevdesk|jasper|crystal|birt|reportlab|itext|fop|dompdf|tcpdf|fpdf"),
    ("acrobat/distiller", r"acrobat|distiller|pdf library|pdfmaker"),
    ("print-to-pdf", r"quartz|macos|mac os|print|cairo|ghostscript|gpl"),
]
_STOP = {
    "de": {"und", "der", "die", "das", "mit", "für", "ist", "nicht", "sie", "wir", "ihre", "vom"},
    "en": {"the", "and", "of", "to", "for", "with", "your", "is", "are", "this", "you", "please"},
    "tr": {"ve", "bir", "bu", "için", "ile", "olarak", "da", "de", "sayın", "tarihinde", "olan"},
}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def producer_bucket(meta: dict[str, Any]) -> str:
    text = " ".join(filter(None, [meta.get("producer"), meta.get("creator")])).lower()
    if not text:
        return "unknown"
    return next((name for name, pattern in PRODUCERS if re.search(pattern, text, re.IGNORECASE)), "other")


def language(text: str) -> str:
    words = [w.lower() for w in _WORD.findall(text)[:600]]
    if len(words) < 15:
        return "unknown"
    scores = {lang: sum(w in stop for w in words) for lang, stop in _STOP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 3 else "other"


def page_shape(width: float, height: float) -> str:
    near = lambda a, b: abs(a - b) <= 15  # noqa: E731
    if near(width, 595) and near(height, 842):
        return "a4-portrait"
    if near(width, 842) and near(height, 595):
        return "a4-landscape"
    if near(width, 612) and near(height, 792):
        return "letter-portrait"
    ratio = width / height
    if ratio > 1.7:
        return "16:9"
    if ratio > 1.25:
        return "4:3/landscape"
    return "portrait-other"


def _band(rect: fitz.Rect, height: float) -> str:
    if rect.y0 < HEADER_FRACTION * height:
        return "top"
    if rect.y1 > (1 - FOOTER_FRACTION) * height:
        return "bottom"
    return "middle"


def profile_pdf(path: Path) -> dict[str, Any] | None:
    try:
        with fitz.open(path) as pdf:
            if pdf.needs_pass or len(pdf) == 0:
                return None
            page = pdf[0]
            w, h = page.rect.width, page.rect.height
            text = page.get_text("text")
            chars = len(text.strip())
            images = picture_rects(page)
            image_cover = sum(r.get_area() for r in images) / max(1.0, w * h)
            blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
            sizes = [s["size"] for b in blocks for l in b["lines"] for s in l["spans"] if s["text"].strip()]
            bands_text = Counter(_band(fitz.Rect(b["bbox"]), h) for b in blocks)
            bands_img = Counter(_band(r, h) for r in images if r.get_area() > 0.002 * w * h)
            body = [
                fitz.Rect(b["bbox"])
                for b in blocks
                if 0.2 * h < b["bbox"][1] < 0.8 * h and (b["bbox"][2] - b["bbox"][0]) < 0.6 * w
            ]
            lefts = sorted({round(r.x0 / w, 1) for r in body})
            columns = 2 if any(abs(a - b) >= 0.3 for a in lefts for b in lefts) and len(body) >= 4 else 1
            try:
                rules = sum(
                    1
                    for d in page.get_drawings()[:4000]
                    for item in d["items"]
                    if item[0] == "l" and abs(item[1].y - item[2].y) < 0.5 and abs(item[1].x - item[2].x) > 0.15 * w
                )
            except Exception:
                rules = 0
            window = fitz.Rect(0.08 * w, 0.14 * h, 0.5 * w, 0.32 * h)
            producer = producer_bucket(pdf.metadata or {})
            has_ocr_producer = written_by_ocr_tool(pdf)
            if chars >= TEXT_LAYER_MIN_CHARS and not has_ocr_producer:
                tier = "born-digital"
            elif chars >= TEXT_LAYER_MIN_CHARS:
                tier = "old-ocr-layer"
            elif image_cover > 0.5:
                tier = "scan"
            else:
                tier = "sparse"
            digit_share = sum(c.isdigit() for c in text) / max(1, chars)
            return {
                "path": str(path),
                "pages": len(pdf),
                "shape": page_shape(w, h),
                "tier": tier,
                "producer": producer,
                "header": ("image" if bands_img["top"] else "") + ("text" if bands_text["top"] else "") or "none",
                "footer": ("image" if bands_img["bottom"] else "") + ("text" if bands_text["bottom"] else "") or "none",
                "columns": columns,
                "table_rules": rules,
                "address_window": any(fitz.Rect(b["bbox"]).intersects(window) for b in blocks),
                "font_max": round(max(sizes), 1) if sizes else None,
                "font_median": round(statistics.median(sizes), 1) if sizes else None,
                "digit_share": round(digit_share, 3),
                "language": language(text),
                "image_cover": round(image_cover, 2),
            }
    except Exception:
        return None


def kind_of(p: dict[str, Any]) -> str:
    if p["tier"] in ("scan", "old-ocr-layer"):
        return "scans"
    if p["shape"] in ("16:9", "4:3/landscape", "a4-landscape") or p["producer"] == "presentation":
        return "presentations"
    if p["table_rules"] >= 4 or p["digit_share"] > 0.12:
        return "invoices"
    if p["address_window"] and p["pages"] <= 4:
        return "letters"
    if p["columns"] >= 2 or p["pages"] >= 5 or p["producer"] == "latex":
        return "work-docs"
    return "other"


def profile_folder(folder: Path, sample: int, seed: int = 20260926) -> Path:
    paths = sorted(
        p
        for p in folder.rglob("*")
        if p.suffix.lower() == ".pdf"
        and p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(folder).parts)
    )
    chosen = random.Random(seed).sample(paths, min(sample, len(paths)))
    profiles = []
    for index, path in enumerate(chosen, 1):
        profile = profile_pdf(path)
        if profile:
            profile["kind"] = kind_of(profile)
            profile["id"] = f"d{index:04d}"
            profiles.append(profile)
    out = CACHE / "profiles" / f"{re.sub(r'[^A-Za-z0-9]+', '-', folder.name).strip('-') or 'folder'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "folder": str(folder),
                "total_pdfs": len(paths),
                "sampled": len(chosen),
                "seed": seed,
                "profiles": profiles,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out.chmod(0o600)
    return out


def summarise(profile_file: Path) -> str:
    data = json.loads(profile_file.read_text(encoding="utf-8"))
    profiles = data["profiles"]
    lines = [
        f"{data['total_pdfs']} PDFs found, {data['sampled']} sampled, {len(profiles)} readable "
        f"(the rest encrypted or broken). Paths stay in {profile_file} (mode 600)."
    ]
    for field in ("kind", "tier", "shape", "producer", "header", "footer", "columns", "language"):
        counts = Counter(str(p[field]) for p in profiles)
        lines.append(f"{field:<9} " + " · ".join(f"{k} {v}" for k, v in counts.most_common()))
    by_kind = Counter()
    for p in profiles:
        if "image" in p["header"]:
            by_kind[p["kind"]] += 1
    lines.append("image headers by kind: " + " · ".join(f"{k} {v}" for k, v in by_kind.most_common()))
    pages = [p["pages"] for p in profiles]
    if pages:  # a folder with no readable PDF has no page statistics
        lines.append(
            f"pages     median {statistics.median(pages)}, 90th pct {sorted(pages)[int(0.9 * len(pages))]}, max {max(pages)}"
        )
    return "\n".join(lines)
