"""The lane catalogue: which engine configurations exist and what each one is. No I/O.

Kept apart from `lanes`, which runs them: scoring, reporting and dependency checks need to
know a lane's family and model, not how to call it.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class Lane:
    id: str
    label: str
    family: str  # "llm" | "classic" | "control" | "pipeline"
    geometry: str
    plan_id: str
    model: str | None = None  # a model in models.MODELS, for lanes that run on llama-server
    macos_only: bool = False
    engine: str | None = None  # the program that reads the page, for lanes without a model


LANES: dict[str, Lane] = {
    lane.id: lane
    for lane in (
        Lane("glm-custom", "GLM-OCR, whole page, a 39-line custom prompt", "llm", "none", "L0", "glm"),
        Lane("glm", "GLM-OCR, whole page, native prompt", "llm", "none", "L1", "glm"),
        Lane("glm-bands", "GLM-OCR, three fixed bands", "llm", "band boxes", "L2a", "glm"),
        Lane("glm-blocks", "GLM-OCR, Apple Vision text blocks", "llm", "block boxes", "L2b", "glm", macos_only=True),
        Lane("paddle", "PaddleOCR-VL-1.6, whole page", "llm", "none", "M1", "paddle"),
        Lane(
            "paddle-blocks",
            "PaddleOCR-VL-1.6, Apple Vision blocks",
            "llm",
            "block boxes",
            "M1b",
            "paddle",
            macos_only=True,
        ),
        Lane("qwen8b", "Qwen3-VL-8B, whole page", "llm", "none", "M2", "qwen8b"),
        Lane("qwen4b", "Qwen3-VL-4B, whole page", "llm", "none", "M3", "qwen4b"),
        Lane("olmocr", "olmOCR-2-7B, whole page", "llm", "none", "M4", "olmocr"),
        Lane("dots", "dots.ocr, whole page, its plain-OCR prompt", "llm", "none", "M5", "dots"),
        Lane("deepseek", "DeepSeek-OCR, whole page, its free-OCR prompt", "llm", "none", "M6", "deepseek"),
        Lane("lighton", "LightOnOCR-2-1B, whole page, no instruction", "llm", "none", "M7", "lighton"),
        Lane("chandra", "Chandra-OCR-2, whole page, its OCR prompt", "llm", "none", "M9", "chandra"),
        Lane("ovis", "OvisOCR2, whole page, its Markdown prompt", "llm", "none", "M10", "ovis"),
        Lane(
            "unlimited",
            "Unlimited-OCR, whole page, its document-parsing prompt after the image",
            "llm",
            "none",
            "M11",
            "unlimited",
        ),
        Lane(
            "lighton-tuned",
            "LightOnOCR-2-1B, page shrunk to 1540 px, longer answers, repetition penalty",
            "llm",
            "none",
            "T1",
            "lighton-t",
        ),
        Lane("ovis-tuned", "OvisOCR2, longer answers, repetition penalty", "llm", "none", "T2", "ovis-t"),
        Lane(
            "paddle-blocks-tuned",
            "PaddleOCR-VL-1.6 on Vision blocks, longer answers, repetition penalty",
            "llm",
            "block boxes",
            "T3",
            "paddle-t",
            macos_only=True,
        ),
        Lane("chandra-tuned", "Chandra-OCR-2, longer answers, repetition penalty", "llm", "none", "T4", "chandra-t"),
        Lane("qwen8b-tuned", "Qwen3-VL-8B, longer answers, repetition penalty", "llm", "none", "T5", "qwen8b-t"),
        Lane(
            "unlimited-tuned", "Unlimited-OCR, longer answers, repetition penalty", "llm", "none", "T6", "unlimited-t"
        ),
        # Prompt variants (docs/prompt-research-2026-10.md): each changes one thing against the model's first lane.
        Lane(
            "unlimited-dry",
            "Unlimited-OCR, tuned, DRY sampling in place of its maker's repetition rule",
            "llm",
            "none",
            "V0",
            "unlimited-t",
        ),
        Lane("deepseek-first", "DeepSeek-OCR, the image before its free-OCR prompt", "llm", "none", "V1", "deepseek"),
        Lane("deepseek-ocr", "DeepSeek-OCR, llama.cpp's short prompt", "llm", "none", "V2", "deepseek"),
        Lane(
            "deepseek-md",
            "DeepSeek-OCR, its grounding prompt for Markdown after the image, longer answers",
            "llm",
            "none",
            "V3",
            "deepseek-t",
        ),
        Lane("dots-layout", "dots.ocr, its full layout prompt, longer answers", "llm", "none", "V4", "dots-t"),
        Lane("qwen8b-md", "Qwen3-VL-8B, its trained keyword for Markdown", "llm", "none", "V5", "qwen8b"),
        Lane("qwen8b-plain", "Qwen3-VL-8B, its cookbook's plain prompt", "llm", "none", "V6", "qwen8b"),
        Lane("qwen8b-2mp", "Qwen3-VL-8B, page shrunk to 2.1 megapixels", "llm", "none", "V7", "qwen8b"),
        Lane("chandra-192dpi", "Chandra-OCR-2, tuned, page shrunk to 192 dpi", "llm", "none", "V12", "chandra-t"),
        Lane("chandra-2mp", "Chandra-OCR-2, tuned, page shrunk to 2.1 megapixels", "llm", "none", "V13", "chandra-t"),
        Lane("chandra-layout", "Chandra-OCR-2, its layout prompt, longer answers", "llm", "none", "V8", "chandra-t"),
        Lane("ovis-first", "OvisOCR2, the image before its prompt", "llm", "none", "V9", "ovis"),
        Lane("olmocr-native", "olmOCR-2, its maker's prompt on a page of 1288 px", "llm", "none", "V11", "olmocr"),
        Lane(
            "vision",
            "Apple Vision, language correction on",
            "classic",
            "word boxes",
            "L4",
            macos_only=True,
            engine="vision",
        ),
        Lane(
            "vision-raw",
            "Apple Vision, language correction off",
            "classic",
            "word boxes",
            "L4b",
            macos_only=True,
            engine="vision",
        ),
        Lane("tesseract", "Tesseract, raw page", "classic", "word boxes", "L5", engine="tesseract"),
        Lane(
            "tesseract-tur",
            "Tesseract, raw page, with Turkish data",
            "classic",
            "word boxes",
            "L5t",
            engine="tesseract",
        ),
        Lane("tesseract-clean", "Tesseract, deskewed + sharpened", "classic", "word boxes", "L5c", engine="tesseract"),
        Lane("rapidocr", "RapidOCR (PP-OCRv5 mobile, latin)", "classic", "line boxes", "N1", engine="rapidocr"),
        Lane(
            "rapidocr-server",
            "RapidOCR (PP-OCRv5 server detector, latin recogniser)",
            "classic",
            "line boxes",
            "N1s",
            engine="rapidocr",
        ),
        Lane("doctr", "docTR via OnnxTR, default models", "classic", "word boxes", "N2", engine="doctr"),
        Lane("easyocr", "EasyOCR, German + English + Turkish, CPU", "classic", "line boxes", "N3", engine="easyocr"),
        Lane(
            "auto-paddle",
            "text layer where trustworthy, PaddleOCR-VL for images and the rest",
            "pipeline",
            "mixed",
            "P1",
            "paddle",
        ),
        Lane(
            "auto-qwen4b",
            "text layer where trustworthy, Qwen3-VL-4B for images and the rest",
            "pipeline",
            "mixed",
            "P2",
            "qwen4b",
        ),
        Lane("pdftotext", "pdftotext text layer", "control", "word boxes", "L6", engine="pdftotext"),
        Lane("pymupdf", "PyMuPDF text layer", "control", "char boxes", "L7", engine="pymupdf"),
    )
}

# Families that read the PDF's own text layer rather than the page image.
LAYER_READERS = ("control", "pipeline")


def variant(lane: Lane) -> bool:
    return lane.plan_id.startswith("V")


def available_here(lane: Lane) -> bool:
    return not lane.macos_only or platform.system() == "Darwin"


def ranked(lane: Lane) -> bool:
    """Competes under the decision rule. Controls and pipelines are reported, never ranked."""
    return lane.family not in LAYER_READERS


def reads_layer(lane_id: str) -> bool:
    return LANES[lane_id].family in LAYER_READERS


def variants_for(lane_id: str, page_variants: dict[str, str]) -> dict[str, str]:
    """Controls and pipelines read the PDF's layer. On the simulated scan the source still has
    that layer, which a real scan would not, so they run on the page as it is and on real scans."""
    if reads_layer(lane_id):
        return {v: f for v, f in page_variants.items() if v != "degraded"}
    return page_variants
