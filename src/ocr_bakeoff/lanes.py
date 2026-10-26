"""Running the lanes: one engine configuration each, all fed the same page images.

LLM lanes change one thing at a time against GLM-OCR: the prompt (glm-custom -> glm),
the layout stage (glm -> glm-bands / glm-blocks), then the model. Classic lanes are the
non-LLM engines, paired to isolate one setting each. Controls read the PDF's existing
text layer: reported, never ranked. The catalogue of lanes lives in `catalog`.
"""

from __future__ import annotations

import base64
import html
import json
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf as fitz
import requests

from .catalog import LANES
from .config import (
    GARBLED_SHARE,
    IMAGE_REGION_MIN_SHARE,
    LAYER_MIN_CHARS,
    MAX_TOKENS,
    MIN_LETTER_SHARE,
    PICTURE_PAGE_SHARE,
    PROMPTS,
    REQUEST_TIMEOUT,
    TESSDATA,
    TESSERACT_LANGS,
    TESSERACT_TURKISH,
    VISION_TOOL,
)
from .corpus import picture_rects, written_by_ocr_tool
from .guards import degeneration
from .render import clean, crop, mime_for
from .text import normalize, run_text, sanitize_model_output

GLM_NATIVE = "Text Recognition:"
PADDLE_NATIVE = "OCR:"
GENERIC_PROMPT = (
    "Transcribe every piece of text in this document image, in reading order. "
    "Include the letterhead, any text inside logos, addresses, dates, tables, "
    "footers and footnotes. Output only the transcribed text."
)
OLMOCR_PROMPT = (
    "Below is the image of one page of a document. Return the natural reading order "
    "of the page as plain text, including headers, footers and footnotes."
)
# Each OCR specialist below gets the prompt its own model card gives for plain reading.
DOTS_NATIVE = "Extract the text content from this image."
DEEPSEEK_NATIVE = "Free OCR."
LIGHTON_NATIVE = ""  # the model card sends the image alone
OVIS_NATIVE = (
    "\nExtract all readable content from the image in natural human reading order and output the result as a single "
    "Markdown document. For charts or images, represent them using an HTML image tag: "
    '<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box '
    "coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. Preserve the original text without translation or paraphrasing."
)
# Unlimited-OCR's card puts the image before the instruction ("<image>document parsing."). The order matters:
# with the instruction first it skipped the page header and looped on the first synthetic page tried.
UNLIMITED_NATIVE = "document parsing."
# Prompt variants, each from the model's own card or code (docs/prompt-research-2026-10.md).
DEEPSEEK_LLAMA = "OCR"  # the llama.cpp pull request that added the model
DEEPSEEK_MARKDOWN = "<|grounding|>Convert the document to markdown."
DOTS_LAYOUT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""
QWEN_MARKDOWN = "qwenvl markdown"
QWEN_PLAIN = "Please output only the text content from the image without any additional descriptions or formatting."
QWEN_MAX_PIXELS = 2048 * 32 * 32  # its cookbook's upper bound
OLMOCR_NATIVE = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
)
OLMOCR_MAX_SIDE = 1288
LONG_ANSWER: dict[str, Any] = {"max_tokens": 8192}
CHANDRA_FACTOR = 192 / 300  # its maker renders at 192 dpi; pages here are 300 dpi
# Unlimited-OCR's own script forbids repeating a run of 35 tokens within the last 128. llama-server has no such
# rule; DRY penalises a repeat that grows past a length, steeply enough here to act as one. Line breaks must
# not restart the count, as its loops repeat whole lines; the server wants a breaker, so it gets one that never occurs.
UNLIMITED_DRY: dict[str, Any] = {
    "dry_multiplier": 4.0,
    "dry_base": 1.75,
    "dry_allowed_length": 34,
    "dry_penalty_last_n": 128,
    "dry_sequence_breakers": ["<|never|>"],
}
# Top 25% / middle / bottom 20%, each seam overlapped by 3% of the page.
BANDS = [(0.00, 0.28), (0.22, 0.83), (0.77, 1.00)]
BLOCK_PAD, BLOCK_CAP, SEAM_LINES = 0.01, 12, 3


# --- Model calls --------------------------------------------------------------


def vlm_call(
    api: str,
    model: str,
    prompt: str,
    image: bytes,
    mime: str,
    *,
    max_tokens: int = MAX_TOKENS,
    repeat_penalty: float | None = None,
    image_first: bool = False,
    sampling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One request. No retries: greedy decoding gives the same answer twice."""
    parts = [
        *([{"type": "text", "text": prompt}] if prompt else []),  # some OCR models take the image alone
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"}},
    ]
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        **({"repeat_penalty": repeat_penalty} if repeat_penalty else {}),
        **(sampling or {}),
        "messages": [{"role": "user", "content": parts[::-1] if image_first else parts}],
    }
    started = time.perf_counter()
    try:
        response = requests.post(f"{api}/chat/completions", json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
        return {
            "text": sanitize_model_output(str(content)),
            "finish_reason": str(data["choices"][0].get("finish_reason") or ""),
            "seconds": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:
        return {"text": "", "finish_reason": "", "seconds": time.perf_counter() - started, "error": type(exc).__name__}


def join_bands(texts: list[str]) -> str:
    """Concatenate band texts, dropping lines repeated across a seam."""
    merged: list[str] = []
    for text in texts:
        lines = text.splitlines()
        tail = {normalize(line) for line in merged[-SEAM_LINES:] if line.strip()}
        cut = 0
        for line in lines[:SEAM_LINES]:
            if line.strip() and normalize(line) in tail:
                cut += 1
            else:
                break
        merged.extend(lines[cut:])
    return "\n".join(merged)


def _area(b: tuple[float, ...]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _overlap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _union(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def merge_blocks(blocks: list[dict[str, Any]]) -> list[tuple[float, float, float, float]]:
    """Pad; make the crops non-overlapping; order top-to-bottom; merge down to the cap.

    Apple Vision reports some text more than once: a title that is also a paragraph, and a
    table whose cells are also listed as paragraphs. Cropping each would read that text two
    or three times, and every repeated number would score as invented. So a block mostly
    inside a bigger one is dropped, and blocks that still overlap are merged.
    """
    boxes = []
    for block in blocks:
        x0, y0, x1, y1 = block["box"]
        boxes.append(
            (max(0.0, x0 - BLOCK_PAD), max(0.0, y0 - BLOCK_PAD), min(1.0, x1 + BLOCK_PAD), min(1.0, y1 + BLOCK_PAD))
        )
    kept: list[tuple[float, float, float, float]] = []
    for box in sorted(boxes, key=_area, reverse=True):
        if not any(_overlap(box, k) >= 0.8 * _area(box) for k in kept):
            kept.append(box)
    merged = True
    while merged:
        merged = False
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                if _overlap(kept[i], kept[j]) > 0.3 * min(_area(kept[i]), _area(kept[j])):
                    kept[i] = _union(kept[i], kept.pop(j))
                    merged = True
                    break
            if merged:
                break
    boxes = sorted(kept, key=lambda b: (round(b[1], 2), b[0]))
    while len(boxes) > BLOCK_CAP:
        gaps = [boxes[i + 1][1] - boxes[i][3] for i in range(len(boxes) - 1)]
        i = gaps.index(min(gaps))
        a, b = boxes[i], boxes[i + 1]
        boxes[i : i + 2] = [(min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))]
    return boxes


VISION_TIMEOUT = 180  # seconds; a first call after a restart needs about 50 to rebuild its model


def vision(command: str, image: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.json"
        done = None
        for _ in range(2):  # Vision can hang waiting for the neural engine; seen for 54 minutes on one real scan
            try:
                done = subprocess.run(
                    [str(VISION_TOOL), command, str(image), str(out)],
                    capture_output=True,
                    check=False,
                    timeout=VISION_TIMEOUT,
                )
                break
            except subprocess.TimeoutExpired:
                done = None
        if done is None:
            return {
                "transcript": "",
                "words": [],
                "blocks": [],
                "error": f"vision_tool gave no answer within {VISION_TIMEOUT} s, twice",
            }
        if done.returncode or not out.exists():
            # Apple Vision refuses some images outright (seen on a tall real scan: CRImageReaderError).
            # That is this engine failing on one page, not a reason to end the whole run.
            return {
                "transcript": "",
                "words": [],
                "blocks": [],
                "error": done.stderr.decode("utf-8", "replace").strip()[-200:] or f"vision_tool exit {done.returncode}",
            }
        return json.loads(out.read_text(encoding="utf-8"))


def shrink(
    data: bytes, max_side: int | None = None, max_pixels: int | None = None, factor: float | None = None
) -> bytes:
    """PNG with the longest side at most `max_side`, or at most `max_pixels` in all, for a model trained on
    pages of that size."""
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    scale = factor if factor else max_side / max(h, w) if max_side else (max_pixels / (h * w)) ** 0.5
    if scale < 1:
        image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".png", image)[1].tobytes()


def whole(
    api: str,
    model: str,
    prompt: str,
    image: Path,
    *,
    max_side: int | None = None,
    max_pixels: int | None = None,
    factor: float | None = None,
    **call_options: Any,
) -> dict[str, Any]:
    data, mime = (
        (shrink(image.read_bytes(), max_side, max_pixels, factor), "image/png")
        if max_side or max_pixels or factor
        else (image.read_bytes(), mime_for(image))
    )
    call = vlm_call(api, model, prompt, data, mime, **call_options)
    return {"text": call["text"], "calls": [call], "prompt": prompt}


_REGION_RE = re.compile(r"^\s*(?:<\|det\|>)?([a-z_]+) ?\[\d+, ?\d+, ?\d+, ?\d+\](?:<\|/det\|>)?[ \t]?", re.MULTILINE)
_LATEX_MARK_RE = re.compile(r"\\\(\s*\^\{([^{}]*)\}\s*\\\)")


def strip_regions(text: str) -> str:
    """Unlimited-OCR writes a label and a box in front of every block ("text [99, 206, 415, 263]Ihre ...") and
    footnote marks as LaTeX ("\\( ^{2} \\)"). Neither is on the page; a picture's line holds nothing else."""
    lines = [
        line
        for line in text.splitlines()
        if not re.fullmatch(r"\s*(?:<\|det\|>)?(?:image|figure) ?\[[\d, ]+\](?:<\|/det\|>)?\s*", line)
    ]
    return _LATEX_MARK_RE.sub(r"\1", _REGION_RE.sub("", "\n".join(lines)))


def unlimited(api: str, model: str, image: Path, **call_options: Any) -> dict[str, Any]:
    """The call keeps the answer as the model gave it, boxes included; the lane's text is what it read."""
    result = whole(api, model, UNLIMITED_NATIVE, image, image_first=True, **call_options)
    return result | {"text": strip_regions(result["text"])}


_GROUNDING_RE = re.compile(
    r"<\|ref\|>.*?<\|/ref\|>|<\|det\|>.*?<\|/det\|>|^\s*[a-z_]+ ?\[\[[\d, \[\]]+\]\][ \t]?", re.MULTILINE
)
_FRONT_MATTER_RE = re.compile(r"\A\s*---\n.*?\n---\n?", re.DOTALL)
_JSON_TEXT_RE = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"')


def strip_grounding(text: str) -> str:
    """DeepSeek-OCR's grounding prompt names each block and its box before the block's text; a picture is a
    block with no text."""
    return re.sub(r"\n{3,}", "\n\n", html.unescape(_GROUNDING_RE.sub("", text))).strip()


def layout_text(answer: str) -> str:
    """dots.ocr's layout answer is a JSON list of blocks in reading order. An answer cut off mid-way is not
    valid JSON, so the blocks it does hold are read one by one. On llama-server the model often answers this
    prompt with the page's plain text and no JSON at all; that answer is kept as it is."""
    try:
        blocks = json.loads(answer)
        blocks = blocks if isinstance(blocks, list) else blocks.get("layout", [blocks])
        return "\n".join(str(b["text"]) for b in blocks if isinstance(b, dict) and b.get("text"))
    except (ValueError, AttributeError):
        found = _JSON_TEXT_RE.findall(answer)
        return "\n".join(json.loads(f'"{m}"') for m in found) if found else answer


def cleaned(result: dict[str, Any], clean_text: Callable[[str], str]) -> dict[str, Any]:
    """The call keeps the answer as the model gave it; the lane's text is what it read."""
    return result | {"text": clean_text(result["text"])}


def regions(
    api: str, model: str, prompt: str, image: Path, boxes, joiner: Callable[[list[str]], str], **call_options: Any
) -> dict[str, Any]:
    data = image.read_bytes()
    calls = [vlm_call(api, model, prompt, crop(data, box), "image/png", **call_options) | {"box": box} for box in boxes]
    return {"text": joiner([c["text"] for c in calls]), "calls": calls, "prompt": prompt}


def on_blocks(api: str, model: str, prompt: str, image: Path, **call_options: Any) -> dict[str, Any]:
    found = vision("blocks", image)
    blocks = found["blocks"]
    boxes = merge_blocks(blocks)
    result = (
        regions(api, model, prompt, image, boxes, "\n".join, **call_options)
        if boxes
        else whole(api, model, prompt, image, **call_options)
    )
    # Without blocks (none found, or Vision refused the image) the model reads the whole page.
    return (
        result
        | {"vision_blocks": len(blocks), "blocks_used": len(boxes)}
        | ({"vision_error": found["error"]} if found.get("error") else {})
    )


# --- Classic engines ----------------------------------------------------------


def unrotate_boxes(words: list[dict[str, Any]], skew: float, image: Path) -> list[dict[str, Any]]:
    """Map normalised word boxes from the deskewed frame back onto the original image."""
    if not skew or not words:
        return words
    import cv2
    import numpy as np

    from .render import page_size

    width, height = page_size(image.read_bytes())
    inverse = cv2.invertAffineTransform(cv2.getRotationMatrix2D((width / 2, height / 2), skew, 1.0))
    mapped = []
    for word in words:
        x0, y0, x1, y1 = (
            word["box"][0] * width,
            word["box"][1] * height,
            word["box"][2] * width,
            word["box"][3] * height,
        )
        corners = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]]) @ inverse.T
        mapped.append(
            dict(
                word,
                box=[
                    corners[:, 0].min() / width,
                    corners[:, 1].min() / height,
                    corners[:, 0].max() / width,
                    corners[:, 1].max() / height,
                ],
            )
        )
    return mapped


def parse_tsv(tsv: str) -> tuple[str, list[dict[str, Any]]]:
    """Tesseract TSV: text by line, and every word with a normalised top-left box."""
    rows = [line.split("\t") for line in tsv.splitlines()[1:]]
    page = next((r for r in rows if len(r) >= 12 and r[0] == "1"), None)
    width, height = (int(page[8]), int(page[9])) if page else (1, 1)
    lines: dict[tuple[str, str, str], list[str]] = {}
    words = []
    for r in rows:
        if len(r) < 12 or r[0] != "5" or not r[11].strip():
            continue
        left, top, w, h = map(int, r[6:10])
        lines.setdefault((r[2], r[3], r[4]), []).append(r[11])
        words.append({"text": r[11], "box": [left / width, top / height, (left + w) / width, (top + h) / height]})
    return "\n".join(" ".join(v) for v in lines.values()), words


def tesseract(image: Path, langs: tuple[str, ...] = TESSERACT_LANGS) -> dict[str, Any]:
    done = run_text(
        # TSV via a variable, not the named config "tsv": named configs live in Tesseract's own
        # tessdata/configs, which the tool's private tessdata folder does not have.
        [
            "tesseract",
            str(image),
            "stdout",
            "-l",
            "+".join(langs),
            "--psm",
            "1",
            "--tessdata-dir",
            str(TESSDATA),
            "-c",
            "tessedit_create_tsv=1",
        ],
        check=False,
    )
    text, words = parse_tsv(done.stdout)
    return {"text": text, "words": words, "calls": [], "error": done.stderr.strip()[-200:] if done.returncode else None}


def tesseract_clean(image: Path) -> dict[str, Any]:
    """Tesseract on the deskewed, sharpened page, with its word boxes put back on the page as it is."""
    cleaned, skew = clean(image.read_bytes())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cleaned.png"
        path.write_bytes(cleaned)
        result = tesseract(path) | {"skew_degrees": skew}
    # Words were found on the deskewed image; a searchable PDF places them on the page as it
    # is, so undo the deskew rotation on every box.
    result["words"] = unrotate_boxes(result.get("words", []), skew, image)
    return result


def apple_vision(image: Path, command: str) -> dict[str, Any]:
    """Apple Vision's transcript and word boxes; command "ocr-raw" turns language correction off."""
    data = vision(command, image)
    return {"text": data["transcript"], "calls": [], "words": data["words"]} | (
        {"error": data["error"]} if data.get("error") else {}
    )


_RAPID = None


def rapid_engine(detector: str = "mobile"):
    """PP-OCRv5 mobile, detector and latin recogniser, pinned rather than left to the default.

    The latin recogniser is the point: the default one is Chinese+English and folds umlauts
    and ß. RapidOCR 3.9 moved its default to PP-OCRv6, which has no latin model at all, so
    an unpinned version would break, or silently change results, on a library upgrade.
    """
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

    from .config import RAPIDOCR_MODELS

    return RapidOCR(
        params={
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.SERVER if detector == "server" else ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.lang_type": LangRec.LATIN,
            "Global.log_level": "error",
            "Global.model_root_dir": str(RAPIDOCR_MODELS),
        }
    )


_RAPID_SERVER = None


def rapidocr(image: Path, detector: str = "mobile") -> dict[str, Any]:
    """The server detector finds more and smaller text regions; the recogniser stays the latin
    mobile one, because PP-OCRv5 has no server recogniser that keeps umlauts."""
    global _RAPID, _RAPID_SERVER
    if detector == "server":
        if _RAPID_SERVER is None:
            _RAPID_SERVER = rapid_engine("server")
        engine = _RAPID_SERVER
    else:
        if _RAPID is None:
            _RAPID = rapid_engine()
        engine = _RAPID
    import cv2

    height, width = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE).shape
    result = engine(str(image))
    lines = []
    for box, text, score in zip(
        result.boxes if result.boxes is not None else [], result.txts or [], result.scores or []
    ):
        xs, ys = [float(p[0]) for p in box], [float(p[1]) for p in box]
        lines.append(
            {
                "text": text,
                "box": [min(xs) / width, min(ys) / height, max(xs) / width, max(ys) / height],
                "confidence": float(score),
            }
        )
    lines.sort(key=lambda line: (round(line["box"][1], 2), line["box"][0]))
    return {"text": "\n".join(line["text"] for line in lines), "calls": [], "lines": lines}


# Engines that need a heavy package of their own run as a worker in a separate environment:
# started once, then fed one image path per line. The tool's own environment stays as it is.
WORKER_PACKAGES = {"doctr": "onnxtr[cpu]", "easyocr": "easyocr"}
_WORKERS: dict[str, subprocess.Popen[str]] = {}


def worker_command(name: str) -> list[str]:
    return [
        "uv",
        "run",
        "--no-project",
        "--python",
        "3.11",
        "--with",
        WORKER_PACKAGES[name],
        "python",
        str(Path(__file__).resolve().parent / "workers" / f"{name}_worker.py"),
    ]


def worker(name: str, image: Path) -> dict[str, Any]:
    process = _WORKERS.get(name)
    if process is None or process.poll() is not None:
        process = subprocess.Popen(
            worker_command(name),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
        if process.stdout.readline().strip() != "ready":
            process.kill()
            raise RuntimeError(f"{name}: worker did not start")
        _WORKERS[name] = process
        import atexit

        atexit.register(process.terminate)
    process.stdin.write(f"{image}\n")
    process.stdin.flush()
    answer = process.stdout.readline()
    if not answer:
        raise RuntimeError(f"{name}: worker ended")
    data = json.loads(answer)
    return {
        "text": data.get("text", ""),
        "calls": [],
        **{k: data[k] for k in ("words", "lines") if k in data},
        **({"error": data["error"]} if data.get("error") else {}),
    }


def text_layer(source: Path, page: int, engine: str) -> dict[str, Any]:
    if source.suffix.lower() != ".pdf":
        return {"text": "", "calls": []}
    if engine == "pdftotext":
        done = run_text(["pdftotext", "-f", str(page), "-l", str(page), "-enc", "UTF-8", str(source), "-"], check=False)
        return {"text": done.stdout, "calls": [], "error": done.stderr.strip()[-200:] if done.returncode else None}
    with fitz.open(source) as pdf:
        return {"text": pdf[page - 1].get_text("text"), "calls": []}


# --- Using the born-digital text where possible ----------------------------------


def garbled(text: str) -> bool:
    """A layer that extracts as nonsense: missing font mappings show up as replacement or
    private-use characters and "(cid:NN)" codes, or as text that is barely letters."""
    visible = [c for c in text if not c.isspace()]
    if not visible:
        return True
    bad = sum(1 for c in visible if c == "\ufffd" or 0xE000 <= ord(c) <= 0xF8FF or ord(c) < 32) + 5 * text.count(
        "(cid:"
    )
    letters = sum(1 for c in visible if c.isalpha())
    return bad > GARBLED_SHARE * len(visible) or letters < MIN_LETTER_SHARE * len(visible)


def layer_verdict(source: Path, page: int) -> dict[str, Any]:
    """Can this page's text layer be used instead of OCR, and which pictures still need it?

    Per page, not per file: a PDF can mix born-digital and scanned pages. The layer is used
    only if it exists, was not written by an OCR tool (old OCR layers have the right length
    and garbage content), the page is not mostly picture, and the text is not garbled. Even
    then, text inside pictures (a logo, an image letterhead or footer) is absent from the
    layer, so those regions are listed for OCR.
    """
    if source.suffix.lower() != ".pdf":
        return {"use_layer": False, "reason": "image file, no text layer", "regions": []}
    with fitz.open(source) as pdf:
        pg = pdf[page - 1]
        ocr_layer = written_by_ocr_tool(pdf)
        text = pg.get_text("text")
        area = pg.rect.width * pg.rect.height
        pictures = picture_rects(pg)
        covered = sum(r.get_area() for r in pictures) / max(1.0, area)
        regions = [
            (
                (r.x0 - pg.rect.x0) / pg.rect.width,
                (r.y0 - pg.rect.y0) / pg.rect.height,
                (r.x1 - pg.rect.x0) / pg.rect.width,
                (r.y1 - pg.rect.y0) / pg.rect.height,
            )
            for r in pictures
            if r.get_area() >= IMAGE_REGION_MIN_SHARE * area
        ]
    chars = len(text.strip())
    # "Is there a layer" before "who wrote it": a scanner's producer string names the scanner
    # even when the file has no text at all.
    if chars < LAYER_MIN_CHARS:
        return {"use_layer": False, "reason": "no usable text layer", "regions": []}
    if ocr_layer:
        return {"use_layer": False, "reason": "layer written by an OCR tool", "regions": []}
    if covered >= PICTURE_PAGE_SHARE:
        return {"use_layer": False, "reason": "page is mostly picture", "regions": []}
    if garbled(text):
        return {"use_layer": False, "reason": "text layer is garbled", "regions": []}
    return {"use_layer": True, "reason": "trustworthy text layer", "regions": regions}


def auto(api: str | None, model: str, prompt: str, image: Path, source: Path, page: int) -> dict[str, Any]:
    verdict = layer_verdict(source, page)
    if not verdict["use_layer"]:
        result = whole(api, model, prompt, image)
        return result | {"used_layer": False, "rule": verdict["reason"]}
    layer = text_layer(source, page, "pdftotext")
    calls = []
    if verdict["regions"]:
        data = image.read_bytes()
        calls = [
            vlm_call(api, model, prompt, crop(data, box), "image/png") | {"box": box} for box in verdict["regions"]
        ]
    pictures = "\n".join(c["text"] for c in calls if c["text"].strip())
    return {
        "text": "\n".join(filter(None, [layer["text"], pictures])),
        "calls": calls,
        "prompt": prompt,
        "used_layer": True,
        "rule": verdict["reason"],
        "picture_regions": len(calls),
    }


# --- Dispatch -----------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """One page image for one lane: the image, the file and page it came from, and the model
    server (api) and model name for lanes that use one."""

    image: Path
    source: Path
    page: int
    api: str | None
    model: str


def glm_custom_prompt() -> str:
    return (PROMPTS / "glm_custom.txt").read_text(encoding="utf-8")


def chandra_prompt() -> str:
    return (PROMPTS / "chandra_ocr.txt").read_text(encoding="utf-8")


def chandra_layout_prompt() -> str:
    return (PROMPTS / "chandra_ocr_layout.txt").read_text(encoding="utf-8")


BAND_BOXES = [(0.0, top, 1.0, bottom) for top, bottom in BANDS]

# The "tuned" lanes change only how a model is asked, after the first full run showed what the shared
# settings cost: answers cut off at 3,072 tokens counted as unusable on 4 to 6 % of pages, no model had a
# repetition penalty, and LightOnOCR was given pages three times the size it was trained on.
TUNED: dict[str, Any] = {"max_tokens": 8192, "repeat_penalty": 1.05}
LIGHTON_MAX_SIDE = 1540  # its card: render at 200 dpi, longest side 1540 px


RUNNERS: dict[str, Callable[[Job], dict[str, Any]]] = {
    "lighton-tuned": lambda j: whole(j.api, j.model, LIGHTON_NATIVE, j.image, max_side=LIGHTON_MAX_SIDE, **TUNED),
    "ovis-tuned": lambda j: whole(j.api, j.model, OVIS_NATIVE, j.image, **TUNED),
    "paddle-blocks-tuned": lambda j: on_blocks(j.api, j.model, PADDLE_NATIVE, j.image, **TUNED),
    "chandra-tuned": lambda j: whole(j.api, j.model, chandra_prompt(), j.image, **TUNED),
    "qwen8b-tuned": lambda j: whole(j.api, j.model, GENERIC_PROMPT, j.image, **TUNED),
    "unlimited-tuned": lambda j: unlimited(j.api, j.model, j.image, **TUNED),
    "unlimited-dry": lambda j: unlimited(j.api, j.model, j.image, **TUNED, sampling=UNLIMITED_DRY),
    "deepseek-first": lambda j: whole(j.api, j.model, DEEPSEEK_NATIVE, j.image, image_first=True),
    "deepseek-ocr": lambda j: whole(j.api, j.model, DEEPSEEK_LLAMA, j.image),
    "deepseek-md": lambda j: cleaned(
        whole(j.api, j.model, DEEPSEEK_MARKDOWN, j.image, image_first=True, **LONG_ANSWER), strip_grounding
    ),
    "dots-layout": lambda j: cleaned(whole(j.api, j.model, DOTS_LAYOUT, j.image, **LONG_ANSWER), layout_text),
    "qwen8b-md": lambda j: whole(j.api, j.model, QWEN_MARKDOWN, j.image),
    "qwen8b-plain": lambda j: whole(j.api, j.model, QWEN_PLAIN, j.image),
    "qwen8b-2mp": lambda j: whole(j.api, j.model, GENERIC_PROMPT, j.image, max_pixels=QWEN_MAX_PIXELS),
    "chandra-192dpi": lambda j: whole(j.api, j.model, chandra_prompt(), j.image, factor=CHANDRA_FACTOR, **TUNED),
    "chandra-2mp": lambda j: whole(j.api, j.model, chandra_prompt(), j.image, max_pixels=QWEN_MAX_PIXELS, **TUNED),
    "chandra-layout": lambda j: whole(j.api, j.model, chandra_layout_prompt(), j.image, **LONG_ANSWER),
    "ovis-first": lambda j: whole(j.api, j.model, OVIS_NATIVE, j.image, image_first=True),
    "olmocr-native": lambda j: cleaned(
        whole(j.api, j.model, OLMOCR_NATIVE, j.image, max_side=OLMOCR_MAX_SIDE),
        lambda text: _FRONT_MATTER_RE.sub("", text),
    ),
    "glm-custom": lambda j: whole(j.api, j.model, glm_custom_prompt(), j.image),
    "glm": lambda j: whole(j.api, j.model, GLM_NATIVE, j.image),
    "glm-bands": lambda j: regions(j.api, j.model, GLM_NATIVE, j.image, BAND_BOXES, join_bands),
    "glm-blocks": lambda j: on_blocks(j.api, j.model, GLM_NATIVE, j.image),
    "paddle": lambda j: whole(j.api, j.model, PADDLE_NATIVE, j.image),
    "paddle-blocks": lambda j: on_blocks(j.api, j.model, PADDLE_NATIVE, j.image),
    "qwen8b": lambda j: whole(j.api, j.model, GENERIC_PROMPT, j.image),
    "qwen4b": lambda j: whole(j.api, j.model, GENERIC_PROMPT, j.image),
    "olmocr": lambda j: whole(j.api, j.model, OLMOCR_PROMPT, j.image),
    "dots": lambda j: whole(j.api, j.model, DOTS_NATIVE, j.image),
    "deepseek": lambda j: whole(j.api, j.model, DEEPSEEK_NATIVE, j.image),
    "lighton": lambda j: whole(j.api, j.model, LIGHTON_NATIVE, j.image),
    "chandra": lambda j: whole(j.api, j.model, chandra_prompt(), j.image),
    "ovis": lambda j: whole(j.api, j.model, OVIS_NATIVE, j.image),
    "unlimited": lambda j: unlimited(j.api, j.model, j.image),
    "vision": lambda j: apple_vision(j.image, "ocr"),
    "vision-raw": lambda j: apple_vision(j.image, "ocr-raw"),
    "tesseract": lambda j: tesseract(j.image),
    "tesseract-clean": lambda j: tesseract_clean(j.image),
    "tesseract-tur": lambda j: tesseract(j.image, (*TESSERACT_LANGS, TESSERACT_TURKISH)),
    "rapidocr": lambda j: rapidocr(j.image),
    "rapidocr-server": lambda j: rapidocr(j.image, "server"),
    "doctr": lambda j: worker("doctr", j.image),
    "easyocr": lambda j: worker("easyocr", j.image),
    "auto-paddle": lambda j: auto(j.api, j.model, PADDLE_NATIVE, j.image, j.source, j.page),
    "auto-qwen4b": lambda j: auto(j.api, j.model, GENERIC_PROMPT, j.image, j.source, j.page),
    "pdftotext": lambda j: text_layer(j.source, j.page, "pdftotext"),
    "pymupdf": lambda j: text_layer(j.source, j.page, "pymupdf"),
}


def run_lane(lane_id: str, image: Path, source: Path, page: int, api: str | None = None) -> dict[str, Any]:
    lane = LANES[lane_id]
    started = time.perf_counter()
    result = RUNNERS[lane_id](Job(image, source, page, api, lane.model or ""))
    calls = result.get("calls", [])
    return {
        **result,
        "lane": lane_id,
        "seconds": time.perf_counter() - started,
        "requests": len(calls),
        "errors": [c["error"] for c in calls if c.get("error")] + ([result["error"]] if result.get("error") else []),
        "degenerate": degeneration(
            result["text"], result.get("prompt", ""), [c.get("finish_reason", "") for c in calls]
        ),
    }
