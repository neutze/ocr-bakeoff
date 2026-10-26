---
type: data
parent:
  - "[[OCR Bakeoff]]"
used_by:
  - "[[Compare OCR Engines]]"
  - "[[Show Lanes]]"
location:
  - mac
---

## Purpose

Every engine the tool can run, one lane each, with its kind (`catalog.py`).

## Format

| id | engine | kind |
|---|---|---|
| `glm-custom` | GLM-OCR 0.9B, whole page, a 39-line custom prompt | vision LLM |
| `glm` | GLM-OCR, whole page, its native `Text Recognition:` prompt | vision LLM |
| `glm-bands` | GLM-OCR on three fixed page bands | vision LLM |
| `glm-blocks` | GLM-OCR on Apple Vision's text blocks (macOS) | vision LLM |
| `paddle` / `paddle-blocks` | PaddleOCR-VL-1.6 0.9B, whole page / on Vision blocks | vision LLM |
| `qwen8b`, `qwen4b` | Qwen3-VL 8B / 4B Instruct | vision LLM |
| `olmocr` | olmOCR-2-7B | vision LLM |
| `dots` | dots.ocr 1.7B, its plain-OCR prompt (leaves page headers and footers out by training) | vision LLM |
| `deepseek` | DeepSeek-OCR, its free-OCR prompt | vision LLM |
| `lighton` | LightOnOCR-2-1B, the image alone | vision LLM |
| `chandra` | Chandra-OCR-2, its OCR prompt (community GGUF) | vision LLM |
| `ovis` | OvisOCR2, its Markdown prompt (community GGUF) | vision LLM |
| `unlimited`, `unlimited-tuned` | Unlimited-OCR (Baidu, built on DeepSeek-OCR), its `document parsing.` prompt with the image placed before it; the region labels and boxes it writes in front of every block are removed before scoring (community GGUF). `-tuned`: longer answers, repetition penalty | vision LLM |
| `unlimited-dry` | Unlimited-OCR as in `unlimited-tuned`, with llama-server's DRY sampling set to stand in for the maker's rule against repeating a 35-token run within 128 tokens | vision LLM, prompt variant |
| `deepseek-first`, `deepseek-ocr`, `deepseek-md` | DeepSeek-OCR asked differently: the image before `Free OCR.`; llama.cpp's short prompt `OCR`; its grounding prompt for Markdown after the image, boxes removed, longer answers | vision LLM, prompt variant |
| `dots-layout` | dots.ocr with its full layout prompt, which returns page headers and footers; the JSON is turned into text; longer answers | vision LLM, prompt variant |
| `qwen8b-md`, `qwen8b-plain`, `qwen8b-2mp` | Qwen3-VL-8B with its trained keyword `qwenvl markdown`; with its cookbook's plain prompt; with this tool's prompt on a page shrunk to 2.1 megapixels | vision LLM, prompt variant |
| `chandra-layout` | Chandra-OCR-2 with its default layout prompt, which labels page headers and footers; longer answers | vision LLM, prompt variant |
| `chandra-192dpi`, `chandra-2mp` | Chandra-OCR-2 as in `chandra-tuned`, on a page shrunk to its maker's 192 dpi, or to 2.1 megapixels as in `qwen8b-2mp` | vision LLM, prompt variant |
| `ovis-first` | OvisOCR2 with the image before its prompt | vision LLM, prompt variant |
| `olmocr-native` | olmOCR-2 with its maker's exact prompt on a page of 1,288 px; the YAML block it starts with is removed | vision LLM, prompt variant |
| `vision`, `vision-raw` | Apple Vision, language correction on / off (macOS) | classic |
| `tesseract`, `tesseract-clean` | Tesseract 5 `deu+eng`, raw / deskewed and sharpened | classic |
| `tesseract-tur` | Tesseract 5 `deu+eng+tur`, raw: the same engine with Turkish language data | classic |
| `rapidocr` | RapidOCR, PP-OCRv5 mobile with the latin recogniser | classic |
| `rapidocr-server` | RapidOCR, PP-OCRv5 server detector with the same latin recogniser | classic |
| `doctr` | docTR through OnnxTR, default models; word boxes | classic, own environment |
| `easyocr` | EasyOCR, German + English + Turkish, CPU | classic, own environment |
| `pdftotext`, `pymupdf` | the PDF's existing text layer | control, never ranked |

The prompt variants each change one thing against the model's first lane; why each exists is in
`docs/prompt-research-2026-10.md`. `--lanes variants` runs them all.

`doctr` and `easyocr` run as workers in a separate environment (`uv run --with …`), started with their first page,
so the tool itself needs neither OnnxTR nor PyTorch. That first page's timing includes the start-up. Their packages
and models are fetched on first use and are not in the first-run list yet.

Speed per page image, M2 Max, measured on the samples (median; band and block lanes make several
calls per page):

| engine | s/page | | engine | s/page |
|---|---|---|---|---|
| `vision` | 0.4 | | `glm-bands` | 22.7 |
| `rapidocr` | 0.9 | | `glm-custom` | 39.0 |
| `tesseract` | 1.6 | | `qwen4b` | 47.1 |
| `glm-blocks` | 4.3 | | `olmocr` | 71.9 |
| `paddle-blocks` / `paddle` | 6.1 / 6.8 | | `qwen8b` | 74.0 |

`olmocr` drops most headers and footers — consistent with its training, which aims at a document's
reading text rather than page furniture. It is here to show that, not as a contender for this job.

Every model runs on its own `llama-server` with identical settings: one slot, 16K context,
temperature 0, 6144 image tokens. `--glm-url` reuses a GLM-OCR server you already run.

## Model licences

The tool ships no weights ([[OCR Bakeoff]] rule 7). Each model's licence, from its Hugging Face page on 2026-10-26:

| licence | models |
|---|---|
| MIT | GLM-OCR, dots.ocr, DeepSeek-OCR, Unlimited-OCR |
| Apache 2.0 | PaddleOCR-VL-1.6, Qwen3-VL 4B and 8B, olmOCR-2, LightOnOCR-2, OvisOCR2 |
| OpenRAIL-M, modified by Datalab | Chandra-OCR-2: free for personal use and research; commercial use above 2 million USD revenue needs Datalab's licence |

HunyuanOCR was a lane until 2026-10-26 and was dropped (owner: "drop"): Tencent's licence does not apply in the EU, the UK or South Korea, so the tool cannot offer it to everyone.

The classic engines are Apache 2.0 (Tesseract, RapidOCR, docTR/OnnxTR, EasyOCR); Apple Vision is part of macOS.

## Difference to today

None.
