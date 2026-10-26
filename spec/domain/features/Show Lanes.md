---
type: feature
parent:
  - "[[OCR Bakeoff]]"
location:
  - mac
repo:
  - "[[tool-ocr-bakeoff]]"
mac_runs: by hand
mac_command:
  - uv run ocr-bakeoff lanes
  - uv run ocr-bakeoff setup
---

## Purpose

Lists every engine ([[Engine Lanes]]) and whether it is ready here; `setup` installs what the chosen lanes need, after showing each item with its size.

## Rules

1. **What a first run installs:** `llama.cpp`, `tesseract` and `poppler` via Homebrew; Tesseract `deu`/`eng` data from `tessdata_best`; model weights into the Hugging Face cache (shared with anything else that uses it); RapidOCR's PP-OCRv5 models into `~/.cache/ocr-bakeoff/rapidocr`.
2. **`doctr` and `easyocr` run as workers in a separate environment** (`uv run --with …`), started with their first page, so the tool itself needs neither OnnxTR nor PyTorch; their packages and models are fetched on first use and are not in the first-run list yet.

## Difference to today

`doctr` and `easyocr` are not in the first-run list (rule 2).
