---
type: app
parent:
  - "[[OCR Bakeoff]]"
location:
  - mac
repo:
  - "[[tool-ocr-bakeoff]]"
command: uv run ocr-bakeoff
---

## Purpose

How the tool runs: `uv run ocr-bakeoff` from the checkout, on a Mac with Homebrew (Apple Silicon runs every model on the GPU), or on Linux without the Apple Vision engines.

## How it runs

```bash
uv run ocr-bakeoff samples ./demo     # a synthetic test folder: letters, invoices, slides, specs, scans
uv run ocr-bakeoff ./demo             # first run lists what it needs, asks once, installs, runs
uv run ocr-bakeoff lanes              # every engine and whether it is ready
uv run ocr-bakeoff selftest           # offline checks; never downloads, live engines only if installed
uv run ocr-bakeoff profile ~/Downloads # survey a folder's layouts, counts only (see below)
```

A full run of every engine over a few dozen documents takes hours. Keep the Mac awake with
`caffeinate -i uv run ocr-bakeoff ./testfiles`. Only one run may work on a folder at a time (a
lock refuses a second one), and an interrupted run resumes where it stopped: every finished page
is cached.

First run installs what the chosen engines need, after showing each item with its size:
`llama.cpp`, `tesseract` and `poppler` via Homebrew; Tesseract `deu`/`eng` data from
`tessdata_best`; model weights into the Hugging Face cache (shared with anything else that uses
it); RapidOCR's PP-OCRv5 models into `~/.cache/ocr-bakeoff/rapidocr` (RapidOCR fetches these from
ModelScope), so a new checkout or virtual environment reuses them. `--yes` skips the question. `--lanes llm` or `--skip qwen8b,olmocr`
narrows the run.

| Command | Feature |
|---|---|
| `ocr-bakeoff <folder>` | [[Compare OCR Engines]] |
| `ocr-bakeoff samples <dir>` | [[Make Sample Folder]] |
| `ocr-bakeoff synth <dir> --profile …` | [[Make Look-alikes]] |
| `ocr-bakeoff profile <folder>` | [[Profile a Folder]] |
| `ocr-bakeoff pick <dir> --profile …` | [[Pick Real Documents]] |
| `ocr-bakeoff lanes`, `setup` | [[Show Lanes]] |
| `ocr-bakeoff selftest` | the offline checks; part of the gate |

## Rules

1. **Requirements:** macOS with Homebrew is the tested path (Apple Silicon: every model runs on the GPU). Linux works
for everything but the Apple Vision engines; install `poppler-utils`, `tesseract-ocr` and
`llama.cpp` yourself. Python 3.11+ and [uv](https://docs.astral.sh/uv/).

Model licences are each model's own; check them before redistributing weights.
2. **The gate is `./scripts/ci.sh`:** ruff, then `uv run ocr-bakeoff selftest`, whose last line `selftest: N/N checks passed` is the merge queue's `GATE_OK`. The selftest never downloads; live checks of Tesseract, RapidOCR and Apple Vision run only when that engine is installed.
3. **RapidOCR's models live in `~/.cache/ocr-bakeoff/rapidocr`**, not in the virtual environment, so a fresh clone neither downloads them nor depends on ModelScope (2026-10-25).
4. **No deploy:** a merge is live once the main checkout holds it.

## Difference to today

None.
