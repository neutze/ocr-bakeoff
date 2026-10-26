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
  - uv run ocr-bakeoff pick <dir> --profile <file>
---

## Purpose

Copies real documents into a [[Test Folder]], spread across producing applications, languages and header styles, to check the look-alike leaders on real paper without reading anything.

## Rules

1. **Born-digital PDFs by default** (their own text layer is the reference), as `real/<kind>/r001.pdf` …; `--tier scan` picks real scans instead (a quarter with an old OCR layer), to be scored with `--reference consensus`.
2. **The mapping back to the originals stays in `real/.pick.json`** (mode 600); the terminal gets counts.
3. **`--language tr`** picks only documents whose profile found that language; **`--exclude <dir>`** (repeatable) keeps a second sample free of documents an earlier one used, and no content is picked twice under two names. `--per-kind` (default 5), `--seed`.

## Difference to today

None.
