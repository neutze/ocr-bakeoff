---
type: functionality
parent:
  - "[[Compare OCR Engines]]"
location:
  - mac
repo:
  - "[[tool-ocr-bakeoff]]"
---

## Purpose

A reference for scans from Google Drive's own OCR (`--reference gdrive`), the one path that sends page images off the machine, and only when you ask for it.

## Rules

**`--reference gdrive`** uploads each scan's image — and each born-digital page, to measure Drive's
own accuracy against the exact text layer — to **your** Google Drive, as Google Docs, which is
how Drive runs OCR. Header and footer references come from separate uploads of the top and
bottom 20% of the page, cut at the emptiest pixel row so no line is sliced. The tool asks for
the narrowest Drive scope (`drive.file`: only files it created), moves each temporary Doc to the
Drive trash after reading it, and caches every result.

First use needs an OAuth client: in Google Cloud Console enable the Drive API, create an OAuth
client ID of type *Desktop app*, download the JSON, and pass `--gdrive-credentials path.json`.

A reference produced by an OCR engine measures agreement with that engine. The table prints
Drive's own score on born-digital pages so you can judge how far to trust the scan columns.

## Difference to today

None.
