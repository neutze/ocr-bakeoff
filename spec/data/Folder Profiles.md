---
type: data
parent:
  - "[[OCR Bakeoff]]"
used_by:
  - "[[Profile a Folder]]"
  - "[[Make Look-alikes]]"
  - "[[Pick Real Documents]]"
location:
  - mac
---

## Purpose

The shapes of a profiled folder's PDFs, with their paths.

## Format

`~/.cache/ocr-bakeoff/profiles/<folder name>.json`, mode 600: one record per file (path, paper format, born-digital or scan, producing application, header and footer as text or image, columns, table rules, font sizes, language).

## Rules

1. **Private:** never print a path, a file name or document text from it; the terminal gets counts only.

## Difference to today

None.
