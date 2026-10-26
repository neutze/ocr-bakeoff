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
  - uv run ocr-bakeoff profile <folder>
---

## Purpose

Surveys a folder's PDFs locally and records only their shape, for [[Make Look-alikes]] and [[Pick Real Documents]].

## Rules

1. **Shape only:** paper format, born-digital or scan, producing application, header and footer as text or image, columns, table rules, font sizes, language. No text, no names.
2. **Counts only on the terminal.** The per-file profiles, with paths, go to [[Folder Profiles]] (mode 600); never print a path, a file name or document text from them.
3. **A sample** of `--sample` files (default 2,000).

## Difference to today

None.
