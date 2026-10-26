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
  - uv run ocr-bakeoff synth <dir> --profile <file>
---

## Purpose

Writes documents with the same shapes as a profiled folder ([[Folder Profiles]]) and invented content, each with an exact reference, so a test set can be shared.

## Rules

1. **Content is invented:** German, English and Turkish letters, invoices, presentations, work documents, and scans of letters, invoices and forms at varied resolution, skew and noise, a few with a junk old OCR layer. Nothing from a profiled file but its structure.
2. **Every document has an exact reference**, scans included, because the generator knows every word it drew.
3. **At least 4 per kind, 6 image headers and 4 image footers**, whatever the profile says. `--count` (default 30), `--seed`.
4. **Look-alikes are cleaner than real paper:** treat a look-alike result as the first cut and confirm the leaders with [[Pick Real Documents]].

## Difference to today

None.
