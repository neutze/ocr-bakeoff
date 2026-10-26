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

The per-page rule for when a born-digital PDF's own text layer can be trusted, measured by the `auto-*` pipelines.

## Rules

In production most PDFs need no OCR at all: their own text layer is exact. The `auto-*` pipelines
measure a per-page rule for when to trust it:

| rule | then |
|---|---|
| image file, or under 40 characters of text | OCR the page |
| layer written by an OCR tool (by its producer) | re-OCR: an old OCR layer has the right length and garbage content |
| page is at least half picture | OCR the page |
| layer garbled (replacement or private-use characters, `(cid:)` codes, too few letters) | OCR the page |
| otherwise | use the layer, and OCR only the pictures on the page — logos, image letterheads and footers |

`auto-paddle` and `auto-qwen4b` apply it with each leading engine. They run on pages as they are
(born-digital as rendered, and real scans), get their own section in the table next to
`pdftotext` alone, and are never ranked against single engines.

## Difference to today

None.
