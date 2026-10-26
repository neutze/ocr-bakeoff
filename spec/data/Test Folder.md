---
type: data
parent:
  - "[[OCR Bakeoff]]"
used_by:
  - "[[Compare OCR Engines]]"
location:
  - mac
---

## Purpose

The folder of documents a run compares the engines on, with the references that score them.

## Format

```
testfiles/
  letters/          the subfolder name is the document kind; results are broken down by it
    a.pdf
    a.gold.txt      optional: reference text for a.pdf page 1 (a.p2.gold.txt for page 2)
  presentations/
  scans/
    receipt.jpg
    receipt.gold.txt
```

- **Born-digital PDFs** (a real text layer, not written by an OCR tool) need nothing: their own
  text layer is the reference. Each is scored twice — as rendered, and as a simulated scan.
- **Scans** need a reference: a `.gold.txt` next to the file, `--reference consensus`, or
  `--reference gdrive` (see *References for scans*). Without one they are run but only scored for
  degeneration and speed.
- **Text that exists only in an image** — a logo, an image letterhead or footer on an otherwise
  born-digital page — is not in the text layer. Add it to a `.gold.txt`; it is joined to the
  layer's header or footer.

`.gold.txt` sections (all optional; `#` starts a comment):

```
[HEADER]
Everything above the first body paragraph
[FOOTER]
Footer, footnotes, page numbers
[NUMBERS]
Every number on the page, exactly as printed
[TEXT]
The whole page, if you want character error rates on scans
```

## Difference to today

None.
