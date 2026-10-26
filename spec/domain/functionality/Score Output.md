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

Scores each engine's output on one page image against its reference, on what breaks OCR in a document archive.

## Rules

| column | meaning |
|---|---|
| drops | pages where header or footer recall fell below 0.8 (on a region of 5+ words) |
| header, footer | share of the reference's header and footer words found |
| numbers, invented | share of the reference's numbers read exactly; numbers in the output that are not on the page |
| äöüß, çğış | German and Turkish special letters kept exactly |
| text F1 | word overlap, blind to reading order; CER beside it also punishes a different order |
| degenerate | loops, echoed prompts, output cut at the cap, empty pages. Loops only count for generative models, and only if the page itself does not repeat that way |
| word boxes | share of visible words read **and** placed where they are, on the simulated scans (IoU ≥ 0.3), which is what a searchable PDF/A needs |

References for born-digital pages keep only text you can see: real text layers carry invisible
text (under pictures, white on white, off the page), and an engine that rightly ignores it is not
charged for it.

## Difference to today

None.
