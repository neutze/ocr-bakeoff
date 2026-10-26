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

Turns the scores into a verdict: a winner, a tie group, or a trade-off.

## Rules

1. Disqualify an engine that degenerates on more than 2% of page images, or drops a header or
   footer (recall below 0.8 on a region of at least 5 words) on more than 10%.
2. Rank the rest by drops, then number recall, then fewest invented numbers, then error rate.
3. Every engine fewer than 2 page images from the leader on every count ties with it. A tie is
   reported as a group with its fastest member, never as a single winner, and a tie with the
   baseline (`--baseline`, default `glm-custom`) never justifies switching away from it.
4. A winner more than 2x slower than the baseline is reported as a trade-off.

Drops are judged on the hard images only: simulated scans of born-digital pages, and real scans.

## Difference to today

None.
