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

A local, automatic reference for scans without a `.gold.txt` (`--reference consensus`).

## Rules

**`--reference consensus`:** local and automatic. Each engine is scored against what at least two
of the *other* panel engines agree on (Qwen3-VL-8B, Apple Vision, Tesseract, RapidOCR — four
families), so no engine agrees with itself. Header and footer references come from the members that
report positions; there is no word order to agree on, so no CER. It checks itself: on every page
that does have a true reference, each engine is also scored against the consensus, and the report
prints both side by side. On 50 such page images it gave the same drop verdict on 91–97% of engine
× page pairs and ranked engines in largely the true order (Spearman 0.84–0.89). **It flatters strong
engines by 1–3 drops per 20 pages**, because what a good engine misses the others often miss too.
It separates good from bad, not the leaders from one another.

## Difference to today

None.
