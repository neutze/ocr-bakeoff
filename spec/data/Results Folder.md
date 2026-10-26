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

Everything a run writes, beside the test folder, so a rerun or a re-score reuses it.

## Format

Results go to `<testfolder>/.ocr-bakeoff/`:

| file | what |
|---|---|
| `summary.md`, `summary.json` | the tables, aggregates only; the JSON is for comparing runs |
| `per_page.csv` | every engine on every page image; names your files, so keep it with them |
| `out/<engine>/<page>.json` | each engine's raw output, with its seconds, and its **provenance**: model file and llama.cpp build, or engine version |
| `runs/<run id>.json` | the machine, every tool version, and what else was running (other llama servers, free memory) when each model started |
| `pages/`, `gold/`, `logs/` | the page images, the answer keys, each model server's log |

Re-runs reuse everything already computed, and re-scoring alone takes seconds; `--force`
recomputes. If one engine's outputs come from different model files or versions, the table says so.

## Rules

1. **Document text stays here**, never on stdout ([[OCR Bakeoff]], rule 2). `per_page.csv` names your files: keep it with them.

## Difference to today

None.
