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
  - uv run ocr-bakeoff <folder>
---

## Purpose

Runs the chosen engines over the same page images of a [[Test Folder]] and reports which reads it best, per engine and per document kind.

## Flow

```mermaid
flowchart TD
    A["ocr-bakeoff &lt;folder&gt;"] --> L{"another run holds the folder's lock?"}
    L -->|yes| X["refused"]
    L -->|no| D{"engines need something not installed?"}
    D -->|yes| ASK["list each item with its size; install on yes or --yes"]
    D -->|no| R["render pages (300 dpi), born-digital also as a simulated scan"]
    ASK --> R
    R --> E["each lane on each page image; finished pages cached"]
    E --> S["Score Output against the references"]
    S --> W["Pick the Winner"]
    W --> O["terminal and summary.md: aggregates only"]
```

## How it runs

`uv run ocr-bakeoff <folder> [--lanes ids|groups] [--skip ids] [--reference consensus|gdrive] [--baseline glm-custom] [--yes] [--force]`. A full run of every engine over a few dozen documents takes hours: `caffeinate -i uv run ocr-bakeoff ./testfiles`.

## Rules

1. **One run per folder:** a lock refuses a second one, and an interrupted run resumes where it stopped.
2. **Every model runs on its own `llama-server` with identical settings:** one slot, 16K context, temperature 0, 6144 image tokens. `--glm-url` reuses a GLM-OCR server you already run.
3. **Re-runs reuse everything already computed;** `--force` recomputes. If one engine's outputs come from different model files or versions, the table says so.
4. **The engines** are the [[Engine Lanes]]; `--lanes llm`, `classic`, `controls`, `pipelines`, `variants` or ids pick them.
5. **What it writes** is the [[Results Folder]].

## Difference to today

None.
