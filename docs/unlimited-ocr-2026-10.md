# Unlimited-OCR on the October 2026 test

2026-10-12. Aggregates only; no document text or file name.

## Verdict

Unlimited-OCR, as it can be run on this Mac, is not a candidate for the OCR worker. On real scans it leaves
43 % of pages unusable (looped, cut off or empty) where Qwen3-VL-8B leaves 1 % and RapidOCR 3 %, and it loses
a header or footer on 34 % of pages where they lose 8 % and 3 %. It is clearly better than DeepSeek-OCR, the
model it is built on, and about five times faster than Qwen3-VL-8B.

## How it was run

- **Weights:** `richardyoung/Unlimited-OCR-GGUF`, `Unlimited-OCR-Q8_0.gguf` with `mmproj-Unlimited-OCR-F16.gguf`
  (3.9 GB), a community conversion. The official weights run on NVIDIA only (transformers, vLLM, SGLang).
- **Runtime:** stock llama-server from Homebrew (build 11429), the first choice. MLX and transformers were not
  needed. One slot, 16K context, temperature 0, 6,144 image tokens, as for every other model. Its own ports
  (11376, 11377); the servers already running were left alone.
- **Prompt:** the maker's `document parsing.`, placed after the image. With the prompt first the model skipped
  the page header and looped. The label and box it writes before every block are removed before scoring.
- **Lanes:** `unlimited` (answers up to 3,072 tokens, like every base lane) and `unlimited-tuned` (8,192 tokens,
  repetition penalty 1.05, like the other tuned lanes).
- **Test:** `tool-ocr-bakeoff` on the two October 2026 sets: 95 real born-digital documents, each as rendered
  and as a simulated scan, and 100 real scans. The other engines' figures are their stored results from
  2026-10-04 to 2026-10-08, scored again by the same code in the same run.

## Results

95 % intervals in brackets. "Lost" is a page whose header or footer recall fell below 0.8.

### Real scans, 100 page images (answer key: what at least two other engines agree on)

| engine | header or footer lost | looped, cut off or empty | numbers found | German letters | word overlap | s per page, median | 90th percentile | peak memory |
|---|---|---|---|---|---|---|---|---|
| RapidOCR, server detector | 2 % (0–5) | 3 % (0–7) | 92.3 % | 97.2 % | 92.0 % | 3.2 | 4.9 | n/a |
| RapidOCR | 3 % (0–7) | 3 % (0–7) | 90.5 % | 95.0 % | 90.4 % | 0.5 | 1.3 | n/a |
| Chandra-OCR-2, tuned | 2 % (0–5) | 1 % (0–3) | 91.1 % | 94.1 % | 90.1 % | 67.8 | 96.1 | 14 GB |
| Qwen3-VL-8B | 8 % (3–14) | 1 % (0–3) | 83.1 % | 97.2 % | 93.0 % | 66.6 | 87.0 | 17 GB |
| Qwen3-VL-8B, tuned | 8 % (3–14) | 1 % (0–3) | 83.8 % | 97.3 % | 94.2 % | 53.7 | 81.9 | 23 GB |
| **Unlimited-OCR, tuned** | **32 % (23–41)** | **39 % (30–49)** | 64.1 % | 75.6 % | 58.8 % | 13.2 | 72.9 | 12 GB |
| **Unlimited-OCR** | **34 % (25–43)** | **43 % (33–53)** | 62.4 % | 83.9 % | 61.2 % | 13.6 | 31.1 | 13 GB |
| DeepSeek-OCR | 44 % (34–54) | 22 % (14–30) | 51.8 % | 80.2 % | 61.9 % | 4.3 | 18.6 | 12 GB |

Of Unlimited-OCR's 43 unusable scans, 15 looped, 13 were cut off at the answer limit and 15 came back empty
(no answer, or the whole scan labelled as a picture).

### Born-digital pages shown as scans, 95 page images (exact answer key)

| engine | header or footer lost | looped, cut off or empty | numbers found | German letters | word overlap | character errors, median | s per page, median |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | 12.6 % (6–20) | 6.3 % (2–12) | 91.6 % | 96.8 % | 87.7 % | 14.7 % | 45.4 |
| Chandra-OCR-2, tuned | 12.6 % (6–20) | 7.4 % (3–13) | 93.4 % | 95.4 % | 86.2 % | 15.7 % | 46.7 |
| RapidOCR, server detector | 15.8 % (10–23) | 1.1 % (0–3) | 88.8 % | 95.5 % | 88.2 % | 23.6 % | 4.1 |
| RapidOCR | 20.0 % (13–28) | 1.1 % (0–3) | 88.1 % | 92.5 % | 87.3 % | 25.1 % | 0.8 |
| **Unlimited-OCR, tuned** | **32.6 % (23–42)** | **15.8 % (8–23)** | 76.0 % | 85.6 % | 73.1 % | 23.6 % | 8.8 |
| **Unlimited-OCR** | **35.8 % (26–46)** | **27.4 % (18–37)** | 73.0 % | 79.0 % | 71.2 % | 23.4 % | 9.2 |
| DeepSeek-OCR | 54.7 % (45–64) | 16.8 % (10–24) | 59.4 % | 74.8 % | 68.7 % | 33.4 % | 4.9 |

On the same pages as rendered, without the scan simulation, Unlimited-OCR did worse, not better: 43.2 % lost and
37.9 % unusable (tuned: 36.8 % and 28.4 %).

## Whole document or one page at a time

The pipeline's plan of one OCR job per page is the right way to use this model here.

- **One page per request works where the model works.** Five clean synthetic pages (letters and work documents),
  one request each: 99 to 100 % of each page's words found, 9 to 14 s a page.
- **Several pages in one request does not work on llama-server.** The same five pages in one request with the
  maker's `Multi page parsing.` prompt: 186 s, the answer hit the limit, and it held 0 to 4 % of each page's
  words. With DRY sampling it stopped by itself after 127 s with the same empty result.
- The maker's own batch script (`infer.py` in its GitHub repository) also sends a PDF one page per request. Its
  multi-page mode needs the official runtimes, which need an NVIDIA GPU.
- The model's selling point, a long document in one pass, is therefore not available on this Mac, and the
  pipeline would not use it anyway.

## What this does not show

- **Not the official model.** A community 8-bit conversion on llama.cpp may differ from the BF16 weights on
  vLLM. In particular the maker's rule against repeating a 35-token run does not exist in llama-server, and
  loops are the largest single failure here. A lane that imitates the rule with DRY sampling (`unlimited-dry`)
  is in the prompt screen that follows this note.
- **Speed was measured beside four idle llama servers** (another application's). The other models' times are from their own runs a week earlier, so the times compare roughly, not exactly.
- **The scan answer key is a consensus of other engines.** It separates good from bad, not leaders from one
  another; the gap here is far larger than that weakness.

## Reproduce

```bash
uv run ocr-bakeoff run <test folder> --lanes unlimited,unlimited-tuned
```
