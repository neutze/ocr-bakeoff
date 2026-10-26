# Prompt research: what each model's maker recommends, against what the test sends

2026-10-12. Desk research only: no model was run for this, except the Unlimited-OCR probe on one synthetic
page noted below. Sources are each model's own card and code on Hugging Face and GitHub, and the llama.cpp
pull request that added the model, read on 2026-10-12. HunyuanOCR was dropped from the tool on 2026-10-26
(its licence does not apply in the EU, the UK or South Korea), and its rows are removed here.

## What the test sends to every model today

- Text first, then the image, in one user message.
- The page rendered at 300 dpi, longest side up to 5,000 px, with a budget of 6,144 image tokens.
- Temperature 0, answers capped at 3,072 tokens. The `-tuned` lanes: 8,192 tokens and a repetition penalty of 1.05.

## Four gaps that apply to several models

1. **Order of image and instruction.** Unlimited-OCR's card puts the image first. With the instruction
   first it skipped the page header and looped on a synthetic page; with the image first it read the page
   completely (measured, one page). OvisOCR2's and DeepSeek-OCR's own code also put the image first. The
   llama.cpp examples for DeepSeek-OCR, GLM-OCR, and dots.ocr put the text first, and a model's
   chat template may reorder the parts, so this has to be measured per model, not assumed.
2. **Answer length.** The makers' own limits are 4,096 (LightOnOCR) to 32,768 tokens (Unlimited-OCR);
   8,192 for GLM-OCR and DeepSeek-OCR, 12,384 for Chandra, 16,384 for OvisOCR2, 24,000 for dots.ocr. The base
   lanes' 3,072 cut 4 to 6 % of pages off.
3. **Page size.** Several models name the size they were trained on: olmOCR 1,288 px longest side, LightOnOCR
   1,540 px (200 dpi), Chandra 192 dpi, GLM-OCR and dots.ocr 200 dpi, Qwen3-VL about 0.5 to 2.1 megapixels,
   PaddleOCR-VL about 1 megapixel. The test sends up to 14 megapixels. Only `lighton-tuned` shrinks the page.
4. **Repetition penalty.** GLM-OCR's own configuration uses 1.1. Unlimited-OCR and
   DeepSeek-OCR instead forbid repeating a 35-token run within a window, which llama-server does not have;
   its DRY sampler is the nearest thing.

## Per model

| model | the test sends | the maker recommends | gap | what to try |
|---|---|---|---|---|
| **Qwen3-VL 8B / 4B** | A prompt written for this test ("Transcribe every piece of text … Output only the transcribed text.") | No fixed OCR prompt. Its cookbook uses `Read all the text in the image.`, `Please output only the text content from the image without any additional descriptions or formatting.`, and for whole documents the trained keywords `qwenvl markdown` and `qwenvl html`. Pages of 0.5 to 2.1 megapixels. | The trained document keywords were never tried; pages are several times larger than the cookbook's. | `qwenvl markdown`, the cookbook's plain prompt, and both at 2 megapixels, against today's prompt. This is the production model, so it matters most. |
| **RapidOCR, Apple Vision, Tesseract** | no prompt | none | none | nothing |
| **DeepSeek-OCR** | `Free OCR.`, text first | Card: image, new line, then `Free OCR.` or `<\|grounding\|>Convert the document to markdown.`; 8,192 tokens. llama.cpp: one short prompt, `OCR`, `OCR markdown` or `<\|grounding\|>OCR`, with `--top-k 1`. | Possibly the order; the grounding prompt was never tried; answers capped at 3,072. It is the worst model in the test (44 to 63 % of pages with a lost header or footer), and Unlimited-OCR, built on it, failed the same way until the order was changed. | Image first; `<\|grounding\|>Convert the document to markdown.` with the boxes removed; `OCR`; 8,192 tokens. Most to gain of all models. |
| **Unlimited-OCR** (new) | `document parsing.` after the image | The same, plus the 35-token repetition rule and up to 32,768 tokens. Several pages in one request with `Multi page parsing.` | The repetition rule is missing in llama-server. | DRY sampling in the tuned lane; the multi-page prompt on whole documents. |
| **dots.ocr** | `Extract the text content from this image.` | That prompt is documented as "text only, except page header and page footer". The full prompt (`prompt_layout_all_en`) returns JSON with every block, including `Page-header` and `Page-footer`; 24,000 tokens; 200 dpi; under 11.3 megapixels. | The test asks for headers and footers with the one prompt that leaves them out. | The full layout prompt, its JSON turned into text; 8,192 tokens or more; pages under 11.3 megapixels. |
| **olmOCR-2** | A prompt written for this test | One exact prompt (`build_no_anchoring_v4_yaml_prompt`: "Attached is one page of a document that you must process. Just return the plain text representation …"), the page at 1,288 px longest side, temperature 0.1. The answer starts with a YAML block. | Not its prompt, and a page up to four times too large. The note in the tool that its pipeline needs PDF anchor text is out of date: this version takes the image alone. | The exact prompt, 1,288 px, the YAML block removed. It is still trained to leave headers and footers out, so expect better text, not fewer losses. |
| **Chandra-OCR-2** | Its `ocr` prompt, word for word | Its default is the other one, `ocr_layout`: every block as a `div` with a label, among them `Page-Header` and `Page-Footer`. 12,384 tokens, 192 dpi, pages at least 1,536 px. | The layout prompt names headers and footers explicitly and was never tried. | `ocr_layout` with 12,384 tokens at 192 dpi. |
| **OvisOCR2** | Its Markdown prompt, word for word, text first | The same prompt after the image; 16,384 tokens; pages up to 2,880 × 2,880 px; its own code removes repeated tails afterwards. | Order, and page size above its limit. | Image first; pages capped at 2,880 px. |
| **PaddleOCR-VL-1.6** | `OCR:` on the whole page or on Apple Vision blocks | `OCR:` on regions found by its own layout model, about 1 megapixel, 512 tokens per region. | The prompt is right. It is a region reader given whole pages; the block lanes already answer that. | Blocks capped at about 1 megapixel. Low priority. |
| **GLM-OCR** | `Text Recognition:`, or the 39-line custom prompt of `glm-custom` | Only three prompts exist: `Text Recognition:`, `Table Recognition:`, `Formula Recognition:`, on regions from PP-DocLayout-V3. Repetition penalty 1.1, 8,192 tokens, 200 dpi. | The 39-line prompt is outside what the model supports, which fits its result. | Penalty 1.1 on `glm-blocks`. Low priority: ADR-0018 retired whole-page GLM-OCR. |
| **LightOnOCR-2** | The image alone; tuned: 1,540 px, 8,192 tokens | The image alone, 1,540 px, 4,096 tokens, temperature 0.2. | None left but temperature 0.2, which the test's rule of repeatable answers excludes. | nothing |

## Proposed experiment

1. Screen every variant above on the 30 synthetic look-alikes (51 page images, exact answer key, nothing
   personal), one variant at a time against the model's current lane.
2. Run only the variants that win there on the two real sets (95 born-digital documents, 100 scans).
3. Variants become lanes of their own, so no existing result changes.

Order by expected gain: DeepSeek-OCR and dots.ocr (the test may have been unfair to both), Qwen3-VL (the
production model), then Chandra, OvisOCR2, olmOCR.

## Not checked

- Whether each GGUF's chat template reorders image and text. Only Unlimited-OCR's was read (it does not).
- No claim here is a measurement except the Unlimited-OCR order finding, on one page.

## Screen results, 2026-10-13

Step 1 of the experiment ran overnight on the 30 synthetic look-alikes. The figures are for the 30 hard page
images (20 simulated scans of born-digital pages, 10 scans), all with an exact answer key. "Lost" is a page
with a lost header or footer; "unusable" is looped, cut off or empty.

| lane | lost | unusable | numbers found | German letters | word overlap | s per page | against the model's first lane |
|---|---|---|---|---|---|---|---|
| `qwen8b` | 0 | 0 | 100.0 % | 98.3 % | 99.7 % | 49.7 | |
| `qwen8b-2mp` | 0 | 0 | 100.0 % | 98.0 % | 99.7 % | **28.7** | same reading, 42 % less time |
| `qwen8b-plain` | 0 | 0 | 100.0 % | 97.4 % | 99.6 % | 47.3 | no difference |
| `qwen8b-md` | 4 | 0 | 97.7 % | 89.7 % | 98.5 % | 48.4 | worse: headers and umlauts lost |
| `deepseek` | 23 | 2 | 57.0 % | 75.8 % | 71.0 % | 4.9 | |
| `deepseek-first` | 20 | 0 | 73.9 % | 87.5 % | 87.4 % | 5.1 | clearly better, still far from usable |
| `deepseek-ocr` | 23 | 7 | 52.9 % | 73.3 % | 65.7 % | 4.9 | worse |
| `deepseek-md` | 25 | 0 | 57.5 % | 66.2 % | 74.3 % | 5.8 | no better |
| `dots` | 25 | 0 | 73.8 % | 69.7 % | 88.8 % | 61.6 | |
| `dots-layout` | 24 | 0 | 74.3 % | 68.2 % | 89.0 % | 61.2 | no difference: it answers with plain text, not JSON |
| `ovis` | 0 | 0 | 98.7 % | 98.4 % | 99.1 % | 8.0 | |
| `ovis-first` | 0 | 0 | 98.7 % | 98.4 % | 99.1 % | 8.0 | identical answers: the order does not reach this model |
| `olmocr` | 20 | 0 | 77.6 % | 83.7 % | 90.2 % | 54.9 | |
| `olmocr-native` | 20 | 0 | 74.6 % | 75.9 % | 87.5 % | 21.1 | no better, 60 % less time |
| `chandra` | 0 | 0 | 100.0 % | 99.7 % | 99.4 % | 38.8 | |
| `chandra-layout` | 0 | 0 | 100.0 % | 99.0 % | 99.4 % | 41.1 | no difference |
| `unlimited` | 4 | 4 | 77.4 % | 69.1 % | 79.6 % | 9.5 | |
| `unlimited-tuned` | 4 | 4 | 77.7 % | 71.5 % | 78.4 % | 10.0 | no difference |
| `unlimited-dry` | 4 | 4 | 77.7 % | 71.5 % | 78.4 % | 4.4 | no difference in what it reads |
| `rapidocr` | 2 | 0 | 96.0 % | 94.4 % | 97.1 % | 0.8 | |

What it says:

1. **The test was not unfair to any model in a way that changes the ranking.** Of eleven variants, two beat
   their model's first lane, and neither moves its model into another group.
2. **Page size is the one finding worth acting on.** Qwen3-VL-8B reads a page of 2.1 megapixels exactly as
   well as the full 300 dpi page, in 29 s instead of 50 s (on the clean renders 28 s instead of 88 s).
3. **The order of image and prompt matters for the DeepSeek family only.** It removes every unusable page for
   DeepSeek-OCR and lifts its numbers from 57 % to 74 %, as it did for Unlimited-OCR. OvisOCR2 gives identical
   answers either way.
4. **The makers' other prompts did not help.** Qwen's document keyword is worse
   than what the test already sends; Chandra's and dots.ocr's layout prompts change nothing.
5. **Turkish letters stay weak everywhere** except dots.ocr (80 %): Qwen3-VL-8B 64 %,
   Chandra 71 %, RapidOCR 44 %, on 14 page images.

Step 2, on the real sets: `qwen8b-2mp` and `deepseek-first`.

## Page size, 2026-10-13 to 2026-10-14

Qwen3-VL-8B on a 2.1-megapixel page (`qwen8b-2mp`) on the real sets: 100 real scans, header or footer
lost 10 % against 8 % (3 pages new, 1 fewer), unusable 1 % either way, 24.9 s a page against 66.6 s. On 95
born-digital pages as rendered: lost 12.6 % against 13.7 %, unusable 6.3 % either way, 41.3 s against 81.4 s.
On the simulated scans, already small images, the time barely changes (42.0 s against 45.4 s).

Chandra-OCR-2 on smaller pages, 30 hard synthetic page images, exact key:

| lane | lost | unusable | numbers found | German letters | Turkish letters | s per page |
|---|---|---|---|---|---|---|
| `chandra-tuned` | 0 | 0 | 100.0 % | 99.0 % | 71.4 % | 98.5 |
| `chandra-192dpi` | 0 | 0 | 100.0 % | 99.4 % | 71.4 % | 59.7 |
| `chandra-2mp` | 0 | 0 | 99.9 % | 98.2 % | 57.1 % | 48.8 |

At its maker's 192 dpi Chandra reads as well as at 300 dpi in about 60 % of the time. At 2 megapixels it is
faster still but kept fewer Turkish letters (on 14 page images). The Mac's other models were busy during
these runs, so the times compare with each other only. Next: `chandra-192dpi` on the real sets.
