# ocr-bakeoff

Which OCR engine reads *your* documents best? Point it at a folder, get a table.

## In plain words

A scanned letter is only a picture until a program reads the words off it. That program is called
OCR. There are dozens now, from classic ones that need a second a page to AI models that need a
minute, and the rankings online test them on books and research papers, not on your letters.

ocr-bakeoff is a taste test. You give it a folder of your own documents. It hands the same pages to
every reader, checks each answer against what the page really says, and tells you which reader did
best. Above all it checks the top and the bottom of the page, because that is where a letter keeps
who sent it, when, and under which reference number, and it is the part many AI readers leave out.

Your documents never leave your computer. If you would rather not test on real papers at all, it
looks at the shape of your folder (counts only, no text, no names) and makes up look-alike
documents with invented text to test on instead.

## What it does

```bash
uv run ocr-bakeoff ./testfiles
```

It runs up to eleven vision LLMs and five classic OCR engines over the same page images and
scores them on what actually breaks OCR in a document archive: **dropped headers and footers**
(letterheads, logos, register lines, footnotes — as text *or* as images), wrong or invented
numbers, lost umlauts, and degenerate output (loops, echoed prompts, truncation). Results come
per engine and per document kind.

Everything runs locally. Nothing leaves the machine unless you opt into the Google Drive
reference (below). Results reproduce: the same documents give the same outputs and the same
scores on a rerun, and re-scoring cached outputs takes seconds.

## Quick start

```bash
uv run ocr-bakeoff samples ./demo     # a synthetic test folder: letters, invoices, slides, specs, scans
uv run ocr-bakeoff ./demo             # first run lists what it needs, asks once, installs, runs
uv run ocr-bakeoff lanes              # every engine and whether it is ready
uv run ocr-bakeoff selftest           # offline checks; never downloads, live engines only if installed
uv run ocr-bakeoff profile ~/Downloads # survey a folder's layouts, counts only (see below)
```

A full run of every engine over a few dozen documents takes hours. Keep the Mac awake with
`caffeinate -i uv run ocr-bakeoff ./testfiles`. Only one run may work on a folder at a time (a
lock refuses a second one), and an interrupted run resumes where it stopped: every finished page
is cached.

First run installs what the chosen engines need, after showing each item with its size:
`llama.cpp`, `tesseract` and `poppler` via Homebrew; Tesseract `deu`/`eng` data from
`tessdata_best`; model weights into the Hugging Face cache (shared with anything else that uses
it); RapidOCR's PP-OCRv5 models into `~/.cache/ocr-bakeoff/rapidocr` (RapidOCR fetches these from
ModelScope), so a new checkout or virtual environment reuses them. `--yes` skips the question. `--lanes llm` or `--skip qwen8b,olmocr`
narrows the run.

## How it works

What the tool does, rule by rule, is in [`spec/`](spec/), one note per command and per scoring rule:

- [OCR Bakeoff](spec/OCR%20Bakeoff.md): the overview and the rules every command keeps.
- Features: [Compare OCR Engines](spec/domain/features/Compare%20OCR%20Engines.md), [Make Look-alikes](spec/domain/features/Make%20Look-alikes.md), [Profile a Folder](spec/domain/features/Profile%20a%20Folder.md), [Pick Real Documents](spec/domain/features/Pick%20Real%20Documents.md), [Make Sample Folder](spec/domain/features/Make%20Sample%20Folder.md), [Show Lanes](spec/domain/features/Show%20Lanes.md).
- How it scores and decides: [Score Output](spec/domain/functionality/Score%20Output.md), [Pick the Winner](spec/domain/functionality/Pick%20the%20Winner.md), [Use the Text Layer](spec/domain/functionality/Use%20the%20Text%20Layer.md), and references for scans: [Consensus Reference](spec/domain/functionality/Consensus%20Reference.md), [Drive Reference](spec/domain/functionality/Drive%20Reference.md).
- Your folder and what it writes: [Test Folder](spec/data/Test%20Folder.md), [Results Folder](spec/data/Results%20Folder.md); every engine: [Engine Lanes](spec/data/Engine%20Lanes.md).

## Requirements

macOS with Homebrew is the tested path (Apple Silicon: every model runs on the GPU). Linux works
for everything but the Apple Vision engines; install `poppler-utils`, `tesseract-ocr` and
`llama.cpp` yourself. Python 3.11+ and [uv](https://docs.astral.sh/uv/).

The tool ships no model weights: it downloads each one from its Hugging Face page after you said
yes, under that model's own licence. Most are MIT or Apache 2.0; Chandra-OCR-2 limits commercial use. The list is in
[Engine Lanes](spec/data/Engine%20Lanes.md#model-licences).

## What it found on one archive

Run on 30 look-alikes, 20 real born-digital documents and 45 real scans from a German, English and
Turkish document archive (aggregates only; no document leaves the archive):

- **Use the text layer wherever the per-page rules allow**, and OCR only its pictures: 0 drops on
  the look-alikes, where the text layer alone dropped 12 of 30.
- **GLM-OCR on whole pages skips everything above the subject line**, with any prompt.
- **Qwen3-VL-8B reads scans most accurately**, but can loop on a hard page (1 of 85 real page
  images, Qwen3-VL-4B 3 of 30): run it behind a loop guard, and re-read a looping page with a
  classic engine.
- **Apple Vision** (0.3 s per page) and **RapidOCR** (under 1 s, never looped) are the fast
  near-equals; only Apple Vision and Tesseract place words for a searchable PDF/A.
- PaddleOCR-VL leads the synthetic set but loops on dense real pages; olmOCR-2 drops headers and
  footers by design.

## Origin

Built to choose a reader for my own document archive, after the model it used turned out to skip
everything above a letter's subject line, letterheads included. What the benchmark got wrong first,
and what it found, is written up on neutze.de:
[Score the Part of the Page You Cannot Afford to Lose](https://neutze.de/software/archive/score-the-part-of-the-page-you-cannot-lose/).

## Licence

MIT, see [LICENSE](LICENSE). The models it downloads keep their own licences (above).
