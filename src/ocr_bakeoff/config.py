"""Paths and fixed parameters. Everything that decides a score lives here, not in flags."""

from __future__ import annotations

import os
import re
from pathlib import Path

CACHE = Path(os.environ.get("OCR_BAKEOFF_CACHE", Path.home() / ".cache" / "ocr-bakeoff"))
BIN = CACHE / "bin"
VISION_TOOL = BIN / "vision_tool"
TESSDATA = CACHE / "tessdata"
RAPIDOCR_MODELS = CACHE / "rapidocr"  # outside the venv, so a fresh clone or venv reuses them
GDRIVE_TOKEN = CACHE / "gdrive-token.json"
PACKAGE = Path(__file__).resolve().parent
PROMPTS = PACKAGE / "prompts"
NATIVE = PACKAGE / "native"

# Rendering. 300 DPI with the CropBox widened to the MediaBox, as the author's document pipeline
# renders for its production OCR lane.
RENDER_DPI = 300
MAX_PAGE_PIXELS = 5000
# Every vision model gets the same output cap and image budget, so neither is the variable.
MAX_TOKENS = 3072
IMAGE_MAX_TOKENS = 6144
REQUEST_TIMEOUT = 900

# A page is "born-digital" when its text layer is substantial and was not written by an
# OCR tool: prior-OCR layers have the right length and garbage content.
TEXT_LAYER_MIN_CHARS = 200
OCR_PRODUCER_RE = re.compile(
    r"tesseract|ocrmypdf|abbyy|finereader|readiris|omnipage|naps2|scan|epson|canon|brother|fujitsu|kofax|"
    r"xerox|ricoh|konica|kyocera|sharp mx|paperport|genius|scanbot|adobe paper capture",
    re.IGNORECASE,
)
# A sparse layer (a slide: a title and four bullets) is still born-digital if the page is
# not mostly picture; a scan never has its page mostly covered by vector text.
SPARSE_MIN_CHARS = 40
SPARSE_MAX_PICTURE_SHARE = 0.5
SOURCE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# A text-layer word belongs in the reference only if its box shows ink on the page rendered
# at this resolution: invisible, white-on-white and off-page text is not held against anyone.
VISIBILITY_DPI = 110
VISIBILITY_MIN_STD = 6.0

# Header = top 20% of the page, footer = bottom 20%.
HEADER_FRACTION = 0.20
FOOTER_FRACTION = 0.20
# Recall below this on either region counts the page as a drop.
RECALL_DROP_THRESHOLD = 0.8
# A region with fewer gold tokens than this is scored but cannot flag a drop.
MIN_REGION_TOKENS = 5
# A reference word is placed right when an engine's box for it overlaps by at least this IoU:
# engines box glyphs tighter or looser than a text layer does.
BOX_MIN_IOU = 0.3

# German and Turkish letters that must survive exactly, reported separately: an engine
# without Turkish data cannot produce ş, ğ or ı, and that is a different finding from
# losing umlauts.
GERMAN_LETTERS = frozenset("äöüÄÖÜß")
TURKISH_LETTERS = frozenset("çÇğĞıİşŞ")
DIACRITICS = GERMAN_LETTERS | TURKISH_LETTERS

# Consensus reference: a token counts when this many other panel engines agree on it, and
# a page gets a consensus only with at least this many other voices.
CONSENSUS_AGREE = 2
CONSENSUS_MIN_VOICES = 3

# The auto-* pipelines' rule for trusting a page's own text layer instead of OCR.
LAYER_MIN_CHARS = 40
PICTURE_PAGE_SHARE = 0.5  # at least this much picture: OCR the page
GARBLED_SHARE = 0.01  # replacement, private-use or control characters, or (cid:) codes
MIN_LETTER_SHARE = 0.4
IMAGE_REGION_MIN_SHARE = 0.003  # of the page: smaller pictures are rules, bullets and dots

# Decision rule (spec: Pick the Winner).
DEGENERATION_MAX = 0.02
DROP_RATE_MAX = 0.10
INCONCLUSIVE_GAP = 2
COST_FACTOR = 2.0
DEFAULT_BASELINE = "glm-custom"

# Tesseract language data, fetched from tessdata_best (osd from tessdata: best has none).
TESSERACT_LANGS = ("deu", "eng")
# Fetched for the tesseract-tur lane only; the other Tesseract lanes keep deu+eng, so their results stand.
TESSERACT_TURKISH = "tur"
