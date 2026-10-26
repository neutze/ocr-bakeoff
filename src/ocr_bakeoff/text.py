"""Normalisation, tokenising and distances. Pure functions, no I/O."""

from __future__ import annotations

import re
import subprocess
import unicodedata

from .config import DIACRITICS, GERMAN_LETTERS, TURKISH_LETTERS

_MARKUP_RES = [
    re.compile(r"</?(table_zone|handwritten)>", re.IGNORECASE),
    re.compile(r"<[^>]+>"),
    re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE),
    re.compile(r"\*\*|__"),
    re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$", re.MULTILINE),
    re.compile(r"^\s*```.*$", re.MULTILINE),
]
_WS_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"\w+(?:[.,:/\-]\w+)*", re.UNICODE)

# Model-output clean-up, from the same author's document pipeline (MIT): HTML, code
# fences and page markers are chrome, not transcription.
_HTML_BLOCK_BREAK_RE = re.compile(
    r"(?i)</?(?:p|div|section|article|header|footer|br|li|ul|ol|table|thead|tbody|tr|th|td|h[1-6])\b[^>]*>"
)
_HTML_TAG_RE = re.compile(r"(?i)</?[a-z][^>]*>")
_FENCE_RE = re.compile(r"(?mi)^\s*```(?:[a-z0-9_-]+)?\s*$")
_PAGE_MARKER_RE = re.compile(r"(?mi)^\s*\[\[PAGE\s+\d+\]\]\s*$")


def sanitize_model_output(text: str) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = _HTML_BLOCK_BREAK_RE.sub("\n", value)
    value = _HTML_TAG_RE.sub("", value)
    lines = [line.rstrip() for line in value.split("\n")]
    lines = [line for line in lines if not _FENCE_RE.match(line) and not _PAGE_MARKER_RE.match(line)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# Characters that print the same as a plain one but compare differently. PDF text layers
# often store a printed hyphen as a soft hyphen (U+00AD), which split "2025-6720" into two
# tokens in the gold, so every engine that read it right was charged with an invented number.
# Footnote markers are superscript digits in the layer and ordinary digits in any OCR output.
_EQUIVALENT = str.maketrans(
    {
        "\u00ad": "-",
        "\u2010": "-",
        "\u2011": "-",
        "\u2212": "-",
        "\u00b9": "1",
        "\u00b2": "2",
        "\u00b3": "3",
        "\u2070": "0",
        "\u2074": "4",
        "\u2075": "5",
        "\u2076": "6",
        "\u2077": "7",
        "\u2078": "8",
        "\u2079": "9",
        "\u00a0": " ",
        "\u202f": " ",
    }
)


def strip_markup(text: str) -> str:
    value = unicodedata.normalize("NFC", text or "").translate(_EQUIVALENT)
    for pattern in _MARKUP_RES:
        value = pattern.sub(" ", value)
    return value.replace("|", " ")


def normalize(text: str) -> str:
    """NFC, markup removed, whitespace collapsed. Case, punctuation and diacritics kept."""
    return _WS_RE.sub(" ", strip_markup(text)).strip()


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(normalize(text))


def number_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if any(c.isdigit() for c in t)]


def diacritic_tokens(text: str, letters: frozenset[str] = DIACRITICS) -> list[str]:
    return [t for t in tokens(text) if any(c in letters for c in t)]


def german_tokens(text: str) -> list[str]:
    return diacritic_tokens(text, GERMAN_LETTERS)


def turkish_tokens(text: str) -> list[str]:
    return diacritic_tokens(text, TURKISH_LETTERS)


def levenshtein(a: str, b: str, limit: int | None = None) -> int:
    """Edit distance, in C via rapidfuzz: CER on a long real page (and on a looping model's
    output of tens of thousands of characters) took minutes per page in pure Python.
    With `limit`, returns limit + 1 as soon as it is exceeded."""
    from rapidfuzz.distance import Levenshtein

    if limit is None:
        return Levenshtein.distance(a, b)
    return Levenshtein.distance(a, b, score_cutoff=limit)


def token_matches(gold: str, candidate: str) -> bool:
    return gold == candidate or (len(gold) >= 4 and levenshtein(gold, candidate, limit=1) <= 1)


def run_text(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run an external tool and read its output as UTF-8, leniently.

    Real PDFs make pdftotext emit bytes that are not valid UTF-8 (one of 40 real documents
    did, and a strict read crashed the run before any engine started). An undecodable byte
    becomes U+FFFD, which the garbled-layer rule already treats as a warning sign.
    """
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **kwargs)
