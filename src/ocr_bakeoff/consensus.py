"""A reference built from independent engines, for pages nobody can give a true one for.

For each page and each engine being scored, the reference is what at least two of the
*other* panel engines agree on. The panel is four engine families (a vision LLM, Apple
Vision, Tesseract, PP-OCR), so an engine never agrees with itself and one family's habits
do not define the answer. Header and footer references come from the panel members that
report positions. Word order is not agreed on, so no character error rate is computed.

Its weakness is stated, not hidden: text every engine misses cannot be held against any of
them, and a mistake most of them share looks right. So it is checked, on pages that do
have a true reference, and that check is printed next to its results.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from .config import CONSENSUS_AGREE, CONSENSUS_MIN_VOICES, FOOTER_FRACTION, HEADER_FRACTION
from .text import diacritic_tokens, german_tokens, number_tokens, tokens, turkish_tokens

PANEL = ("qwen8b", "vision", "tesseract-clean", "rapidocr")


def agreed(token_lists: list[list[str]], k: int = CONSENSUS_AGREE) -> list[str]:
    """Tokens at least k voices produced, as often as the k-th most frequent voice has them."""
    counters = [Counter(t) for t in token_lists]
    result: list[str] = []
    for token in set().union(*counters) if counters else set():
        counts = sorted((c[token] for c in counters), reverse=True)
        if len(counts) >= k and counts[k - 1] > 0:
            result.extend([token] * counts[k - 1])
    return result


def region_tokens(record: dict[str, Any], region: str) -> list[str] | None:
    """Tokens a positioned engine saw in the header or footer band; None if it gives no positions."""
    items = record.get("words") or record.get("lines")
    if not items:
        return None
    out: list[str] = []
    for item in items:
        x0, y0, x1, y1 = item["box"]
        if (region == "header" and y0 < HEADER_FRACTION) or (region == "footer" and y1 > 1 - FOOTER_FRACTION):
            out.extend(tokens(item["text"]))
    return out


def consensus_gold(records: dict[str, dict[str, Any]], exclude: str) -> dict[str, Any] | None:
    members = [lane for lane in PANEL if lane != exclude and lane in records]
    if len(members) < CONSENSUS_MIN_VOICES:
        return None
    full = agreed([tokens(records[m]["text"]) for m in members])
    positioned = [m for m in members if region_tokens(records[m], "header") is not None]
    header = (
        agreed([region_tokens(records[m], "header") for m in positioned]) if len(positioned) >= CONSENSUS_AGREE else []
    )
    footer = (
        agreed([region_tokens(records[m], "footer") for m in positioned]) if len(positioned) >= CONSENSUS_AGREE else []
    )
    text = " ".join(full)
    return {
        "full_text": text,
        "header_text": " ".join(header),
        "footer_text": " ".join(footer),
        "numbers": number_tokens(text),
        "diacritics": diacritic_tokens(text),
        "german": german_tokens(text),
        "turkish": turkish_tokens(text),
        "source": "consensus",
        "cer": False,
        "voices": members,
    }


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    rank = lambda xs: [sorted(xs).index(x) for x in xs]  # noqa: E731 - ties share the lowest rank
    ra, rb = rank(a), rank(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(num / den, 2) if den else None


def _mean_of(pairs: list[dict[str, Any]], which: str, key: str) -> float | None:
    values = [p[which][key] for p in pairs if p[which][key] is not None]
    return statistics.fmean(values) if values else None


def check(pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Compare consensus scoring with true-reference scoring on the same page images.

    pairs: one per (engine, image) with keys lane, truth (score row), consensus (score row),
    true_header (tokens), consensus_header (tokens)."""
    if not pairs:
        return None
    engines: dict[str, dict[str, Any]] = {}
    for lane in sorted({p["lane"] for p in pairs}):
        mine = [p for p in pairs if p["lane"] == lane]
        engines[lane] = {
            "images": len(mine),
            "drops_true": sum(p["truth"]["drop"] for p in mine),
            "drops_consensus": sum(p["consensus"]["drop"] for p in mine),
            "numbers_true": _mean_of(mine, "truth", "numeric_recall"),
            "numbers_consensus": _mean_of(mine, "consensus", "numeric_recall"),
        }
    hits = sum(sum((Counter(p["true_header"]) & Counter(p["consensus_header"])).values()) for p in pairs)
    true_total = sum(len(p["true_header"]) for p in pairs)
    cons_total = sum(len(p["consensus_header"]) for p in pairs)
    lanes = list(engines)
    order_key = lambda which: [engines[l][f"drops_{which}"] - (engines[l][f"numbers_{which}"] or 0) for l in lanes]  # noqa: E731
    same_drop_verdict = sum(1 for p in pairs if p["truth"]["drop"] == p["consensus"]["drop"])
    return {
        "images": len({p["image"] for p in pairs}),
        "engines": engines,
        "header_recall": round(hits / true_total, 3) if true_total else None,
        "header_precision": round(hits / cons_total, 3) if cons_total else None,
        "same_drop_verdict": round(same_drop_verdict / len(pairs), 3),
        "rank_agreement": _spearman(order_key("true"), order_key("consensus")),
    }
