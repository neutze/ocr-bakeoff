"""Gold, per-image metrics, aggregation and the decision rule (spec: Pick the Winner). No file I/O."""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Any

from .catalog import LANES, LAYER_READERS, ranked
from .config import (
    BOX_MIN_IOU,
    COST_FACTOR,
    DEGENERATION_MAX,
    DROP_RATE_MAX,
    INCONCLUSIVE_GAP,
    MIN_REGION_TOKENS,
    RECALL_DROP_THRESHOLD,
)
from .guards import repetition_loop
from .text import (
    diacritic_tokens,
    german_tokens,
    levenshtein,
    normalize,
    number_tokens,
    token_matches,
    tokens,
    turkish_tokens,
)

# Drops are judged where OCR is hard: simulated scans of born-digital pages, and real scans.
DECISION_VARIANTS = ("degraded", "scan")
_SECTION_RE = re.compile(r"^\[(TEXT|HEADER|FOOTER|NUMBERS)\]$", re.IGNORECASE)


# --- Gold ---------------------------------------------------------------------


def parse_typed(text: str) -> dict[str, str]:
    """[TEXT] / [HEADER] / [FOOTER] / [NUMBERS] sections; lines starting with # are comments."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = _SECTION_RE.match(stripped)
        if match:
            current = match.group(1).lower()
            sections.setdefault(current, [])
        elif current and stripped:
            sections[current].append(stripped)
    return {name: "\n".join(lines) for name, lines in sections.items() if lines}


def gold_record(full: str | None, header: str, footer: str, numbers_from: str, source: str) -> dict[str, Any]:
    letters_from = full if full is not None else header + "\n" + footer
    return {
        "full_text": full,
        "header_text": header,
        "footer_text": footer,
        "numbers": number_tokens(numbers_from),
        "diacritics": diacritic_tokens(letters_from),
        "german": german_tokens(letters_from),
        "turkish": turkish_tokens(letters_from),
        "source": source,
    }


# --- Per-image metrics --------------------------------------------------------


def matched_recall(gold: list[str], output: list[str], *, fuzzy: bool = True) -> float | None:
    """Share of gold tokens found in the output, each output token usable once."""
    if not gold:
        return None
    pool = Counter(output)
    hits, leftovers = 0, []
    for token, count in Counter(gold).items():
        taken = min(count, pool[token])
        hits += taken
        pool[token] -= taken
        leftovers.extend([token] * (count - taken))
    if fuzzy:
        for token in leftovers:
            if len(token) < 4:
                continue
            for candidate, available in pool.items():
                if available > 0 and abs(len(candidate) - len(token)) <= 1 and token_matches(token, candidate):
                    pool[candidate] -= 1
                    hits += 1
                    break
    return hits / len(gold)


def character_error_rate(output: str, gold: str) -> float:
    a, b = normalize(output), normalize(gold)
    start = 0
    while start < min(len(a), len(b)) and a[start] == b[start]:
        start += 1
    end = 0
    while end < min(len(a), len(b)) - start and a[-1 - end] == b[-1 - end]:
        end += 1
    return levenshtein(a[start : len(a) - end], b[start : len(b) - end]) / max(1, len(b))


def bag_of_words_f1(output: list[str], gold: list[str]) -> float | None:
    if not gold:
        return None
    common = sum((Counter(output) & Counter(gold)).values())
    if not common:
        return 0.0
    precision, recall = common / max(1, len(output)), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_image(text: str, gold: dict[str, Any], *, with_cer: bool = True) -> dict[str, Any]:
    out_tokens, out_numbers = tokens(text), number_tokens(text)
    header_gold, footer_gold = tokens(gold["header_text"]), tokens(gold["footer_text"])
    header, footer = matched_recall(header_gold, out_tokens), matched_recall(footer_gold, out_tokens)
    header_drop = len(header_gold) >= MIN_REGION_TOKENS and header is not None and header < RECALL_DROP_THRESHOLD
    footer_drop = len(footer_gold) >= MIN_REGION_TOKENS and footer is not None and footer < RECALL_DROP_THRESHOLD
    row = {
        "header_recall": header,
        "footer_recall": footer,
        "header_drop": header_drop,
        "footer_drop": footer_drop,
        "drop": header_drop or footer_drop,
        "numeric_recall": matched_recall(gold["numbers"], out_numbers, fuzzy=False),
        "invented_numbers": sum((Counter(out_numbers) - Counter(gold["numbers"])).values()),
        "diacritic_recall": matched_recall(gold["diacritics"], diacritic_tokens(text), fuzzy=False),
        "german_recall": matched_recall(gold.get("german", []), german_tokens(text), fuzzy=False),
        "turkish_recall": matched_recall(gold.get("turkish", []), turkish_tokens(text), fuzzy=False),
        "cer": None,
        "bow_f1": None,
    }
    if gold["full_text"] is not None:
        row["bow_f1"] = bag_of_words_f1(out_tokens, tokens(gold["full_text"]))
        if with_cer and gold.get("cer", True):  # a consensus is a bag of agreed words: no order to compare
            row["cer"] = character_error_rate(text, gold["full_text"])
    return row


def judged_degeneration(record: dict[str, Any], gold: dict[str, Any], lane_id: str | None = None) -> dict[str, bool]:
    """A loop is only a loop if the page itself does not repeat that way.

    The guard (half the lines duplicates, one line five times) fired on real documents for
    the PDF's own text layer and for engines that cannot loop, because some pages really are
    rows of identical lines. Counting that as degeneration would disqualify every engine that
    read the page faithfully, so repetition the reference also has is not held against it."""
    flags = dict(record["degenerate"])
    # The loop guard exists for generative models. A classic engine or a text-layer reader
    # transcribes what is there, so repetition in its output is the page's own.
    if lane_id is not None and LANES[lane_id].family in ("classic", "control"):
        flags["loop"] = False
    reference = gold.get("full_text")
    if flags.get("loop") and reference and repetition_loop(reference) is not None:
        flags["loop"] = False
    return flags


# --- Word positions (what a searchable PDF/A needs) ----------------------------


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def word_box_recall(engine_words: list[dict[str, Any]], gold_words: list[tuple[str, tuple]]) -> float | None:
    """Share of visible reference words an engine both read and placed: same word, boxes
    overlapping by IoU >= 0.3 (engines box glyphs tighter or looser than a text layer does)."""
    gold = [(tokens(text), box) for text, box in gold_words]
    gold = [(t[0], box) for t, box in gold if t]
    if not gold:
        return None
    pool = []
    for w in engine_words:
        t = tokens(w["text"])
        if t:
            pool.append([t[0], w["box"], False])
    hits = 0
    for word, box in gold:
        best, best_iou = None, BOX_MIN_IOU
        for candidate in pool:
            if not candidate[2] and token_matches(word, candidate[0]):
                overlap = _iou(box, candidate[1])
                if overlap >= best_iou:
                    best, best_iou = candidate, overlap
        if best is not None:
            best[2] = True
            hits += 1
    return hits / len(gold)


# --- Aggregation --------------------------------------------------------------


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    return round(statistics.fmean(values), 4) if values else None


def _p(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(q * len(ordered)) - 1)], 2)


def region_recall(row: dict[str, Any]) -> float | None:
    values = [v for v in (row["header_recall"], row["footer_recall"]) if v is not None]
    return statistics.fmean(values) if values else None


def aggregate(rows: list[dict[str, Any]], lane_ids: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for lane_id in lane_ids:
        lane_rows = [r for r in rows if r["lane"] == lane_id]
        if not lane_rows:
            continue
        lane = LANES[lane_id]
        # Pipelines and controls see pages as they are (born-digital as rendered, and real scans).
        decision = [
            r
            for r in lane_rows
            if r["variant"] in (("clean", "scan") if lane.family in LAYER_READERS else DECISION_VARIANTS)
        ]
        kinds = sorted({r["kind"] for r in decision})
        per_kind = {}
        for kind in kinds:
            kind_rows = [r for r in decision if r["kind"] == kind]
            recalls = [v for v in (region_recall(r) for r in kind_rows) if v is not None]
            per_kind[kind] = {
                "images": len(kind_rows),
                "drops": sum(r["drop"] for r in kind_rows),
                "region_recall": round(statistics.fmean(recalls), 3) if recalls else None,
            }
        seconds = [r["seconds"] for r in lane_rows]
        result[lane_id] = {
            "label": lane.label,
            "family": lane.family,
            "role": "ranked" if ranked(lane) else lane.family,
            "layer_used": sum(1 for r in lane_rows if r.get("used_layer")),
            "rules": dict(Counter(r["rule"] for r in lane_rows if r.get("rule"))),
            "geometry": lane.geometry,
            "plan_id": lane.plan_id,
            "images": len(lane_rows),
            "decision_images": len(decision),
            "drops": sum(r["drop"] for r in decision),
            "header_drops": sum(r["header_drop"] for r in decision),
            "footer_drops": sum(r["footer_drop"] for r in decision),
            "drop_rate": round(sum(r["drop"] for r in decision) / len(decision), 4) if decision else None,
            "degenerate_images": sum(r["degenerate"] for r in lane_rows),
            "degeneration_rate": round(sum(r["degenerate"] for r in lane_rows) / len(lane_rows), 4),
            **{flag: sum(r[flag] for r in lane_rows) for flag in ("echo", "loop", "truncated", "empty")},
            "header_recall": _mean(decision, "header_recall"),
            "footer_recall": _mean(decision, "footer_recall"),
            "numeric_recall": _mean(decision, "numeric_recall"),
            "numeric_miss_images": sum(
                1 for r in decision if r["numeric_recall"] is not None and r["numeric_recall"] < 1
            ),
            "invented_numbers": sum(r["invented_numbers"] for r in decision),
            "diacritic_recall": _mean(decision, "diacritic_recall"),
            "text_f1": _mean(decision, "bow_f1"),
            "german_recall": _mean(decision, "german_recall"),
            "turkish_recall": _mean(decision, "turkish_recall"),
            # For a control, only pages whose gold *is* the text layer check the scorer: typed
            # additions (image headers, logos) are text a layer reader cannot have.
            "cer_clean": _mean(
                [
                    r
                    for r in lane_rows
                    if r["variant"] == "clean" and (lane.family != "control" or r.get("gold") == "text layer")
                ],
                "cer",
            ),
            "cer_degraded": _mean([r for r in lane_rows if r["variant"] == "degraded"], "cer"),
            "cer_scan": _mean([r for r in lane_rows if r["variant"] == "scan"], "cer"),
            "boxes_clean": _mean([r for r in lane_rows if r["variant"] == "clean"], "box_recall"),
            "boxes_scan": _mean([r for r in lane_rows if r["variant"] == "degraded"], "box_recall"),
            "seconds_p50": _p(seconds, 0.5),
            "seconds_p95": _p(seconds, 0.95),
            "errors": sum(r["errors"] for r in lane_rows),
            "per_kind": per_kind,
            "scan_images": sum(1 for r in lane_rows if r["variant"] == "scan"),
            "empty_scans": sum(1 for r in lane_rows if r["variant"] == "scan" and r["empty"]),
        }
    return result


COUNT_METRICS = ("drops", "degenerate_images", "numeric_miss_images", "invented_numbers")


def disqualification(m: dict[str, Any]) -> list[str]:
    """Rule 1: why an engine is out, if it is."""
    reasons = []
    if m["degeneration_rate"] > DEGENERATION_MAX:
        reasons.append(f"degenerate output on {m['degenerate_images']}/{m['images']} images")
    if m["drop_rate"] is not None and m["drop_rate"] > DROP_RATE_MAX:
        reasons.append(f"header/footer dropped on {m['drops']}/{m['decision_images']} images")
    return reasons


def rank_key(m: dict[str, Any]) -> tuple:
    """Rule 2: fewest drops, then most numbers, then fewest invented, then lowest error rate."""
    cer = next((m[k] for k in ("cer_scan", "cer_degraded") if m[k] is not None), math.inf)
    return (m["drops"], -(m["numeric_recall"] or 0.0), m["invented_numbers"], cer)


def within_gap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Rule 3: fewer than INCONCLUSIVE_GAP images apart on every count is no difference."""
    return all(abs(b[k] - a[k]) < INCONCLUSIVE_GAP for k in COUNT_METRICS)


def decide(agg: dict[str, dict[str, Any]], baseline: str) -> dict[str, Any]:
    """Rule 1 disqualifies, rule 2 ranks, rule 3 calls narrow gaps a tie, rule 4 flags slow winners.

    The baseline must be a ranked lane (or one that was not run): a control or pipeline is never
    ranked, so it cannot be the engine a winner has to beat."""
    if baseline in LANES and not ranked(LANES[baseline]):
        raise ValueError(f"baseline {baseline} is a {LANES[baseline].family} lane, which is never ranked")
    decision: dict[str, Any] = {"disqualified": {}, "ranking": [], "winner": None, "verdict": "", "notes": []}
    survivors = []
    for lane_id, m in agg.items():
        if m["role"] != "ranked":
            continue
        reasons = disqualification(m)
        if reasons:
            decision["disqualified"][lane_id] = reasons
        else:
            survivors.append(lane_id)

    ranking = sorted(survivors, key=lambda lane_id: rank_key(agg[lane_id]))
    decision["ranking"] = ranking
    if not ranking:
        decision["verdict"] = "No engine survives rule 1: every one drops headers/footers or degenerates too often."
        return decision
    best = ranking[0]
    # Rule 3 applies to the whole top of the table, not just first against second: every
    # survivor fewer than INCONCLUSIVE_GAP images from the best on every count ties with it.
    leading = [lane for lane in ranking if within_gap(agg[best], agg[lane])]
    decision["leading_group"] = leading
    if len(leading) > 1:
        fastest = min(leading, key=lambda lane: agg[lane]["seconds_p50"] or math.inf)
        decision["fastest_in_group"] = fastest
        decision["notes"].append(
            f"{', '.join(leading)} tie: each is fewer than {INCONCLUSIVE_GAP} images from {best} on every count. "
            f"Fastest of them: {fastest} ({agg[fastest]['seconds_p50']} s/page)"
        )
    if baseline in leading and best != baseline:
        decision["verdict"] = (
            f"{best} ranks first but {baseline} is fewer than {INCONCLUSIVE_GAP} images from it on every count: inconclusive, no switch."
        )
        return decision
    if len(leading) > 1 and best != baseline:
        decision["verdict"] = (
            f"No single winner: {len(leading)} engines tie at the top. On accuracy they are equivalent here; "
            f"{decision['fastest_in_group']} is the fastest of them."
        )
        return decision
    if baseline not in agg:
        decision["winner"] = best
        decision["verdict"] = (
            f"{best} ranks first (baseline {baseline} was not run, so rules 3 and 4 were not applied)."
        )
        return decision
    if best == baseline:
        decision["winner"] = baseline
        decision["verdict"] = f"The baseline, {baseline}, ranks first. No switch."
        return decision
    # The baseline is ranked and not in the leading group: disqualified, or at least
    # INCONCLUSIVE_GAP images behind the best on some count.
    decision["winner"] = best
    decision["verdict"] = f"{best} wins."
    base_p50, best_p50 = agg[baseline]["seconds_p50"], agg[best]["seconds_p50"]
    if base_p50 and best_p50 and best_p50 > COST_FACTOR * base_p50:
        decision["verdict"] += (
            f" It is more than {COST_FACTOR:g}x slower than {baseline}: a trade-off to decide, not adopted by rule."
        )
    return decision


def reference_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """How good is a reference itself: its score rows against born-digital pages' exact layer."""
    if not rows:
        return None
    return {
        "pages": len(rows),
        "header_recall": _mean(rows, "header_recall"),
        "footer_recall": _mean(rows, "footer_recall"),
        "numeric_recall": _mean(rows, "numeric_recall"),
        "cer": _mean(rows, "cer"),
        "drops": sum(r["drop"] for r in rows),
    }
