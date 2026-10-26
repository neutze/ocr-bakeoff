"""Scoring a run's saved outputs: one row per engine and page image, and where each came from.

Reads the references (text-layer gold, typed .gold.txt, Drive's reading) and what the engines
wrote to <results>/out/, picks the reference for every page image (the true one, or with
--reference consensus a stand-in built from the other engines) and scores against it. With a consensus, pages that do have a true reference are scored both
ways: that pair is the consensus's own check.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import consensus
from .catalog import LANES, variants_for
from .corpus import Page, reference_path
from .render import map_box, page_size
from .score import gold_record, judged_degeneration, parse_typed, reference_summary, score_image, word_box_recall
from .text import tokens

# Image variants a consensus is built for: the hard ones, where no text layer helps.
CONSENSUS_VARIANTS = ("scan", "degraded")


@dataclass
class Scored:
    rows: list[dict[str, Any]] = field(default_factory=list)
    gold_sources: Counter[str] = field(default_factory=Counter)
    no_gold: int = 0
    consensus_pairs: list[dict[str, Any]] = field(default_factory=list)
    run_ids: dict[str, Counter[str]] = field(default_factory=dict)  # lane -> run id -> images
    provenances: dict[str, set[str]] = field(default_factory=dict)  # lane -> distinct provenance records

    def contributing_runs(self) -> set[str]:
        return {run_id for runs in self.run_ids.values() for run_id in runs}

    def unrecorded_images(self) -> dict[str, int]:
        return {lane: runs["unrecorded"] for lane, runs in self.run_ids.items() if runs.get("unrecorded")}

    def warnings(self) -> list[str]:
        return [
            f"{lane}: outputs come from {len(prints)} different model files or engine versions; its numbers mix them"
            for lane, prints in self.provenances.items()
            if len(prints) > 1
        ]


def load_gold(page: Page, workdir: Path) -> dict[str, Any] | None:
    """The page's reference: its text layer (plus typed picture text), else a typed .gold.txt,
    else Drive's reading. None if it has none."""
    typed = parse_typed(Path(page.typed_gold).read_text(encoding="utf-8")) if page.typed_gold else {}
    if page.tier == "born-digital":
        layer = json.loads((workdir / "gold" / f"{page.key}.textlayer.json").read_text(encoding="utf-8"))
        # Typed header/footer on a born-digital page add text the layer lacks: a logo, an image footer.
        header = "\n".join(filter(None, [layer["header_text"], typed.get("header")]))
        footer = "\n".join(filter(None, [layer["footer_text"], typed.get("footer")]))
        full = "\n".join(filter(None, [layer["full_text"], typed.get("header"), typed.get("footer")]))
        g = gold_record(full, header, footer, full, "text layer + typed" if typed else "text layer")
        return g | {"words": layer.get("words", []), "geometry_ok": bool(layer.get("geometry_ok"))}
    if typed:
        full = typed.get("text")
        numbers = typed.get("numbers") or full or (typed.get("header", "") + "\n" + typed.get("footer", ""))
        return gold_record(full, typed.get("header", ""), typed.get("footer", ""), numbers, "typed")
    ref = reference_path(workdir, f"{page.key}.scan")
    if ref.exists():
        data = json.loads(ref.read_text(encoding="utf-8"))
        return gold_record(
            data["full_text"], data["header_text"], data["footer_text"], data["full_text"], data["engine"]
        )
    return None


def reference_quality(pages: list[Page], workdir: Path, with_cer: bool) -> dict[str, Any] | None:
    """How good is Google Drive itself? Scored against born-digital pages' exact text layer."""
    rows = []
    for page in pages:
        if page.tier != "born-digital":
            continue
        ref = reference_path(workdir, f"{page.key}.clean")
        if not ref.exists():
            continue
        data = json.loads(ref.read_text(encoding="utf-8"))
        rows.append(score_image(data["full_text"], load_gold(page, workdir), with_cer=with_cer))
    return reference_summary(rows)


def load_outputs(results: Path, page_key: str, variant: str) -> dict[str, dict[str, Any]]:
    """Every lane's output for one page image, panel members included even if not selected."""
    return {
        lane_id: json.loads(f.read_text(encoding="utf-8"))
        for lane_id in LANES
        if (f := results / "out" / lane_id / f"{page_key}.{variant}.json").exists()
    }


def box_recall(
    results: Path,
    page: Page,
    variant: str,
    record: dict[str, Any],
    gold: dict[str, Any],
    clean_sizes: dict[str, tuple[int, int]],
) -> float | None:
    """Word-box recall where the reference has trustworthy positions: the rendered page, and
    the simulated scan with each reference box carried through the same transform."""
    if (
        variant not in ("clean", "degraded")
        or not record.get("words")
        or not gold.get("words")
        or not gold.get("geometry_ok")
    ):
        return None
    reference = [(w[0], tuple(w[1:5])) for w in gold["words"]]
    if variant == "degraded":
        if page.key not in clean_sizes:
            clean_sizes[page.key] = page_size((results / "pages" / page.variants["clean"]).read_bytes())
        width, height = clean_sizes[page.key]
        reference = [(text, map_box(box, page.seed, width, height)) for text, box in reference]
    return word_box_recall(record["words"], reference)


def score_outputs(pages: list[Page], lanes: list[str], results: Path, *, use_consensus: bool, with_cer: bool) -> Scored:
    scored = Scored()
    clean_sizes: dict[str, tuple[int, int]] = {}
    for page in pages:
        gold = load_gold(page, results)
        if gold is None and not use_consensus:
            scored.no_gold += 1
            continue
        counted = _score_page(
            page, gold, lanes, results, scored, clean_sizes, use_consensus=use_consensus, with_cer=with_cer
        )
        if gold is None and not counted:
            scored.no_gold += 1
    return scored


def _score_page(
    page: Page,
    gold: dict[str, Any] | None,
    lanes: list[str],
    results: Path,
    scored: Scored,
    clean_sizes: dict[str, tuple[int, int]],
    *,
    use_consensus: bool,
    with_cer: bool,
) -> bool:
    """Add this page's rows to `scored`. Returns whether any image of it was scored."""
    on_disk: dict[str, dict[str, dict[str, Any]]] = {}
    counted = False
    for lane in lanes:
        for variant in variants_for(lane, page.variants):
            if variant not in on_disk:
                on_disk[variant] = load_outputs(results, page.key, variant)
            available = on_disk[variant]
            if lane not in available:
                continue
            record = dict(available[lane])
            lane_gold = gold
            if lane_gold is None:
                lane_gold = consensus.consensus_gold(available, exclude=lane) if variant in CONSENSUS_VARIANTS else None
                if lane_gold is None:
                    continue
            elif use_consensus and variant in CONSENSUS_VARIANTS and LANES[lane].family in ("llm", "classic"):
                # Where the true answer is known, score the consensus too: that is its check.
                stand_in = consensus.consensus_gold(available, exclude=lane)
                if stand_in is not None:
                    scored.consensus_pairs.append(
                        {
                            "lane": lane,
                            "image": (page.key, variant),
                            "truth": score_image(record["text"], gold, with_cer=False),
                            "consensus": score_image(record["text"], stand_in, with_cer=False),
                            "true_header": tokens(gold["header_text"]),
                            "consensus_header": tokens(stand_in["header_text"]),
                        }
                    )
            if not counted:
                scored.gold_sources[lane_gold["source"]] += 1
                counted = True
            record["degenerate"] = judged_degeneration(record, lane_gold, lane)
            scored.run_ids.setdefault(lane, Counter())[record.get("run_id", "unrecorded")] += 1
            if record.get("provenance") is not None:
                scored.provenances.setdefault(lane, set()).add(json.dumps(record["provenance"], sort_keys=True))
            scored.rows.append(
                {
                    "file": page.rel,
                    "page": page.page,
                    "kind": page.kind,
                    "tier": page.tier,
                    "variant": variant,
                    "lane": lane,
                    "gold": lane_gold["source"],
                    **score_image(record["text"], lane_gold, with_cer=with_cer),
                    "degenerate": any(record["degenerate"].values()),
                    **record["degenerate"],
                    "seconds": round(record["seconds"], 2),
                    "requests": record["requests"],
                    "errors": len(record["errors"]),
                    "used_layer": record.get("used_layer"),
                    "rule": record.get("rule"),
                    "box_recall": box_recall(results, page, variant, record, lane_gold, clean_sizes),
                }
            )
    return counted
