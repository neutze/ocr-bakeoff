"""The result tables: printed to the terminal, and written as Markdown and CSV.

The terminal and summary.md carry aggregates only. per_page.csv names the files, so it
stays in the results folder next to the test files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

FAMILY_ORDER = {"llm": 0, "classic": 1, "pipeline": 2, "control": 3}


def fmt(value: Any, kind: str = "") -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{value:.0%}"
    if isinstance(value, float):
        return f"{value:.2f}" if kind != "s" else f"{value:.1f}"
    return str(value)


def grid(headers: list[str], rows: list[list[str]], markdown: bool = False) -> str:
    if markdown:
        lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))  # noqa: E731
    return "\n".join([line(headers), "  ".join("─" * w for w in widths)] + [line(r) for r in rows])


def boxes_cell(m: dict[str, Any]) -> str:
    """Word-box recall on the simulated scans where measured; otherwise what the engine gives."""
    if m.get("boxes_scan") is not None:
        return fmt(m["boxes_scan"])
    return {"none": "none", "line boxes": "lines only", "block boxes": "none", "band boxes": "none"}.get(
        m["geometry"], m["geometry"]
    )


def order(agg: dict[str, dict[str, Any]], decision: dict[str, Any]) -> list[str]:
    ranked = decision["ranking"]
    rest = [lane for lane in agg if lane not in ranked]
    rest.sort(key=lambda lane: (FAMILY_ORDER[agg[lane]["family"]], lane))
    return ranked + rest


def render(
    agg: dict[str, dict[str, Any]],
    decision: dict[str, Any],
    meta: dict[str, Any],
    reference: dict[str, Any] | None,
    *,
    markdown: bool = False,
) -> str:
    out: list[str] = []
    h = (lambda text: f"## {text}") if markdown else (lambda text: f"\n{text}\n{'═' * len(text)}")
    lanes = order(agg, decision)
    competitors = [lane for lane in lanes if agg[lane]["role"] == "ranked"]

    out.append(h("OCR bake-off"))
    out.append(
        f"{meta['documents']} documents, {meta['pages']} pages, {meta['images']} page images scored · "
        f"kinds: {', '.join(f'{k} {n}' for k, n in meta['kinds'].items())} · "
        f"references: {', '.join(f'{k} {n}' for k, n in meta['gold_sources'].items()) or 'none'}"
        + (f" · {meta['no_gold']} pages without a reference (degeneration and speed only)" if meta["no_gold"] else "")
    )

    out.append(h("Overall — the decision set is simulated scans of born-digital pages plus real scans"))
    headers = [
        "#",
        "engine",
        "drops",
        "header",
        "footer",
        "numbers",
        "invented",
        "äöüß",
        "çğış",
        "text F1",
        "CER digital",
        "CER scan",
        "degenerate",
        "s/page",
        "word boxes",
    ]
    rows = []
    for lane in competitors:
        m = agg[lane]
        rank = str(decision["ranking"].index(lane) + 1) if lane in decision["ranking"] else "×"
        rows.append(
            [
                rank,
                lane,
                f"{m['drops']}/{m['decision_images']}",
                fmt(m["header_recall"]),
                fmt(m["footer_recall"]),
                fmt(m["numeric_recall"]),
                fmt(m["invented_numbers"]),
                fmt(m.get("german_recall")),
                fmt(m.get("turkish_recall")),
                fmt(m.get("text_f1")),
                fmt(m["cer_degraded"]),
                fmt(m["cer_scan"]),
                f"{m['degenerate_images']}/{m['images']}",
                fmt(m["seconds_p50"], "s"),
                boxes_cell(m),
            ]
        )
    out.append(grid(headers, rows, markdown))
    out.append(
        "# = rank under the decision rule, × = disqualified. header/footer/numbers/umlauts = recall (1.00 is "
        "perfect); äöüß / çğış = German / Turkish special letters kept exactly; text F1 = word overlap, blind to reading order (1.00 is perfect); CER = character error rate, which also punishes a different reading order (0.00 is perfect); "
        "word boxes = share of visible words read and placed where they are, on the simulated scans (what a searchable PDF/A needs)."
    )

    kinds = sorted({k for lane in competitors for k in agg[lane]["per_kind"]})
    if kinds:
        out.append(h("By document kind — header+footer recall (drops/images)"))
        rows = []
        for lane in competitors:
            cells = []
            for kind in kinds:
                k = agg[lane]["per_kind"].get(kind)
                cells.append(f"{fmt(k['region_recall'])} ({k['drops']}/{k['images']})" if k else "—")
            rows.append([lane, *cells])
        out.append(grid(["engine", *kinds], rows, markdown))
        out.append("A handful of images per kind shows direction, not a per-kind ranking.")

    pipelines = [lane for lane in lanes if agg[lane]["role"] == "pipeline"]
    if pipelines:
        out.append(
            h("Using the born-digital text where possible — pages as they are (born-digital as rendered, real scans)")
        )
        rows = []
        for lane in pipelines + [l for l in lanes if l == "pdftotext"]:
            m = agg[lane]
            rows.append(
                [
                    lane,
                    f"{m['drops']}/{m['decision_images']}",
                    fmt(m["header_recall"]),
                    fmt(m["footer_recall"]),
                    fmt(m["numeric_recall"]),
                    fmt(m["invented_numbers"]),
                    fmt(m["cer_clean"]),
                    fmt(m["cer_scan"]),
                    f"{m['layer_used']}/{m['images']}" if m["role"] == "pipeline" else "all",
                    fmt(m["seconds_p50"], "s"),
                ]
            )
        out.append(
            grid(
                [
                    "pipeline",
                    "drops",
                    "header",
                    "footer",
                    "numbers",
                    "invented",
                    "CER digital",
                    "CER scan",
                    "layer used",
                    "s/page",
                ],
                rows,
                markdown,
            )
        )
        for lane in pipelines:
            rules = agg[lane]["rules"]
            out.append(f"{lane}: " + " · ".join(f"{k} {v}" for k, v in sorted(rules.items(), key=lambda kv: -kv[1])))
        out.append("pdftotext = the text layer alone: it cannot see text inside pictures, and has nothing on scans.")

    controls = [lane for lane in lanes if agg[lane]["role"] == "control"]
    if controls:
        out.append(h("Controls — the PDF's own text layer, not competitors"))
        rows = [
            [lane, fmt(agg[lane]["cer_clean"]), f"{agg[lane]['empty_scans']}/{agg[lane]['scan_images']}"]
            for lane in controls
        ]
        out.append(
            grid(
                ["reader", "CER born-digital (checks the scorer)", "empty on scans (the pass-through gap)"],
                rows,
                markdown,
            )
        )

    if reference:
        out.append(h("Reference quality — Google Drive, against born-digital pages' exact text layer"))
        out.append(
            grid(
                ["pages", "header", "footer", "numbers", "CER", "drops"],
                [
                    [
                        str(reference["pages"]),
                        fmt(reference["header_recall"]),
                        fmt(reference["footer_recall"]),
                        fmt(reference["numeric_recall"]),
                        fmt(reference["cer"]),
                        str(reference["drops"]),
                    ]
                ],
                markdown,
            )
        )
        out.append(
            "Scans are scored against Drive's reading. Where Drive is wrong, a correct engine is marked wrong: "
            "read scan scores in the light of this row."
        )

    check = meta.get("consensus_check")
    if check:
        out.append(
            h("Consensus check — the same page images scored against the true reference and against the consensus")
        )
        out.append(
            f"{check['images']} page images with a true reference · the consensus holds {fmt(check['header_recall'])} of the true "
            f"header words, and {fmt(check['header_precision'])} of its header words are true · the same drop verdict on "
            f"{check['same_drop_verdict']:.0%} of engine × image pairs · engine ranking agreement (Spearman) "
            f"{fmt(check['rank_agreement'])}"
        )
        rows = [
            [
                lane,
                f"{e['drops_true']}/{e['images']}",
                f"{e['drops_consensus']}/{e['images']}",
                fmt(e["numbers_true"]),
                fmt(e["numbers_consensus"]),
            ]
            for lane, e in check["engines"].items()
        ]
        out.append(
            grid(
                ["engine", "drops (true)", "drops (consensus)", "numbers (true)", "numbers (consensus)"], rows, markdown
            )
        )
        out.append(
            "Where these columns agree, the consensus can stand in for a true reference on pages that have none."
        )

    out.append(h("Verdict"))
    for lane, reasons in decision["disqualified"].items():
        out.append(f"- {lane} disqualified: " + "; ".join(reasons))
    out.extend(f"- {note}" for note in decision["notes"])
    out.append(f"- {decision['verdict']}")
    conditions = meta.get("conditions") or []
    unrecorded = meta.get("unrecorded_images") or {}
    if conditions or unrecorded:
        out.append(h("Timing conditions — s/page is only comparable within the same conditions"))
        for c in conditions:
            others = ", ".join(c["other_llama_servers"]) or "none"
            out.append(
                f"- run {c['run_id']} on {c['machine']} · {c['llama_server'] or 'llama-server ?'} · "
                f"other llama servers: {others} · lowest free memory {c['min_free_memory_gb']} GB"
            )
        if unrecorded:
            out.append(
                "- page images from runs before conditions were recorded: "
                + ", ".join(f"{lane} {n}" for lane, n in unrecorded.items())
            )
    for warning in meta.get("warnings") or []:
        out.append(f"- warning: {warning}")
    skipped = meta.get("skipped") or {}
    if skipped:
        out.append(h("Not run"))
        out.extend(f"- {lane}: {why}" for lane, why in skipped.items())
    return "\n".join(out) + "\n"


def write(
    results: Path,
    agg: dict[str, Any],
    decision: dict[str, Any],
    meta: dict[str, Any],
    reference: dict[str, Any] | None,
    rows: list[dict[str, Any]],
) -> None:
    results.mkdir(parents=True, exist_ok=True)
    (results / "summary.md").write_text(render(agg, decision, meta, reference, markdown=True), encoding="utf-8")
    # The same aggregates, machine-readable, for comparing runs.
    (results / "summary.json").write_text(
        json.dumps(
            {"meta": meta, "engines": agg, "decision": decision, "reference": reference},
            ensure_ascii=False,
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    if rows:
        with (results / "per_page.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
