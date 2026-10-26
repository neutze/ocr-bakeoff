"""ocr-bakeoff: one command from a folder of documents to a results table."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

from . import consensus, deps, env
from .catalog import LANES, available_here, ranked, variant, variants_for
from .config import DEFAULT_BASELINE
from .corpus import RESULTS_DIRNAME, Page, discover, prepare
from .evaluate import reference_quality, score_outputs
from .lanes import run_lane
from .models import MODELS, Server, cached_files
from .report import render, write
from .score import aggregate, decide

# The --help description. A string of its own, not the module docstring, so the formatter
# keeps its indented examples as they are.
USAGE = """ocr-bakeoff: one command from a folder of documents to a results table.

    ocr-bakeoff ./testfiles                      # run everything that can run here
    ocr-bakeoff ./testfiles --lanes llm --yes    # only the vision LLMs, install without asking
    ocr-bakeoff lanes                            # what each engine is and whether it is ready
    ocr-bakeoff samples ./demo                   # a synthetic test folder to try it on
    ocr-bakeoff selftest                         # offline checks, no model needed
"""

COMMANDS = {"run", "lanes", "setup", "samples", "selftest", "profile", "synth", "pick"}
GROUPS = {
    "all": lambda lane: True,
    "llm": lambda lane: lane.family == "llm",
    "classic": lambda lane: lane.family == "classic",
    "controls": lambda lane: lane.family == "control",
    "pipelines": lambda lane: lane.family == "pipeline",
    "variants": variant,
}


def resolve_lanes(spec: str, skip: str) -> tuple[list[str], dict[str, str]]:
    chosen: list[str] = []
    for token in filter(None, (t.strip() for t in spec.split(","))):
        if token in GROUPS:
            chosen += [lane.id for lane in LANES.values() if GROUPS[token](lane)]
        elif token in LANES:
            chosen.append(token)
        else:
            raise SystemExit(f"unknown lane or group: {token} (see: ocr-bakeoff lanes)")
    chosen = list(dict.fromkeys(chosen))
    skipped = {lane: "skipped by --skip" for lane in filter(None, skip.split(",")) if lane in chosen}
    skipped.update({lane: "needs macOS" for lane in chosen if not available_here(LANES[lane])})
    return [lane for lane in chosen if lane not in skipped], skipped


def check_baseline(baseline: str) -> None:
    """The baseline is what a winner must beat, so it has to be an engine that is ranked."""
    lane = LANES.get(baseline)
    if lane is None:
        raise SystemExit(f"--baseline {baseline}: no such lane (see: ocr-bakeoff lanes)")
    if not ranked(lane):
        raise SystemExit(
            f"--baseline {baseline}: a {lane.family} lane is reported but never ranked, so it cannot be the "
            f"engine a winner must beat. Choose an LLM or classic lane (see: ocr-bakeoff lanes)."
        )


def lock_results(results: Path):
    """One run per results folder. A second run on the same folder would find the first one's
    model server on its port, skip that model, and fight it for the GPU, which corrupts every
    timing. The lock is released when the process ends, however it ends."""
    import fcntl
    import os

    results.mkdir(parents=True, exist_ok=True)
    handle = (results / ".lock").open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        raise SystemExit(
            f"Another ocr-bakeoff run is already working on {results} (pid {owner}).\n"
            f"Wait for it to finish, or stop it; progress is cached either way, and re-running continues."
        ) from None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def save_run(results: Path, run: dict) -> None:
    folder = results / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{run['run_id']}.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_run(args: argparse.Namespace) -> None:
    testdir = args.testdir.expanduser().resolve()
    if not testdir.is_dir():
        raise SystemExit(f"not a folder: {testdir}")
    check_baseline(args.baseline)
    results = (args.out or testdir / RESULTS_DIRNAME).expanduser().resolve()
    _lock = lock_results(results)  # noqa: F841 - held for the life of the run
    versions = env.versions()
    run = {
        "run_id": env.now().replace(":", "").replace("-", "")[:15],
        "started": env.now(),
        "testdir": str(testdir),
        "lanes_requested": args.lanes,
        "machine": env.machine(),
        "versions": versions,
        "snapshots": [env.snapshot("start")],
    }
    save_run(results, run)
    pages = discover(testdir, args.max_pages)
    if not pages:
        raise SystemExit(f"no PDF or image files under {testdir}")
    lanes, skipped = resolve_lanes(args.lanes, args.skip)
    needs, unavailable = deps.check([LANES[lane] for lane in lanes])
    if args.glm_url:
        needs = [need for need in needs if MODELS["glm"].repo not in need.what]
    skipped.update(unavailable)
    lanes = [lane for lane in lanes if lane not in unavailable]
    deps.ensure(needs, args.yes)

    kinds, tiers = Counter(p.kind for p in pages), Counter(p.tier for p in pages)
    print(
        f"{len({p.source for p in pages})} files, {len(pages)} pages · "
        + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items()))
        + " · "
        + ", ".join(f"{t} {n}" for t, n in sorted(tiers.items()))
    )
    print(f"results: {results}")
    prepare(pages, results)
    if args.reference == "gdrive":
        fetch_gdrive_references(args, pages, results)
    run_engines(args, pages, lanes, results, run, versions, skipped)
    run["finished"] = env.now()
    save_run(results, run)

    use_consensus = args.reference == "consensus"
    scored = score_outputs(pages, lanes, results, use_consensus=use_consensus, with_cer=not args.skip_cer)
    rows = scored.rows
    agg = aggregate(rows, [lane for lane in LANES if lane in lanes])
    decision = decide(agg, args.baseline)
    reference = reference_quality(pages, results, not args.skip_cer) if args.reference == "gdrive" else None
    meta = {
        "documents": len({p.source for p in pages}),
        "pages": len(pages),
        "images": len({(r["file"], r["page"], r["variant"]) for r in rows}),
        "kinds": dict(sorted(kinds.items())),
        "gold_sources": dict(scored.gold_sources),
        "no_gold": scored.no_gold,
        "skipped": dict(skipped),
        "conditions": env.run_conditions(results, scored.contributing_runs()),
        "unrecorded_images": scored.unrecorded_images(),
        "consensus_check": consensus.check(scored.consensus_pairs) if use_consensus else None,
        "warnings": scored.warnings(),
    }
    print(render(agg, decision, meta, reference))
    write(results, agg, decision, meta, reference, rows)
    print(f"Written: {results / 'summary.md'}, summary.json and per_page.csv; run record in {results / 'runs'}")


def fetch_gdrive_references(args: argparse.Namespace, pages: list[Page], results: Path) -> None:
    """Drive's reading of every born-digital page (to measure Drive itself) and of every scan
    without a typed reference. Cached: a page image is uploaded once."""
    from .gdrive import DriveOCR, connect, ensure_reference

    ocr = DriveOCR(connect(args.gdrive_credentials), args.gdrive_language)
    for page in pages:
        if page.tier == "born-digital":
            ensure_reference(ocr, results, f"{page.key}.clean", results / "pages" / page.variants["clean"])
        elif not page.typed_gold:
            ensure_reference(ocr, results, f"{page.key}.scan", results / "pages" / page.variants["scan"])
    print(f"Google Drive reference: {ocr.uploads} uploads (cached results reused)")


def run_engines(
    args: argparse.Namespace,
    pages: list[Page],
    lanes: list[str],
    results: Path,
    run: dict,
    versions: dict,
    skipped: dict[str, str],
) -> None:
    """Every lane over every page image that has no output yet, one model server at a time.
    A model whose server cannot start is recorded in `skipped` and the run goes on."""
    by_model: dict[str | None, list[str]] = {}
    for lane in lanes:
        by_model.setdefault(LANES[lane].model, []).append(lane)
    for model, group in by_model.items():
        work = []
        for lane in group:
            items = [
                (page, variant, results / "pages" / name)
                for page in pages
                for variant, name in variants_for(lane, page.variants).items()
            ]
            items = items[: args.limit] if args.limit else items
            work += [
                (lane, p, v, path)
                for p, v, path in items
                if args.force or not (results / "out" / lane / f"{p.key}.{v}.json").exists()
            ]
        if not work:
            continue
        run["snapshots"].append(env.snapshot(f"before {model or 'classic engines'}"))
        save_run(results, run)
        provenance = {lane: env.lane_provenance(lane, versions, args.glm_url) for lane in group}
        external = args.glm_url if model == "glm" else None
        server = Server(MODELS[model], results / "logs", external) if model else nullcontext()
        try:
            with server as running:
                if model:
                    print(f"\n[{model}] server up · {len(work)} page images across {', '.join(group)}")
                for n, (lane, page, variant, path) in enumerate(work, 1):
                    record = run_lane(lane, path, page.path, page.page, running.api if model else None)
                    if model and not external and record["errors"] and not running.healthy():
                        # A server that died (the machine slept, or something ended every llama-server) would
                        # otherwise leave an empty answer for this and every later page of the model.
                        print(f"  [{model}] server gone; starting it again and reading the page once more", flush=True)
                        running.__exit__(None, None, None)
                        running.__enter__()
                        record = run_lane(lane, path, page.path, page.page, running.api)
                    record |= {"run_id": run["run_id"], "finished_at": env.now(), "provenance": provenance[lane]}
                    out = results / "out" / lane / f"{page.key}.{variant}.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                    flags = ",".join(k for k, v in record["degenerate"].items() if v) or "ok"
                    errors = f" errors={len(record['errors'])}" if record["errors"] else ""
                    print(
                        f"  {lane:<16} {n:>3}/{len(work):<3} {page.kind:<14} {variant:<9} {record['seconds']:6.1f}s  {flags}{errors}",
                        flush=True,
                    )
        except (FileNotFoundError, RuntimeError, MemoryError, TimeoutError) as exc:
            print(f"\n[{model}] not run: {exc}", file=sys.stderr)
            skipped.update({lane: str(exc) for lane in group})


def cmd_lanes(_: argparse.Namespace) -> None:
    needs, unavailable = deps.check(list(LANES.values()))
    print(f"{'lane':<16} {'plan':<5} {'family':<8} {'status':<34} description")
    for lane in LANES.values():
        if lane.id in unavailable:
            status = unavailable[lane.id]
        elif lane.model and cached_files(MODELS[lane.model]) is None:
            status = f"model not downloaded (~{MODELS[lane.model].approx_gb:.1f} GB)"
        else:
            status = "ready"
        print(f"{lane.id:<16} {lane.plan_id:<5} {lane.family:<8} {status:<34} {lane.label}")
    if needs:
        print("\nA run will first install: " + "; ".join(f"{n.what} ({n.size})" for n in needs))


def cmd_setup(args: argparse.Namespace) -> None:
    lanes, _ = resolve_lanes(args.lanes, "")
    needs, unavailable = deps.check([LANES[lane] for lane in lanes])
    deps.ensure(needs, args.yes)
    print("ready" + (f"; cannot run here: {', '.join(unavailable)}" if unavailable else ""))


def cmd_samples(args: argparse.Namespace) -> None:
    from .samples import write_samples

    made = write_samples(args.dir.expanduser().resolve())
    print(f"{len(made)} sample documents in {args.dir} — now run: ocr-bakeoff {args.dir}")


def cmd_profile(args: argparse.Namespace) -> None:
    from .profile import profile_folder, summarise

    folder = args.folder.expanduser().resolve()
    print(
        f"profiling a sample of up to {args.sample} PDFs under {folder} (structure only, no text leaves this machine)...",
        flush=True,
    )
    print(summarise(profile_folder(folder, args.sample)))


def cmd_synth(args: argparse.Namespace) -> None:
    from .synth import generate

    out = args.dir.expanduser().resolve()
    summary = generate(out, args.profile.expanduser() if args.profile else None, args.count, args.seed)
    docs = summary["documents"]
    print(
        f"{len(docs)} look-alike documents in {out} (" + ", ".join(f"{k} {v}" for k, v in summary["plan"].items()) + ")"
    )
    print(
        "languages: "
        + ", ".join(f"{k} {v}" for k, v in Counter(d["lang"] for d in docs).most_common())
        + " · headers: "
        + ", ".join(f"{k} {v}" for k, v in Counter(d["header"] for d in docs).most_common())
        + " · footers: "
        + ", ".join(f"{k} {v}" for k, v in Counter(d["footer"] for d in docs).most_common())
    )
    print(f"now run: ocr-bakeoff {out}")


def cmd_pick(args: argparse.Namespace) -> None:
    from .pick import pick

    out = args.dir.expanduser().resolve()
    summary = pick(
        args.profile.expanduser(),
        out,
        args.per_kind,
        args.seed,
        args.tier,
        [d.expanduser().resolve() for d in args.exclude or []],
        args.language,
    )
    fmt = lambda c: ", ".join(f"{k} {v}" for k, v in c.most_common())  # noqa: E731
    print(
        f"{sum(summary['per_kind'].values())} real {args.tier} documents copied to {out} under opaque names "
        f"(mapping in {out / '.pick.json'}, mode 600): " + ", ".join(f"{k} {v}" for k, v in summary["per_kind"].items())
    )
    if summary["skipped"]:
        print("skipped: " + ", ".join(f"{k} {v}" for k, v in summary["skipped"].items()))
    print(
        f"languages: {fmt(summary['languages'])} · headers: {fmt(summary['headers'])} · producers: {fmt(summary['producers'])}"
    )


def cmd_selftest(_: argparse.Namespace) -> None:
    from .selftest import main as selftest

    selftest()


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "run")  # `ocr-bakeoff ./testfiles` is the one command
    parser = argparse.ArgumentParser(
        prog="ocr-bakeoff", description=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the bake-off on a folder of documents")
    run.add_argument("testdir", type=Path)
    run.add_argument("--lanes", default="all", help=f"ids or groups ({', '.join(GROUPS)}), comma-separated")
    run.add_argument("--skip", default="", help="lane ids to leave out")
    run.add_argument("--max-pages", type=int, default=1, help="pages per document (default 1)")
    run.add_argument(
        "--reference",
        choices=["none", "gdrive", "consensus"],
        default="none",
        help="reference for scans without a .gold.txt: gdrive uploads page images to your Google Drive; "
        "consensus uses what at least two other independent engines agree on, and checks itself "
        "on pages that do have a true reference",
    )
    run.add_argument("--gdrive-credentials", type=Path, help="OAuth client JSON (Desktop app), first use only")
    run.add_argument("--gdrive-language", default="de")
    run.add_argument("--yes", action="store_true", help="install missing dependencies without asking")
    run.add_argument("--out", type=Path, help=f"results folder (default: <testdir>/{RESULTS_DIRNAME})")
    run.add_argument("--baseline", default=DEFAULT_BASELINE, help="the engine a winner must beat (default: glm-custom)")
    run.add_argument("--glm-url", help="use an already running GLM-OCR server, e.g. http://127.0.0.1:11346/v1")
    run.add_argument("--limit", type=int, default=0, help="first N page images per lane (a timing pass)")
    run.add_argument("--force", action="store_true", help="re-run page images that already have output")
    run.add_argument("--skip-cer", action="store_true", help="skip character error rate (slow on long pages)")
    run.set_defaults(func=cmd_run)
    sub.add_parser("lanes", help="list engines and whether each is ready").set_defaults(func=cmd_lanes)
    setup = sub.add_parser("setup", help="install dependencies only")
    setup.add_argument("--lanes", default="all")
    setup.add_argument("--yes", action="store_true")
    setup.set_defaults(func=cmd_setup)
    samples = sub.add_parser("samples", help="write a synthetic test folder")
    samples.add_argument("dir", type=Path)
    samples.set_defaults(func=cmd_samples)
    profile = sub.add_parser(
        "profile", help="survey a folder's document structure (counts only) for look-alike generation"
    )
    profile.add_argument("folder", type=Path)
    profile.add_argument("--sample", type=int, default=2000)
    profile.set_defaults(func=cmd_profile)
    synth = sub.add_parser("synth", help="generate look-alike documents with invented content, shaped like a profile")
    synth.add_argument("dir", type=Path)
    synth.add_argument(
        "--profile", type=Path, help="a profile JSON from `ocr-bakeoff profile` (default: generic shapes)"
    )
    synth.add_argument("--count", type=int, default=30)
    synth.add_argument("--seed", type=int, default=20260926)
    synth.set_defaults(func=cmd_synth)
    pick = sub.add_parser(
        "pick", help="copy real born-digital documents from a profile into a test folder, opaque names, counts only"
    )
    pick.add_argument("dir", type=Path)
    pick.add_argument("--profile", type=Path, required=True)
    pick.add_argument(
        "--exclude",
        type=Path,
        action="append",
        help="an earlier test folder whose documents must not be picked again (repeatable)",
    )
    pick.add_argument("--per-kind", type=int, default=5)
    pick.add_argument("--language", help="only documents whose profile found this language (de, en, tr, other)")
    pick.add_argument("--seed", type=int, default=20260928)
    pick.add_argument(
        "--tier",
        choices=["born-digital", "scan"],
        default="born-digital",
        help="scan: --per-kind is the total, a quarter with an old OCR text layer; they need --reference gdrive or .gold.txt",
    )
    pick.set_defaults(func=cmd_pick)
    sub.add_parser("selftest", help="offline checks; no model, no download").set_defaults(func=cmd_selftest)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Everything finished so far is cached: run the same command again to continue.",
            file=sys.stderr,
        )
        sys.exit(130)
