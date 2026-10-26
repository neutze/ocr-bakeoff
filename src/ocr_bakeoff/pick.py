"""Pick real documents from a profiled folder into a test folder, by kind, under opaque names.

For checking look-alike results on real paper without anyone reading the documents: the
files are copied locally as <kind>/r001.pdf ..., the mapping back to the originals stays in
a private file (mode 600) inside the test folder, and the terminal gets counts only.
Born-digital only by default, because those carry their own exact reference.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf as fitz

from .corpus import file_digest, text_layer_ok

PICK_KINDS = ("letters", "invoices", "presentations", "work-docs")


def _spread(candidates: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    """Round-robin over (producer, language, header style) so one generator cannot fill a kind."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for c in candidates:
        groups.setdefault((c.get("producer"), c.get("language"), c.get("header")), []).append(c)
    for group in groups.values():
        rng.shuffle(group)
    order = list(groups)
    rng.shuffle(order)
    chosen: list[dict[str, Any]] = []
    while len(chosen) < count and any(groups.values()):
        for key in order:
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop())
    return chosen


def excluded_by(folders: list[Path]) -> tuple[set[str], set[str]]:
    """Sources and file contents already used by earlier test folders, so a new sample is new."""
    sources, digests = set(), set()
    for folder in folders:
        record = folder / ".pick.json"
        if record.exists():
            sources.update(m["source"] for m in json.loads(record.read_text(encoding="utf-8"))["files"])
        digests.update(
            file_digest(f)
            for f in folder.rglob("r*")
            if f.is_file() and f.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
        )
    return sources, digests


def pick(
    profile_file: Path,
    out: Path,
    per_kind: int,
    seed: int,
    tier: str = "born-digital",
    exclude: list[Path] | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    """tier "born-digital": per_kind documents of each kind. tier "scan": per_kind scans in total,
    into scans/ (a scan's document type cannot be known without reading it), a quarter of them
    with an old OCR text layer, as real archives have. Files already in an `exclude` folder, and
    second copies of the same content under another name, are skipped."""
    rng = random.Random(seed)
    profiles = json.loads(profile_file.read_text(encoding="utf-8"))["profiles"]
    if language:
        profiles = [p for p in profiles if p.get("language") == language]
    mapping, made = [], Counter()
    used_sources, used_digests = excluded_by(exclude or [])
    skipped = Counter()
    if tier == "scan":
        with_layer = round(per_kind / 4)
        plans = [
            ("scans", [p for p in profiles if p.get("tier") == "scan"], per_kind - with_layer),
            ("scans", [p for p in profiles if p.get("tier") == "old-ocr-layer"], with_layer),
        ]
    else:
        plans = [
            (kind, [p for p in profiles if p.get("kind") == kind and p.get("tier") == "born-digital"], per_kind)
            for kind in PICK_KINDS
        ]
    for kind, pool, wanted in plans:
        pool = [p for p in pool if Path(p["path"]).exists()]
        taken = 0
        for candidate in _spread(pool, len(pool), rng):
            if taken >= wanted:
                break
            source = Path(candidate["path"])
            if str(source) in used_sources:
                skipped["already used"] += 1
                continue
            try:
                digest = file_digest(source)
            except OSError:
                continue
            if digest in used_digests:
                skipped["same content already picked or used"] += 1
                continue
            try:
                with fitz.open(source) as pdf:
                    # the tool's own rule must agree with the profile's tier
                    if text_layer_ok(pdf, 0) != (tier == "born-digital"):
                        continue
            except Exception:
                continue
            taken += 1
            made[kind] += 1
            used_digests.add(digest)
            target = out / kind / f"r{sum(made.values()):03d}{source.suffix.lower()}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            mapping.append(
                {
                    "file": str(target.relative_to(out)),
                    "source": str(source),
                    "profile_id": candidate.get("id"),
                    "producer": candidate.get("producer"),
                    "language": candidate.get("language"),
                    "header": candidate.get("header"),
                    "footer": candidate.get("footer"),
                    "shape": candidate.get("shape"),
                    "tier": candidate.get("tier"),
                    "sha": digest,
                }
            )
    private = out / ".pick.json"
    private.write_text(
        json.dumps({"profile": str(profile_file), "seed": seed, "files": mapping}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    private.chmod(0o600)
    return {
        "per_kind": dict(made),
        "skipped": dict(skipped),
        "old_ocr_layer": sum(1 for m in mapping if m.get("tier") == "old-ocr-layer"),
        "languages": Counter(m["language"] for m in mapping),
        "headers": Counter(m["header"] for m in mapping),
        "producers": Counter(m["producer"] for m in mapping),
    }
