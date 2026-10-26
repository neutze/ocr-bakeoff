"""Dependencies: work out what the chosen lanes need, show it with sizes, ask once, install.

Python packages come with the project (uv). What this module handles is everything
else: command-line tools (Homebrew on macOS), Tesseract language data, model weights
(Hugging Face cache, shared with anything else on the machine that uses it), RapidOCR's
own small models, and the Apple Vision helper, which is compiled locally, not downloaded.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .catalog import Lane
from .config import BIN, NATIVE, RAPIDOCR_MODELS, TESSDATA, TESSERACT_LANGS, TESSERACT_TURKISH, VISION_TOOL
from .models import MODELS, cached_files, download

TESSDATA_URL = "https://github.com/tesseract-ocr/{repo}/raw/main/{lang}.traineddata"
RAPIDOCR_READY = RAPIDOCR_MODELS / "ready"


@dataclass
class Need:
    what: str
    size: str
    action: Callable[[], None]


def _brew(formula: str) -> Callable[[], None]:
    return lambda: subprocess.run(["brew", "install", formula], check=True)


def _fetch(url: str, dest: Path) -> Callable[[], None]:
    def run() -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=300) as response, part.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        part.replace(dest)  # a cut-off download never looks complete

    return run


def _warm_rapidocr() -> None:
    from .lanes import rapid_engine

    rapid_engine()
    RAPIDOCR_READY.parent.mkdir(parents=True, exist_ok=True)
    RAPIDOCR_READY.write_text("ok")


def _compile_vision() -> None:
    BIN.mkdir(parents=True, exist_ok=True)
    subprocess.run(["swiftc", "-O", "-o", str(VISION_TOOL), str(NATIVE / "vision_tool.swift")], check=True)


def check(lanes: list[Lane]) -> tuple[list[Need], dict[str, str]]:
    """What is missing for these lanes, and which lanes cannot run here at all (with why)."""
    needs: list[Need] = []
    unavailable: dict[str, str] = {}
    on_mac = platform.system() == "Darwin"
    has_brew = shutil.which("brew") is not None

    tools = {
        "pdftotext": ("poppler", "~30 MB", True),
        "tesseract": ("tesseract", "~40 MB", True),
        "llama-server": ("llama.cpp", "~20 MB", any(lane.model for lane in lanes)),
    }
    for binary, (formula, size, wanted) in tools.items():
        if wanted and shutil.which(binary) is None:
            if has_brew:
                needs.append(Need(f"{binary} (brew install {formula})", size, _brew(formula)))
            else:
                raise SystemExit(
                    f"{binary} is missing and Homebrew is not available. Install {formula} with your "
                    f"package manager (Debian/Ubuntu: poppler-utils, tesseract-ocr, llama.cpp) and re-run."
                )

    langs = [*TESSERACT_LANGS, *((TESSERACT_TURKISH,) if any(lane.id == "tesseract-tur" for lane in lanes) else ())]
    for lang, repo, size in [*((l, "tessdata_best", "~9-15 MB") for l in langs), ("osd", "tessdata", "~10 MB")]:
        dest = TESSDATA / f"{lang}.traineddata"
        if not dest.exists():
            needs.append(
                Need(
                    f"Tesseract {lang} language data ({repo})",
                    size,
                    _fetch(TESSDATA_URL.format(repo=repo, lang=lang), dest),
                )
            )

    for name in sorted({lane.model for lane in lanes if lane.model}):
        spec = MODELS[name]
        if cached_files(spec) is None:
            needs.append(
                Need(f"model {spec.label} from {spec.repo}", f"~{spec.approx_gb:.1f} GB", lambda s=spec: download(s))
            )

    if any(lane.id == "rapidocr" for lane in lanes) and not RAPIDOCR_READY.exists():
        needs.append(Need("RapidOCR PP-OCRv5 detector + latin recogniser (from ModelScope)", "~15 MB", _warm_rapidocr))

    vision_lanes = [lane.id for lane in lanes if lane.macos_only]
    if vision_lanes:
        if not on_mac:
            unavailable.update({lane: "Apple Vision needs macOS" for lane in vision_lanes})
        elif shutil.which("swiftc") is None:
            unavailable.update(
                {lane: "needs Xcode Command Line Tools (xcode-select --install)" for lane in vision_lanes}
            )
        elif not VISION_TOOL.exists() or VISION_TOOL.stat().st_mtime < (NATIVE / "vision_tool.swift").stat().st_mtime:
            needs.append(Need("Apple Vision helper (compiled locally, nothing downloaded)", "—", _compile_vision))
    return needs, unavailable


def ensure(needs: list[Need], assume_yes: bool) -> None:
    if not needs:
        return
    print("\nFirst run needs:")
    for need in needs:
        print(f"  • {need.what:<70} {need.size}")
    if not assume_yes:
        if not sys.stdin.isatty():
            raise SystemExit("Missing dependencies and no terminal to ask: re-run with --yes to install them.")
        if input("Download and install these now? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("Nothing installed.")
    for need in needs:
        print(f"  → {need.what}", flush=True)
        need.action()
    print()
