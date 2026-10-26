"""Run conditions and provenance: what produced an output, and what else was running.

A time per page means nothing without the conditions it was measured under: other
llama-server processes share the GPU and memory, and on a machine that hosts other
models that is the normal state, not an accident. And a cached output is only comparable
to another if the same model file and the same engine version produced both.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import re
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from . import __version__
from .catalog import LANES
from .models import MODELS, available_memory_gb, cached_files
from .text import run_text


def _run(*cmd: str) -> str:
    try:
        done = run_text(cmd, timeout=15, check=False)
        return (done.stdout + done.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


_MODEL_PATH = re.compile(r"(?:^|\s)(?:-m|--model)\s+(.+?\.gguf)(?=\s|$)")
_HF_REPO = re.compile(r"(?:^|\s)(?:-hf|--hf-repo)\s+(\S+)")
_PORT = re.compile(r"(?:^|\s)--port\s+(\d+)")


def parse_llama_servers(ps_output: str) -> list[dict[str, Any]]:
    """llama-server processes from `ps -eo pid,rss,command`: port, model, resident memory.

    Matched with patterns on the whole command, not by splitting on spaces: model paths
    such as ~/Library/Application Support/... contain spaces."""
    servers = []
    for line in ps_output.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[2].split(" ", 1)[0].endswith("llama-server"):
            continue
        command = parts[2]
        path, repo, port = _MODEL_PATH.search(command), _HF_REPO.search(command), _PORT.search(command)
        model = Path(path.group(1)).name if path else (repo.group(1) if repo else "?")
        servers.append(
            {
                "pid": int(parts[0]),
                "port": port.group(1) if port else None,
                "model": model,
                "rss_gb": round(int(parts[1]) / 1e6, 2) if parts[1].isdigit() else None,
            }
        )
    return servers


def llama_servers() -> list[dict[str, Any]]:
    return parse_llama_servers(_run("ps", "-eo", "pid,rss,command"))


def snapshot(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "at": now(),
        "available_memory_gb": round(available_memory_gb(), 1),
        "llama_servers": llama_servers(),
    }


def machine() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version()}
    if platform.system() == "Darwin":
        info["model"] = _run("sysctl", "-n", "hw.model")
        info["cpu"] = _run("sysctl", "-n", "machdep.cpu.brand_string")
        mem = _run("sysctl", "-n", "hw.memsize")
        info["memory_gb"] = round(int(mem) / 1e9) if mem.isdigit() else None
        info["macos"] = _run("sw_vers", "-productVersion")
    return info


def _package(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def versions() -> dict[str, Any]:
    llama = _run("llama-server", "--version")
    tesseract = _run("tesseract", "--version").splitlines()
    poppler = _run("pdftotext", "-v").splitlines()
    commit = _run("git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD")
    return {
        "ocr_bakeoff": __version__ + (f"+{commit}" if re.fullmatch(r"[0-9a-f]{7,12}", commit or "") else ""),
        "llama_server": next((line.strip() for line in llama.splitlines() if "version" in line.lower()), None),
        "tesseract": tesseract[0] if tesseract else None,
        "pdftotext": poppler[0] if poppler else None,
        "pymupdf": _package("pymupdf"),
        "rapidocr": _package("rapidocr"),
        "onnxruntime": _package("onnxruntime"),
    }


def lane_provenance(lane_id: str, versions: dict[str, Any], glm_url: str | None) -> dict[str, Any]:
    """What produced an output: the model file, or the engine and its version. Two outputs
    of one lane are comparable only if this matches."""
    lane = LANES[lane_id]
    if lane.model:
        if lane.model == "glm" and glm_url:
            return {"model": "external GLM server", "url": glm_url}
        spec, files = MODELS[lane.model], cached_files(MODELS[lane.model])
        return {
            "model_repo": spec.repo,
            "weights": spec.weights,
            "mmproj": spec.mmproj,
            "weights_bytes": files[0].stat().st_size if files else None,
            "llama_server": versions.get("llama_server"),
        }
    return {
        "vision": {"macos": machine().get("macos")},
        "tesseract": {"tesseract": versions.get("tesseract")},
        "rapidocr": {
            "rapidocr": versions.get("rapidocr"),
            "models": "PP-OCRv5 server det + latin rec"
            if lane_id == "rapidocr-server"
            else "PP-OCRv5 mobile det + latin rec",
        },
        "doctr": {"package": "onnxtr[cpu]", "models": "default detector and recogniser"},
        "easyocr": {"package": "easyocr", "languages": "de, en, tr", "device": "cpu"},
        "pdftotext": {"pdftotext": versions.get("pdftotext")},
        "pymupdf": {"pymupdf": versions.get("pymupdf")},
    }.get(lane.engine or "", {})


def run_conditions(results: Path, run_ids: set[str]) -> list[dict[str, Any]]:
    """Per contributing run: the machine, and the busiest moment it saw (most other llama
    servers, least free memory). Timings from different conditions are not comparable."""
    own_ports = {str(m.port) for m in MODELS.values()}
    found = []
    for run_id in sorted(r for r in run_ids if r != "unrecorded"):
        path = results / "runs" / f"{run_id}.json"
        if not path.exists():
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        others = [[s for s in snap["llama_servers"] if s.get("port") not in own_ports] for snap in run["snapshots"]]
        busiest = max(others, key=len) if others else []
        found.append(
            {
                "run_id": run_id,
                "machine": run["machine"].get("model") or run["machine"].get("platform"),
                "llama_server": run["versions"].get("llama_server"),
                "other_llama_servers": [f"{s['model']} ({s['rss_gb']} GB)" for s in busiest],
                "min_free_memory_gb": min((snap["available_memory_gb"] for snap in run["snapshots"]), default=None),
            }
        )
    return found
