"""Vision-language models: what is known, what is on disk, and a server per model.

Every model runs on its own llama-server with the same settings: one slot, 16K context,
temperature 0, and the same image budget. One slot matters: recent llama.cpp splits the
context across parallel slots by default, which would leave a slot less room than one
page needs (6144 image + ~1200 prompt + 3072 output) and truncate silently.
"""

from __future__ import annotations

import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from .config import IMAGE_MAX_TOKENS

HEALTH_TIMEOUT = 600
MEMORY_HEADROOM_GB = 3.0


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    repo: str
    weights: str
    mmproj: str
    port: int
    approx_gb: float
    languages: str
    note: str
    context: int = 16384
    server_args: tuple[str, ...] = ()  # extra llama-server flags this model needs to answer at all


MODELS: dict[str, ModelSpec] = {
    m.name: m
    for m in (
        ModelSpec(
            "glm",
            "GLM-OCR 0.9B (Q8_0)",
            "ggml-org/GLM-OCR-GGUF",
            "GLM-OCR-Q8_0.gguf",
            "mmproj-GLM-OCR-Q8_0.gguf",
            11360,
            1.4,
            '"8 languages", unnamed by its card',
            "OCR specialist; region recogniser",
        ),
        ModelSpec(
            "paddle",
            "PaddleOCR-VL-1.6 0.9B",
            "PaddlePaddle/PaddleOCR-VL-1.6-GGUF",
            "PaddleOCR-VL-1.6-GGUF.gguf",
            "PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
            11361,
            1.8,
            "109 languages incl. German and Turkish",
            "OCR specialist; top of OmniDocBench v1.6",
        ),
        ModelSpec(
            "qwen8b",
            "Qwen3-VL-8B-Instruct (Q4_K_M)",
            "Qwen/Qwen3-VL-8B-Instruct-GGUF",
            "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
            "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
            11362,
            5.8,
            "broad multilingual",
            "general vision LLM",
        ),
        ModelSpec(
            "qwen4b",
            "Qwen3-VL-4B-Instruct (Q4_K_M)",
            "Qwen/Qwen3-VL-4B-Instruct-GGUF",
            "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
            "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
            11363,
            3.0,
            "broad multilingual",
            "general vision LLM",
        ),
        ModelSpec(
            "olmocr",
            "olmOCR-2-7B (Q4_K_M)",
            "lmstudio-community/olmOCR-2-7B-1025-GGUF",
            "olmOCR-2-7B-1025-Q4_K_M.gguf",
            "mmproj-olmOCR-2-7B-1025-F16.gguf",
            11364,
            6.0,
            "English-centric training data",
            "OCR fine-tune of Qwen2.5-VL; its real pipeline adds PDF anchor text this tool cannot supply",
        ),
        ModelSpec(
            "dots",
            "dots.ocr 1.7B (Q8_0)",
            "ggml-org/dots.ocr-GGUF",
            "dots.ocr-Q8_0.gguf",
            "mmproj-dots.ocr-Q8_0.gguf",
            11365,
            3.6,
            "multilingual",
            "OCR specialist; its plain-OCR prompt leaves page headers and footers out by training",
        ),
        ModelSpec(
            "deepseek",
            "DeepSeek-OCR (Q8_0)",
            "ggml-org/DeepSeek-OCR-GGUF",
            "DeepSeek-OCR-Q8_0.gguf",
            "mmproj-DeepSeek-OCR-Q8_0.gguf",
            11366,
            4.0,
            "multilingual",
            "OCR specialist; compresses the page into few vision tokens",
        ),
        ModelSpec(
            "lighton",
            "LightOnOCR-2-1B (Q8_0)",
            "ggml-org/LightOnOCR-2-1B-GGUF",
            "LightOnOCR-2-1B-Q8_0.gguf",
            "mmproj-LightOnOCR-2-1B-Q8_0.gguf",
            11367,
            1.4,
            "multilingual",
            "OCR specialist; takes the image alone, no instruction",
        ),
        ModelSpec(
            "chandra",
            "Chandra-OCR-2 (Q4_K_M)",
            "prithivMLmods/chandra-ocr-2-GGUF",
            "chandra-ocr-2.Q4_K_M.gguf",
            "chandra-ocr-2.mmproj-q8_0.gguf",
            11369,
            3.9,
            "multilingual",
            "OCR specialist; top of olmOCR-Bench; community quantisation",
            # Its chat template thinks first. With thinking on, most answers ended inside the thinking channel
            # and the answer itself came back empty (72 of 117 real pages on 2026-10-06).
            server_args=("--reasoning", "off"),
        ),
        ModelSpec(
            "ovis",
            "OvisOCR2 (Q8_0)",
            "bartowski/ATH-MaaS_OvisOCR2-GGUF",
            "ATH-MaaS_OvisOCR2-Q8_0.gguf",
            "mmproj-ATH-MaaS_OvisOCR2-f16.gguf",
            11370,
            1.3,
            "multilingual",
            "OCR specialist; top three of OmniDocBench v1.6; community quantisation",
        ),
        ModelSpec(
            "unlimited",
            "Unlimited-OCR (Q8_0)",
            "richardyoung/Unlimited-OCR-GGUF",
            "Unlimited-OCR-Q8_0.gguf",
            "mmproj-Unlimited-OCR-F16.gguf",
            11376,
            4.0,
            "multilingual",
            "OCR specialist built on DeepSeek-OCR; made to parse many pages in one pass; community quantisation",
        ),
    )
}

# The same weights for the "tuned" lanes, on their own ports, with room for the longer answers:
# 6,144 image tokens, a prompt of up to about 1,500 and 8,192 of output need more than 16K.
TUNED_CONTEXT = 24576
for _name, _port in (
    ("lighton", 11371),
    ("ovis", 11372),
    ("paddle", 11373),
    ("chandra", 11374),
    ("qwen8b", 11375),
    ("unlimited", 11377),
    ("deepseek", 11378),
    ("dots", 11379),
):
    _base = MODELS[_name]
    MODELS[f"{_name}-t"] = ModelSpec(
        f"{_name}-t",
        f"{_base.label}, tuned",
        _base.repo,
        _base.weights,
        _base.mmproj,
        _port,
        _base.approx_gb,
        _base.languages,
        _base.note,
        context=TUNED_CONTEXT,
        server_args=_base.server_args,
    )


def cached_file(repo: str, filename: str) -> Path | None:
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(repo_id=repo, filename=filename)
    return Path(found) if isinstance(found, str) and Path(found).exists() else None


def cached_files(spec: ModelSpec) -> tuple[Path, Path] | None:
    weights, mmproj = cached_file(spec.repo, spec.weights), cached_file(spec.repo, spec.mmproj)
    return (weights, mmproj) if weights and mmproj else None


def download(spec: ModelSpec) -> None:
    from huggingface_hub import hf_hub_download

    for filename in (spec.weights, spec.mmproj):
        hf_hub_download(repo_id=spec.repo, filename=filename)


def available_memory_gb() -> float:
    """Free + inactive pages on macOS; MemAvailable on Linux."""
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 1e9
    result = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False)
    page_size, counts = 16384, {}
    for line in result.stdout.splitlines():
        if "page size of" in line:
            page_size = int(line.split("page size of")[1].split()[0])
        name, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        if value.isdigit():
            counts[name.strip()] = int(value)
    return (counts.get("Pages free", 0) + counts.get("Pages inactive", 0)) * page_size / 1e9


def server_command(spec: ModelSpec, weights: Path, mmproj: Path) -> list[str]:
    """Local paths only, never -hf: a server start can never trigger a download."""
    return [
        "llama-server",
        "-m",
        str(weights),
        "--mmproj",
        str(mmproj),
        "--host",
        "127.0.0.1",
        "--port",
        str(spec.port),
        "-c",
        str(spec.context),
        "-np",
        "1",
        "-ngl",
        "99",
        "-fa",
        "on",
        "--temp",
        "0",
        "--image-max-tokens",
        str(IMAGE_MAX_TOKENS),
        *spec.server_args,
    ]


class Server:
    """A llama-server for one model, started for its lanes and stopped after them."""

    def __init__(self, spec: ModelSpec, log_dir: Path, external_url: str | None = None) -> None:
        self.spec, self.log_dir, self.external_url = spec, log_dir, external_url
        self.process: subprocess.Popen[bytes] | None = None
        self.log: IO[bytes] | None = None

    @property
    def api(self) -> str:
        return self.external_url.rstrip("/") if self.external_url else f"http://127.0.0.1:{self.spec.port}/v1"

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.api.removesuffix("/v1") + "/health", timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

    def __enter__(self) -> Server:
        if self.external_url:
            if not self.healthy():
                raise RuntimeError(f"{self.spec.name}: external server {self.external_url} is not healthy")
            return self
        files = cached_files(self.spec)
        if files is None:
            raise FileNotFoundError(f"{self.spec.name}: weights not downloaded")
        if self.healthy():
            raise RuntimeError(
                f"{self.spec.name}: port {self.spec.port} is already in use, by another ocr-bakeoff run or "
                f"another llama-server. Stop it, or pass --glm-url for a GLM server you run yourself."
            )
        available = available_memory_gb()
        if available < self.spec.approx_gb + MEMORY_HEADROOM_GB:
            raise MemoryError(
                f"{self.spec.name}: {available:.1f} GB available, needs ~{self.spec.approx_gb + MEMORY_HEADROOM_GB:.1f} GB"
            )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log = (self.log_dir / f"{self.spec.name}.log").open("wb")
        self.process = subprocess.Popen(server_command(self.spec, *files), stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            if self.healthy():
                return self
            if self.process.poll() is not None:
                self.__exit__(None, None, None)
                raise RuntimeError(
                    f"{self.spec.name}: llama-server exited ({self.process.returncode}); "
                    f"see {self.log_dir / (self.spec.name + '.log')}: {self.last_error()}"
                )
            time.sleep(2)
        self.__exit__(None, None, None)
        raise TimeoutError(f"{self.spec.name}: not healthy after {HEALTH_TIMEOUT}s")

    def last_error(self) -> str:
        path = self.log_dir / f"{self.spec.name}.log"
        lines = path.read_text(errors="replace").splitlines() if path.exists() else []
        hits = [line for line in lines if "error" in line.lower() or "failed" in line.lower()]
        return (hits or lines or ["no log"])[-1][-200:]

    def __exit__(self, *exc: object) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
