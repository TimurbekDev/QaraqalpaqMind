"""Runtime environment checks for GPU training.

Everything here answers questions that are cheap to ask before a run and
expensive to discover during one: is there a GPU, does it support bfloat16, is
the Hugging Face cache on a disk that survives the pod, is there room for the
model.

On a cloud GPU the default cache location is the trap. `~/.cache/huggingface`
usually sits on the container's ephemeral overlay filesystem, so a 16 GB model
download is lost the moment the pod restarts - and re-downloaded on every run.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .logging import get_logger

logger = get_logger(__name__)

# Qwen3-8B in bf16 safetensors, plus tokenizer and config.
QWEN3_8B_DOWNLOAD_GB = 16.4


@dataclass(frozen=True, slots=True)
class GpuInfo:
    available: bool
    name: str = ""
    total_memory_gb: float = 0.0
    capability: tuple[int, int] = (0, 0)
    count: int = 0

    @property
    def supports_bf16(self) -> bool:
        """bfloat16 needs compute capability 8.0 (Ampere) or newer."""
        return self.capability >= (8, 0)

    @property
    def supports_flash_attention_2(self) -> bool:
        """flash-attn 2 requires Ampere or newer."""
        return self.capability >= (8, 0)

    def summary(self) -> str:
        if not self.available:
            return "no CUDA device"
        return (
            f"{self.name} ({self.total_memory_gb:.1f} GB, "
            f"sm_{self.capability[0]}{self.capability[1]}, x{self.count})"
        )


def gpu_info() -> GpuInfo:
    """Describe the CUDA device, without requiring torch to be installed."""
    try:
        import torch
    except ImportError:
        return GpuInfo(available=False)

    if not torch.cuda.is_available():
        return GpuInfo(available=False)

    properties = torch.cuda.get_device_properties(0)
    return GpuInfo(
        available=True,
        name=properties.name,
        total_memory_gb=properties.total_memory / 1024**3,
        capability=(properties.major, properties.minor),
        count=torch.cuda.device_count(),
    )


def hf_cache_dir() -> Path:
    """Where Hugging Face will actually put downloads."""
    for variable in ("HF_HUB_CACHE", "HF_HOME"):
        value = os.getenv(variable, "").strip()
        if value:
            return Path(value) / "hub" if variable == "HF_HOME" else Path(value)
    return Path.home() / ".cache" / "huggingface" / "hub"


def free_disk_gb(path: Path) -> float:
    """Free space on the filesystem holding `path`, following it up to a parent."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1024**3


def cache_is_ephemeral() -> bool:
    """True when the HF cache is somewhere a pod restart would wipe.

    A heuristic, and deliberately conservative: it flags the default home
    directory location, which on a container image is the overlay filesystem.
    """
    cache = hf_cache_dir()
    if os.getenv("HF_HOME") or os.getenv("HF_HUB_CACHE"):
        return False
    return str(cache).startswith(str(Path.home()))


@dataclass(frozen=True, slots=True)
class Finding:
    level: str  # "ok" | "warn" | "error"
    message: str


def preflight(required_download_gb: float = QWEN3_8B_DOWNLOAD_GB) -> list[Finding]:
    """Check the machine can actually run a training job. Cheap, and worth it."""
    findings: list[Finding] = []
    gpu = gpu_info()

    if not gpu.available:
        findings.append(Finding("error", "No CUDA device visible. Training will not run."))
    else:
        findings.append(Finding("ok", f"GPU: {gpu.summary()}"))
        if not gpu.supports_bf16:
            findings.append(
                Finding(
                    "error",
                    f"{gpu.name} is sm_{gpu.capability[0]}{gpu.capability[1]}; bf16 needs "
                    "sm_80+. Set runtime.bf16=false and runtime.fp16=true, and expect "
                    "more loss-scaling instability.",
                )
            )
        if gpu.total_memory_gb < 20:
            findings.append(
                Finding(
                    "warn",
                    f"{gpu.total_memory_gb:.0f} GB of VRAM. The shipped QLoRA config peaks "
                    "near 13 GB, but below 20 GB leaves little room for fragmentation.",
                )
            )

    cache = hf_cache_dir()
    if cache_is_ephemeral():
        findings.append(
            Finding(
                "warn",
                f"HF cache is {cache}, which on a cloud pod is usually ephemeral storage. "
                f"A {required_download_gb:.0f} GB model download will be lost on restart "
                "and repeated every run. Set HF_HOME to a path on the persistent volume.",
            )
        )
    else:
        findings.append(Finding("ok", f"HF cache: {cache}"))

    free = free_disk_gb(cache)
    if free < required_download_gb + 10:
        findings.append(
            Finding(
                "error" if free < required_download_gb else "warn",
                f"{free:.0f} GB free where the cache lives. The base model needs "
                f"{required_download_gb:.0f} GB, plus room for checkpoints (~1.9 GB each).",
            )
        )
    else:
        findings.append(Finding("ok", f"Disk: {free:.0f} GB free at {cache}"))

    return findings
