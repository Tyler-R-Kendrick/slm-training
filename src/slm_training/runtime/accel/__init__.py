"""Accelerator helpers: CUDA / Ascend NPU / CPU with SOTA-ready knobs.

Auto-detects the best device, configures thread pools for CPU, and exposes
AMP + torch.compile wrappers used by train/eval. Fused NEON/Cactus kernels
remain outside this module (see slm_training.runtime.cactus).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class AccelInfo:
    device: str
    backend: str  # cuda | npu | dml | cpu
    amp: bool
    compile: bool
    num_threads: int
    note: str = ""


def detect_device(preferred: str | None = None) -> AccelInfo:
    """Pick CUDA → Ascend NPU → DirectML → CPU unless explicitly forced."""
    import torch

    if preferred and preferred.lower() in {"dml", "directml"}:
        info = _detect_directml()
        if info is None:
            raise RuntimeError(
                "DirectML was requested but torch-directml is unavailable or has no device"
            )
        return info

    if preferred and preferred not in {"auto", "best"}:
        backend = "cpu"
        if preferred.startswith("cuda"):
            backend = "cuda"
        elif preferred.startswith("npu"):
            backend = "npu"
        elif preferred.startswith("privateuseone"):
            backend = "dml"
        threads = _configure_cpu_threads()
        info = AccelInfo(
            device=preferred,
            backend=backend,
            amp=backend in {"cuda", "npu"},
            compile=backend != "dml",
            num_threads=threads,
            note="user-preferred",
        )
        if backend == "cuda":
            configure_cuda_training()
        return info

    if torch.cuda.is_available():
        threads = _configure_cpu_threads()
        configure_cuda_training()
        return AccelInfo(
            device="cuda",
            backend="cuda",
            amp=True,
            compile=True,
            num_threads=threads,
            note=torch.cuda.get_device_name(0),
        )

    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            threads = _configure_cpu_threads()
            return AccelInfo(
                device="npu:0",
                backend="npu",
                amp=True,
                compile=True,
                num_threads=threads,
                note="torch_npu",
            )
    except Exception:  # noqa: BLE001
        pass

    directml = _detect_directml()
    if directml is not None:
        return directml

    threads = _configure_cpu_threads()
    return AccelInfo(
        device="cpu",
        backend="cpu",
        amp=False,
        compile=True,  # inductor CPU still helps small models
        num_threads=threads,
        note="cpu-only",
    )


def _detect_directml() -> AccelInfo | None:
    """Return the optional Torch-DirectML device used by Windows/WSL GPUs."""
    try:
        import torch_directml

        device = torch_directml.device()
        if device is None:
            return None
        return AccelInfo(
            device=str(device),
            backend="dml",
            amp=False,
            compile=False,
            num_threads=_configure_cpu_threads(),
            note="torch-directml",
        )
    except Exception:  # noqa: BLE001
        return None


def _configure_cpu_threads(num: int | None = None) -> int:
    import torch

    cpus = os.cpu_count() or 1
    n = int(num or max(1, cpus))
    # Leave one core for Node grammar bridge / OS when possible.
    if cpus >= 4:
        n = min(n, cpus - 1)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        torch.set_num_threads(n)
        torch.set_num_interop_threads(max(1, min(4, n)))
    except Exception:  # noqa: BLE001
        pass
    return n


def configure_cuda_training(*, allow_tf32: bool = True) -> dict[str, Any]:
    """Apply high-throughput CUDA defaults for HF Jobs / pod full trains.

    Safe no-op when CUDA is unavailable (CPU CI, ZeroGPU outside ``@spaces.GPU``).
    Call once after ``detect_device`` selects cuda.

    ZeroGPU Spaces demos must **not** use ``torch.compile``; this helper does not
    enable compile — callers pass ``--compile`` / ``--fast-train`` only on Jobs/pods.
    """
    import torch

    applied: dict[str, Any] = {"cuda": False}
    if not torch.cuda.is_available():
        return applied

    applied["cuda"] = True
    applied["device_name"] = torch.cuda.get_device_name(0)
    applied["capability"] = torch.cuda.get_device_capability(0)

    # Prefer TF32 on Ampere+ (A10G / A100 / H100 / RTX PRO 6000 Jobs flavors).
    if allow_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            applied["tf32"] = True
        except Exception:  # noqa: BLE001
            applied["tf32"] = False
        try:
            torch.set_float32_matmul_precision("high")
            applied["matmul_precision"] = "high"
        except Exception:  # noqa: BLE001
            pass

    try:
        torch.backends.cudnn.benchmark = True
        applied["cudnn_benchmark"] = True
    except Exception:  # noqa: BLE001
        applied["cudnn_benchmark"] = False

    # Avoid fragmenting allocator on long Jobs runs with varying batch shapes.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    applied["alloc_conf"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    return applied


def is_zerogpu_environment() -> bool:
    """True inside a Hugging Face Spaces ZeroGPU runtime.

    ZeroGPU is for Gradio demos (short ``@spaces.GPU`` slices). Full training
    must use HF Jobs or pods — ``torch.compile`` is unsupported on ZeroGPU.
    """
    hardware = (
        os.environ.get("SPACE_HARDWARE", "")
        or os.environ.get("SPACES_HARDWARE", "")
        or ""
    ).lower()
    if "zerogpu" in hardware:
        return True
    for key in ("SPACES_ZERO_GPU", "ZERO_GPU"):
        if os.environ.get(key, "").strip().lower() in {"1", "true", "yes"}:
            return True
    return False


def prefer_fast_train_env() -> bool:
    """True when SLM_FAST_TRAIN / HF Jobs markers request the speed knobs.

    Explicit ``SLM_FAST_TRAIN=0`` (or false/no) disables even on Jobs.
    Never auto-enables on ZeroGPU Spaces (compile / long trains unsupported).
    """
    if is_zerogpu_environment():
        return False
    raw = os.environ.get("SLM_FAST_TRAIN", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    for key in ("HF_JOBS_FAST_TRAIN",):
        if os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}:
            return True
    # HF Jobs runtimes expose HF_JOB_ID (and sometimes JOB_ID).
    if os.environ.get("HF_JOB_ID") or os.environ.get("JOB_ID"):
        return True
    return False


def maybe_compile(
    module: Any,
    *,
    enabled: bool = True,
    mode: str = "default",
    dynamic: bool | None = None,
) -> Any:
    """Wrap module with torch.compile when available/safe; fall back on failure."""
    if not enabled:
        return module
    if is_zerogpu_environment():
        # ZeroGPU forbids torch.compile; AoTI is the Spaces alternative.
        return module
    import torch

    if not hasattr(torch, "compile"):
        return module
    # CPU inductor needs Python.h / g++; skip unless explicitly forced.
    if not torch.cuda.is_available() and os.environ.get("SLM_FORCE_CPU_COMPILE") != "1":
        return module
    try:
        # reduce-overhead needs CUDA graphs; on CPU use default/inductor.
        if mode == "reduce-overhead" and not torch.cuda.is_available():
            mode = "default"
        # CUDA graphs prefer static shapes; default dynamic for other modes.
        if dynamic is None:
            dynamic = mode != "reduce-overhead"
        compiled = torch.compile(module, mode=mode, dynamic=dynamic)
        return compiled
    except Exception:  # noqa: BLE001
        return module


@contextmanager
def autocast_context(device: str, *, enabled: bool = True) -> Iterator[None]:
    """AMP autocast for cuda/npu; no-op on CPU."""
    import torch

    if not enabled:
        yield
        return
    if device.startswith("cuda") and torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
        return
    if device.startswith("npu"):
        try:
            with torch.autocast(device_type="npu", dtype=torch.float16):
                yield
            return
        except Exception:  # noqa: BLE001
            pass
    yield


def grad_scaler(device: str, *, enabled: bool = True) -> Any:
    """Return a GradScaler when CUDA fp16 AMP is used; else a no-op shim."""
    import torch

    if not enabled or not device.startswith("cuda"):
        return _NullScaler()
    # bf16 on Ampere+ typically skips scaler; keep for fp16 fallbacks.
    try:
        return torch.amp.GradScaler("cuda", enabled=False)
    except Exception:  # noqa: BLE001
        return _NullScaler()


class _NullScaler:
    def scale(self, loss: Any) -> Any:
        return loss

    def step(self, optimizer: Any) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def unscale_(self, optimizer: Any) -> None:
        return None


def sync_device(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.startswith("npu") and hasattr(torch, "npu"):
        try:
            torch.npu.synchronize()
        except Exception:  # noqa: BLE001
            pass
