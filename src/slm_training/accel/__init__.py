"""Accelerator helpers: CUDA / Ascend NPU / CPU with SOTA-ready knobs.

Auto-detects the best device, configures thread pools for CPU, and exposes
AMP + torch.compile wrappers used by train/eval. Fused NEON/Cactus kernels
remain outside this module (see slm_training.cactus).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class AccelInfo:
    device: str
    backend: str  # cuda | npu | cpu
    amp: bool
    compile: bool
    num_threads: int
    note: str = ""


def detect_device(preferred: str | None = None) -> AccelInfo:
    """Pick cuda → npu → cpu unless `preferred` forces a device string."""
    import torch

    if preferred and preferred not in {"auto", "best"}:
        backend = "cpu"
        if preferred.startswith("cuda"):
            backend = "cuda"
        elif preferred.startswith("npu"):
            backend = "npu"
        threads = _configure_cpu_threads()
        return AccelInfo(
            device=preferred,
            backend=backend,
            amp=backend in {"cuda", "npu"},
            compile=True,
            num_threads=threads,
            note="user-preferred",
        )

    if torch.cuda.is_available():
        threads = _configure_cpu_threads()
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

    threads = _configure_cpu_threads()
    return AccelInfo(
        device="cpu",
        backend="cpu",
        amp=False,
        compile=True,  # inductor CPU still helps small models
        num_threads=threads,
        note="cpu-only",
    )


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


def maybe_compile(
    module: Any,
    *,
    enabled: bool = True,
    mode: str = "default",
    dynamic: bool = True,
) -> Any:
    """Wrap module with torch.compile when available/safe; fall back on failure."""
    if not enabled:
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
