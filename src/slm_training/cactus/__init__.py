"""Cactus export / bench / inference adapter (ship MVP).

Kernel boundary: this module packages checkpoints and benchmarks the
**PyTorch** generate path. Native Cactus / NEON kernels stay outside this
repo (transpile offline with cactus-compute). Do not fold kernel code into
`slm_training.models` — keep on-device engines behind this adapter only.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CactusBench:
    latency_ms: float
    rss_mb: float | None
    tokens_per_sec: float | None
    overhead: float
    backend: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cactus_runtime_available() -> bool:
    """True when cactus-compute Python package is importable."""
    try:
        import cactus  # type: ignore  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def export_checkpoint_bundle(
    checkpoint: Path,
    out_dir: Path,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """
    Package a TwoTower checkpoint for Cactus/on-device handoff.

    Full PyTorch→Cactus transpilation requires the cactus toolchain on the host.
    This MVP writes a portable bundle (weights + tokenizer + manifest) that the
    transpiler can consume offline.
    """
    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    target = out_dir / "model.pt"
    shutil.copy2(checkpoint, target)
    tok = checkpoint.with_suffix(".tokenizer.json")
    if tok.exists():
        shutil.copy2(tok, out_dir / "tokenizer.json")
    manifest = {
        "format": "slm-training-cactus-bundle-v0",
        "checkpoint": str(target.name),
        "tokenizer": "tokenizer.json" if tok.exists() else None,
        "cactus_available": cactus_runtime_available(),
        "kernel_separate": True,
        "meta": meta or {},
        "transpile_hint": (
            "Install cactus-compute and run its PyTorch transpiler on model.pt "
            "when targeting .cact / on-device engine. Kernel code is not vendored here."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return out_dir


def bench_pytorch_generate(
    checkpoint: Path,
    *,
    prompt: str = "Build a hero card with title and body.",
    design_md: str | None = None,
    device: str = "cpu",
    repeats: int = 3,
) -> CactusBench:
    """Benchmark PyTorch generate path; used as overhead baseline when Cactus absent."""
    from slm_training.models.twotower import TwoTowerModel

    model = TwoTowerModel.from_checkpoint(checkpoint, device=device)
    # Warmup
    _ = model.generate(prompt, design_md=design_md)
    times: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        out = model.generate(prompt, design_md=design_md)
        times.append((time.perf_counter() - t0) * 1000.0)
    latency = sum(times) / len(times)
    toks = max(1, len(out.split()))
    tps = (toks / (latency / 1000.0)) if latency > 0 else None
    # Map latency to a soft overhead multiplier vs a 150ms reference.
    overhead = max(1.0, min(2.0, latency / 150.0))
    return CactusBench(
        latency_ms=round(latency, 2),
        rss_mb=None,
        tokens_per_sec=round(tps, 2) if tps else None,
        overhead=round(overhead, 3),
        backend="pytorch",
        notes=(
            "PyTorch baseline only; Cactus/NEON kernel remains a separate transpile "
            "target via export_checkpoint_bundle."
        ),
    )


def write_bench(path: Path | str, bench: CactusBench) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bench.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_overhead(bench_path: Path | str | None, default: float = 1.08) -> float:
    if bench_path is None:
        return default
    path = Path(bench_path)
    if not path.exists():
        return default
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("overhead") or default)


class CactusOrTorchBackend:
    """Generate adapter: prefer Cactus when available, else PyTorch checkpoint."""

    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        self.checkpoint = Path(checkpoint)
        self.device = device
        self._torch = None

    def generate(self, prompt: str, design_md: str | None = None) -> str:
        if cactus_runtime_available():
            # Placeholder for native cactus engine call once model is transpiled.
            # Fall through to torch until .cact artifact exists beside checkpoint.
            cact = self.checkpoint.parent / "model.cact"
            if cact.exists():
                raise NotImplementedError(
                    "Native cactus .cact generate wiring lands with transpiled artifact"
                )
        if self._torch is None:
            from slm_training.models.twotower import TwoTowerModel

            self._torch = TwoTowerModel.from_checkpoint(
                self.checkpoint, device=self.device
            )
        return self._torch.generate(prompt, design_md=design_md)
