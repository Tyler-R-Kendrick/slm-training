"""Config for the model-building harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelBuildConfig:
    train_dir: Path
    test_dir: Path | None = None
    suite: str = "smoke"
    run_root: Path = Path("outputs/runs")
    run_id: str = "latest"
    steps: int = 200
    batch_size: int = 4
    lr: float = 3e-4
    seed: int = 0
    device: str = "cpu"
    model_name: str = "twotower"  # twotower | stub
    # TwoTower hyperparams
    d_model: int = 128
    n_heads: int = 4
    context_layers: int = 2
    denoiser_layers: int = 4
    mask_min: float = 0.15
    mask_max: float = 0.85
    gen_steps: int = 8
    # False for from-scratch POC; set True when swapping in a pretrained context tower
    freeze_context: bool = False
    # Stub-only
    noise_rate: float = 0.0

    @property
    def run_dir(self) -> Path:
        return self.run_root / self.run_id

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def train_records(self) -> Path:
        return self.train_dir / "records.jsonl"
