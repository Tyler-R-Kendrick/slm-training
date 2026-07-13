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
    # Prefer HF when available; tests/CI can pass --context-backend scratch.
    context_backend: str = "hf"  # scratch | hf
    hf_model_name: str = "HuggingFaceTB/SmolLM2-135M"
    # False for scratch POC; True by default when context_backend=hf (see factory)
    freeze_context: bool = True
    local_files_only: bool = False
    grammar_constrained: bool = True
    grammar_top_k: int = 16
    structural_bias: float = 1.25
    grammar_ltr_repair: bool = False
    grammar_ltr_max_tokens: int = 64
    grammar_ltr_primary: bool = False
    grammar_finalize_validate: bool = False
    ltr_loss_weight: float = 0.5
    fidelity_loss_weight: float = 0.0
    design_md_in_context: bool = True
    design_md_budget: int = 1800
    schema_in_context: bool = False
    retrieval_k: int = 0
    best_of_n: int = 1
    use_curriculum: bool = False
    # Stub-only
    noise_rate: float = 0.0
    # Eval-driven training: run suite eval every N steps (0 disables).
    eval_every: int = 0
    eval_suite: str = "smoke"
    # Cap rico_held size during matrix / CPU evals (None = full suite).
    rico_eval_limit: int | None = None

    @property
    def run_dir(self) -> Path:
        return self.run_root / self.run_id

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def train_records(self) -> Path:
        return self.train_dir / "records.jsonl"
