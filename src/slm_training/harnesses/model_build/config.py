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
    model_name: str = "twotower"  # twotower | grammar_diffusion | stub
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
    # Grammar / DSL backend id: openui | openui-lark | openui-langcore | toy-layout
    grammar_dsl: str = "openui"
    grammar_top_k: int = 16
    structural_bias: float = 1.25
    grammar_ltr_repair: bool = False
    # Length-safe for compositional tokenizer (fixture gold up to ~160 tokens).
    grammar_ltr_max_tokens: int = 256
    grammar_ltr_stages: tuple[int, ...] | None = None
    grammar_ltr_primary: bool = False
    grammar_finalize_validate: bool = False
    ltr_loss_weight: float = 0.5
    fidelity_loss_weight: float = 0.0
    # None = preserve checkpoint on load; factory defaults new models to True.
    design_md_in_context: bool | None = None
    design_md_budget: int = 1800
    schema_in_context: bool = False
    slot_contract_in_context: bool = False
    slot_contract_constrained_decode: bool = False
    template_fill_decode: bool = False
    retrieval_k: int = 0
    best_of_n: int = 1
    use_curriculum: bool = False
    # Soft A/B/C mix (anti-leak); False restores hard stage cutovers.
    mix_curriculum: bool = True
    # Stub-only
    noise_rate: float = 0.0
    # Eval-driven training: run suite eval every N steps (0 disables).
    eval_every: int = 0
    eval_suite: str = "smoke"
    # Comma-separated suites for mid-train scoreboard (overrides single eval_suite when set).
    eval_suites: str = ""
    # Cap rico_held size during matrix / CPU evals (None = full suite).
    rico_eval_limit: int | None = None
    # Accelerator / throughput
    use_amp: bool = False
    use_compile: bool = False
    compile_mode: str = "default"
    grad_accum_steps: int = 1
    parallel_unmask: str = "adaptive"
    parallel_workers: int = 2
    remask_ratio: float = 0.0
    mdlm_schedule: bool = False
    mdlm_eps: float = 1e-3
    # Train-speed bundle (also set via --fast-train)
    cache_context: bool = True
    fuse_ltr_loss: bool = True
    grammar_fastpath: bool = True
    grammar_fastpath_mode: str = "hybrid"  # force | mask | hybrid
    grammar_draft_window: int = 8
    fastpath_aux_weight: float = 0.0
    fastpath_gate_threshold: float = 0.5
    # V4 critic / remask levers
    honest_slot_contract: bool = False
    suffix_rollback_window: int = 0
    remask_use_gate: bool = False
    remask_use_entropy: bool = False
    visible_corrupt_rate: float = 0.0
    trust_gate_train: bool = False
    grammar_prefer_structural: bool = True
    grammar_trust_model: bool = False
    grammar_sample_decode: bool = False
    grammar_sample_temperature: float = 0.8
    grammar_block_decode: bool = False
    grammar_block_size: int = 32
    # Grammar-diffusion (block production codec)
    block_size: int = 4
    production_loss_weight: float = 1.0
    slot_loss_weight: float = 0.5
    confidence_loss_weight: float = 0.25
    extendability_decode: bool = True
    # Cycle telemetry (train/infer span JSON)
    telemetry: bool = True

    @property
    def run_dir(self) -> Path:
        return self.run_root / self.run_id

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def train_records(self) -> Path:
        return self.train_dir / "records.jsonl"
