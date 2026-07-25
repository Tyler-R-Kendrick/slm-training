"""CAP2-05 program-scale SemanticPlanV1 latent-codec bottleneck harness (AP-029 / SLM-325).

Bridges the CAP2-02 common ``LatentCodec`` interface (uniform scalar,
mixed-radix FSQ, binary LFQ, learned VQ, continuous) onto tensorized
``SemanticPlanV1`` program factors instead of oracle integer state ids. Reuses
the existing codec implementations' ``semantic_trace`` mode, the generic
``LatentCodecModel`` trainer, and the no-bypass audit extended in
``latent_codec_trainer.audit_no_bypass`` so decoder inputs are proven to
contain only codec output -- never a raw, unencoded gold factor.

Fixture mode is wired end to end against the SLM-144 deterministic fixture
corpus (``build_fixture_plan_corpus``). Small-corpus and full-corpus modes
require an externally built immutable ``SemanticPlanV1`` JSONL manifest and
fail closed with a clear error when one is not supplied; this iteration does
not synthesize a large corpus, so no fabricated result can appear under
those modes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_plan.canonicalize import plan_factor_fingerprints
from slm_training.data.semantic_plan.corpus import build_fixture_plan_corpus
from slm_training.models.binary_lfq import BinaryLFQCodec, BinaryLFQConfig
from slm_training.models.continuous_latent import ContinuousLatentCodec, ContinuousLatentConfig
from slm_training.models.latent_codec_trainer import (
    LatentCodecModel,
    audit_no_bypass,
    evaluate_latent_codec,
    train_latent_codec,
)
from slm_training.models.learned_vq import LearnedVQCodec, LearnedVQConfig
from slm_training.models.mixed_radix_fsq import MixedRadixFSQCodec, MixedRadixFSQConfig
from slm_training.models.semantic_plan_factors import (
    ProgramFactorTensorizerConfig,
    concat_factor_tensors,
    tensorize_semantic_plan,
    total_feature_dim,
)
from slm_training.models.uniform_scalar_codec import (
    UniformScalarCodec,
    UniformScalarCodecConfig,
)

# Factor families reported by plan_factor_fingerprints() used for the
# factor-wise distortion proxy below. "exact" and "coverage" are excluded:
# "exact" collapses to whole-plan identity (redundant with reconstruction
# rate) and "coverage" is derived bookkeeping, not a tensorized factor family.
_FIXTURE_FINGERPRINT_FACTORS: tuple[str, ...] = (
    "archetype",
    "role_set",
    "component_family",
    "cardinality",
    "symbol_role",
    "topology",
    "bindings",
)

__all__ = [
    "ProgramFactorArm",
    "ProgramFactorResult",
    "ProgramFactorMatrixReport",
    "load_fixture_plans",
    "load_corpus_manifest",
    "build_program_factor_arms",
    "build_matrix",
    "evaluate_program_factor_arm",
    "run_matrix",
]


@dataclass(frozen=True)
class ProgramFactorArm:
    """One arm of the CAP2-05 program-factor latent-codec matrix."""

    arm_id: str
    codec: str  # uniform_scalar | fsq | lfq | vq | continuous
    seed: int = 0
    hidden_dim: int = 64
    train_steps: int = 800
    lr: float = 2e-2
    K: int = 0
    d: int = 0
    radixes: tuple[int, ...] | None = None
    latent_dim: int | None = None
    commitment_cost: float = 0.25

    @property
    def capacity(self) -> int:
        if self.codec == "fsq":
            if self.radixes is None:
                raise ValueError(f"fsq arm {self.arm_id} requires radixes")
            cap = 1
            for level in self.radixes:
                cap *= level
            return cap
        if self.codec == "lfq":
            return 2**self.d
        if self.codec == "vq":
            return self.K
        if self.codec == "continuous":
            return self.latent_dim if self.latent_dim is not None else self.d
        return self.K**self.d


@dataclass(frozen=True)
class ProgramFactorResult:
    """Measured result for one CAP2-05 program-factor arm."""

    arm_id: str
    codec: str
    codec_levels: tuple[int, ...]
    seed: int
    num_records: int
    capacity: int
    exact_reconstruction_rate: float
    occupied_codewords: int
    code_utilization: float
    empirical_entropy_bits: float
    nominal_bits: float
    serialized_bytes_per_example: float
    nominal_bytes_per_example: float
    no_bypass_ok: bool
    leakage: bool
    factor_distortion: dict[str, float]
    elapsed_seconds: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "codec": self.codec,
            "codec_levels": list(self.codec_levels),
            "seed": self.seed,
            "num_records": self.num_records,
            "capacity": self.capacity,
            "exact_reconstruction_rate": self.exact_reconstruction_rate,
            "occupied_codewords": self.occupied_codewords,
            "code_utilization": self.code_utilization,
            "empirical_entropy_bits": self.empirical_entropy_bits,
            "nominal_bits": self.nominal_bits,
            "serialized_bytes_per_example": self.serialized_bytes_per_example,
            "nominal_bytes_per_example": self.nominal_bytes_per_example,
            "no_bypass_ok": self.no_bypass_ok,
            "leakage": self.leakage,
            "factor_distortion": self.factor_distortion,
            "elapsed_seconds": self.elapsed_seconds,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ProgramFactorMatrixReport:
    """Full CAP2-05 program-factor fixture/corpus matrix report."""

    run_id: str
    mode: str
    num_records: int
    corpus_path: str | None
    arms: tuple[ProgramFactorResult, ...]
    version: str = "cap2-05-v1"
    timestamp: str = field(default_factory=lambda: _utc_now())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "num_records": self.num_records,
            "corpus_path": self.corpus_path,
            "arms": [a.to_dict() for a in self.arms],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _hash_run_id(parts: tuple[Any, ...]) -> str:
    import hashlib

    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def load_fixture_plans(count: int = 32, seed: int = 0) -> list[SemanticPlanV1]:
    """Deterministic SLM-144 fixture corpus (gold plans, train split only)."""
    corpus = build_fixture_plan_corpus(count=count, seed=seed)
    return [plan for _, plan in corpus["train"]]


def load_corpus_manifest(path: Path) -> list[SemanticPlanV1]:
    """Load an immutable JSONL manifest of ``SemanticPlanV1`` records.

    Each non-blank line must be a JSON object accepted by
    ``SemanticPlanV1.from_dict``. Raises if the manifest is empty so an
    accidentally blank file cannot silently produce a zero-arm report.
    """
    plans: list[SemanticPlanV1] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        plans.append(SemanticPlanV1.from_dict(json.loads(stripped)))
    if not plans:
        raise ValueError(f"corpus manifest {path} contains no records")
    return plans


def build_program_factor_arms(seeds: tuple[int, ...] = (0,)) -> list[ProgramFactorArm]:
    """CAP2-05 arms at roughly matched nominal capacity for the fixture corpus."""
    arms: list[ProgramFactorArm] = []
    for seed in seeds:
        arms.extend(
            [
                ProgramFactorArm(
                    "plan_fsq_2_3_3_4_5",
                    "fsq",
                    seed=seed,
                    radixes=(2, 3, 3, 4, 5),
                    train_steps=1600,
                ),
                ProgramFactorArm("plan_lfq_d6", "lfq", seed=seed, d=6, train_steps=1600),
                ProgramFactorArm(
                    "plan_vq_64_d8",
                    "vq",
                    seed=seed,
                    K=64,
                    latent_dim=8,
                    # semantic_trace mode must learn a genuine feature->code
                    # function (unlike oracle_state's direct per-index lookup),
                    # so this needs more steps than the CAP2-02 oracle-state
                    # arm of the same family to reliably reach exact recall.
                    train_steps=4000,
                ),
                ProgramFactorArm(
                    "plan_continuous_d6",
                    "continuous",
                    seed=seed,
                    latent_dim=6,
                    train_steps=1200,
                ),
                ProgramFactorArm(
                    "plan_uniform_b2d6",
                    "uniform_scalar",
                    seed=seed,
                    K=2,
                    d=6,
                    train_steps=3000,
                    lr=1e-2,
                ),
            ]
        )
    return arms


def build_matrix(
    seeds: tuple[int, ...] = (0,),
    arms_filter: tuple[str, ...] | None = None,
) -> list[ProgramFactorArm]:
    arms = build_program_factor_arms(seeds)
    if arms_filter:
        wanted = set(arms_filter)
        arms = [a for a in arms if a.arm_id in wanted]
    return arms


def _build_model(arm: ProgramFactorArm, num_records: int, feature_dim: int) -> LatentCodecModel:
    """Construct a LatentCodecModel over program-factor features (semantic_trace mode)."""
    if arm.codec == "uniform_scalar":
        cfg = UniformScalarCodecConfig(
            num_states=num_records,
            K=arm.K,
            d=arm.d,
            hidden_dim=arm.hidden_dim,
            feature_dim=feature_dim,
            mode="semantic_trace",
        )
        codec = UniformScalarCodec(cfg)
    elif arm.codec == "fsq":
        if arm.radixes is None:
            raise ValueError(f"fsq arm {arm.arm_id} requires radixes")
        cfg = MixedRadixFSQConfig(
            num_states=num_records,
            levels=arm.radixes,
            hidden_dim=arm.hidden_dim,
            feature_dim=feature_dim,
            mode="semantic_trace",
        )
        codec = MixedRadixFSQCodec(cfg)
    elif arm.codec == "lfq":
        cfg = BinaryLFQConfig(
            num_states=num_records,
            d=arm.d,
            hidden_dim=arm.hidden_dim,
            feature_dim=feature_dim,
            mode="semantic_trace",
        )
        codec = BinaryLFQCodec(cfg)
    elif arm.codec == "vq":
        cfg = LearnedVQConfig(
            num_states=num_records,
            codebook_size=arm.K,
            latent_dim=arm.latent_dim or arm.d,
            hidden_dim=arm.hidden_dim,
            feature_dim=feature_dim,
            mode="semantic_trace",
            commitment_cost=arm.commitment_cost,
        )
        codec = LearnedVQCodec(cfg)
    elif arm.codec == "continuous":
        cfg = ContinuousLatentConfig(
            num_states=num_records,
            latent_dim=arm.latent_dim or arm.d,
            hidden_dim=arm.hidden_dim,
            feature_dim=feature_dim,
            mode="semantic_trace",
        )
        codec = ContinuousLatentCodec(cfg)
    else:
        raise ValueError(f"unknown codec {arm.codec!r}")
    return LatentCodecModel(codec, num_records)


def _factor_wise_distortion(
    plans: list[SemanticPlanV1],
    codes: torch.Tensor,
) -> dict[str, float]:
    """Proxy factor-wise distortion computed purely from trained hard codes.

    For each factor family, considers every pair of records whose canonical
    fingerprint *differs* in that family and reports the fraction whose
    learned hard codes fail to also differ. This measures whether the
    codec's bottleneck preserves each factor family's distinctions in
    isolation, without requiring a second regression-style training pass
    (most families here are not raw scalars, so a reconstruction-error
    metric would not be meaningful).
    """
    fingerprints = [plan_factor_fingerprints(plan) for plan in plans]
    n = len(plans)
    distortion: dict[str, float] = {}
    for factor in _FIXTURE_FINGERPRINT_FACTORS:
        total_pairs = 0
        undistinguished = 0
        for i in range(n):
            for j in range(i + 1, n):
                if fingerprints[i][factor] == fingerprints[j][factor]:
                    continue
                total_pairs += 1
                if torch.equal(codes[i], codes[j]):
                    undistinguished += 1
        distortion[factor] = undistinguished / total_pairs if total_pairs else 0.0
    return distortion


def evaluate_program_factor_arm(
    arm: ProgramFactorArm,
    plans: list[SemanticPlanV1],
    tensorizer_config: ProgramFactorTensorizerConfig | None = None,
) -> ProgramFactorResult:
    """Train and evaluate one program-factor latent-codec arm on a plan corpus."""
    start = time.monotonic()
    cfg = tensorizer_config or ProgramFactorTensorizerConfig()
    feature_dim = total_feature_dim(cfg)
    features = torch.stack(
        [concat_factor_tensors(tensorize_semantic_plan(plan, cfg)) for plan in plans]
    )
    num_records = len(plans)
    targets = torch.arange(num_records, dtype=torch.long)

    torch.manual_seed(arm.seed)
    model = _build_model(arm, num_records, feature_dim)
    train_latent_codec(model, features, targets, steps=arm.train_steps, lr=arm.lr)
    eval_metrics = evaluate_latent_codec(model, features, targets)
    exact_rate = eval_metrics["exact_reconstruction_rate"]
    occupied = eval_metrics["occupied_codewords"]
    capacity = arm.capacity
    # Continuous arms cannot leak in the discrete sense; only discrete
    # below-capacity arms reaching 100% reconstruction are leakage violations.
    leakage = exact_rate >= 1.0 and capacity < num_records and arm.codec != "continuous"

    no_bypass_ok = audit_no_bypass(model, features)

    with torch.no_grad():
        # Use encoding.hard rather than the model's returned code_index: for
        # the continuous codec family, code_index is a dummy all-zero
        # placeholder (there is no discrete flat index), so comparing it
        # would make every pair "undistinguished" regardless of the actual
        # latent vectors. encoding.hard is the real per-family hard code for
        # every codec (discrete symbols or the continuous latent itself).
        encoding = model.codec.encode(features, hard=True)
    factor_distortion = _factor_wise_distortion(plans, encoding.hard)

    storage = model.codec.physical_storage((num_records,))
    notes = [
        f"program-factor {arm.codec} codec hidden_dim={arm.hidden_dim} "
        f"steps={arm.train_steps} feature_dim={feature_dim}; "
        f"utilization={eval_metrics['utilization']:.4f} "
        f"entropy_bits={eval_metrics['empirical_entropy_bits']:.4f}"
    ]
    if not no_bypass_ok:
        notes.append("NO-BYPASS AUDIT FAILED: decoder input may not be codec-only")

    return ProgramFactorResult(
        arm_id=arm.arm_id,
        codec=arm.codec,
        codec_levels=model.codec.spec.levels,
        seed=arm.seed,
        num_records=num_records,
        capacity=capacity,
        exact_reconstruction_rate=exact_rate,
        occupied_codewords=occupied,
        code_utilization=occupied / max(1, capacity),
        empirical_entropy_bits=eval_metrics["empirical_entropy_bits"],
        nominal_bits=model.codec.nominal_bits(),
        serialized_bytes_per_example=storage.bytes_per_example,
        nominal_bytes_per_example=storage.nominal_bytes,
        no_bypass_ok=no_bypass_ok,
        leakage=leakage,
        factor_distortion=factor_distortion,
        elapsed_seconds=time.monotonic() - start,
        notes=tuple(notes),
    )


def run_matrix(
    *,
    mode: str = "fixture",
    corpus_path: Path | None = None,
    fixture_count: int = 32,
    fixture_seed: int = 0,
    seeds: tuple[int, ...] = (0,),
    arms_filter: tuple[str, ...] | None = None,
    tensorizer_config: ProgramFactorTensorizerConfig | None = None,
) -> ProgramFactorMatrixReport:
    """Run the CAP2-05 program-factor fixture/corpus matrix.

    ``mode="fixture"`` uses the deterministic SLM-144 fixture corpus and
    accepts no external manifest. ``mode="small_corpus"`` and
    ``mode="full_corpus"`` require an immutable JSONL manifest at
    ``corpus_path``; this iteration does not synthesize small/full corpora,
    so those modes fail closed with a clear error when no manifest is
    supplied rather than silently substituting the fixture corpus.
    """
    if mode == "fixture":
        if corpus_path is not None:
            raise ValueError("fixture mode does not accept corpus_path")
        plans = load_fixture_plans(count=fixture_count, seed=fixture_seed)
    elif mode in ("small_corpus", "full_corpus"):
        if corpus_path is None:
            raise ValueError(
                f"{mode!r} mode requires corpus_path pointing to an immutable "
                "SemanticPlanV1 JSONL manifest"
            )
        plans = load_corpus_manifest(corpus_path)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    arms = build_matrix(seeds=seeds, arms_filter=arms_filter)
    results = [evaluate_program_factor_arm(arm, plans, tensorizer_config) for arm in arms]
    run_id = _hash_run_id(("cap2-05", mode, len(plans), tuple(a.arm_id for a in arms), seeds))
    return ProgramFactorMatrixReport(
        run_id=run_id,
        mode=mode,
        num_records=len(plans),
        corpus_path=str(corpus_path) if corpus_path else None,
        arms=tuple(results),
    )
