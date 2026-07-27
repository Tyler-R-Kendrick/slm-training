"""CAP2-07 oracle-latent causal-necessity, support, robustness, and collapse audit
(AP-031 / SLM-330; milestone "Oracle Bottleneck").

Given the AP-029 (SLM-325) SemanticPlanV1 program-factor `LatentCodec` bridge
and the AP-030 (SLM-328) rate-distortion sweep's frozen-decoder-plus-linear-
regressor pipeline, this harness asks a different question than either: does
the *frozen* decoder actually depend causally on the retained oracle latent,
or could it reconstruct the same output from the prompt/decoder weights
alone?

It reuses -- never duplicates -- the AP-029/AP-030 stack unchanged:

* ``slm_training.models.semantic_plan_factors`` for deterministic
  ``SemanticPlanV1`` tensorization (the "gold target" every condition is
  scored against).
* ``slm_training.harnesses.experiments.cap2_rate_distortion_sweep``'s
  ``FactorReconstructionModel`` / ``train_factor_reconstruction`` /
  ``_build_codec`` / ``DISTORTION_THRESHOLD`` -- the exact frozen decoder and
  distortion-threshold convention AP-030 already established. This harness
  trains that same model once per arm, freezes it, and only ever perturbs
  the *latent fed into it* -- it introduces no new decode path.
* ``slm_training.models.latent_codec_trainer.audit_no_bypass`` -- the
  no-bypass audit that already caught encoder collapse in CAP2-06; a
  collapsed encoder is disqualified from a causal-necessity claim here too.
* ``slm_training.evals.power_protocol.bootstrap_paired_ci`` /
  ``exact_paired_binary_test`` -- the repo's existing paired-inference stats
  helpers (SLM-183), reused rather than reimplemented.

Scope decision (read before the results, mirroring CAP2-06's own "Scope
decision"): AP-029/AP-030 do not build a decoder from oracle latents back to
full OpenUI program text -- that is explicitly gated future work in both of
those design docs ("add one token/tree decoder only after factor gates
pass"). Without that decoder there is no serialized program to hand the real
``binding_aware_meaningful_v2`` scorer (``evals/meaningful_program.py``),
which takes a prediction *string*, not a factor tensor. Building that
program decoder now, just to satisfy this audit, would itself be a new,
unaudited decode path -- exactly what the "no shadow paths" / "never
introduce a new unconstrained decode path" constraints forbid. Instead this
harness defines and clearly labels a **strict-semantics proxy**: an example
counts as a strict semantic match only if *every* factor family's
reconstruction MSE is at or under AP-030's own ``DISTORTION_THRESHOLD``
(0.05, reused unchanged, not a new number). This is a proxy for strict
semantic exactness at the factor-reconstruction stage of the pipeline, not a
claim of numeric equivalence to meaning-v2 -- exactly the same honesty
posture CAP2-06 already documented for its own MSE gate.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any

import torch

from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_plan.corpus import build_fixture_plan_corpus
from slm_training.evals.power_protocol import bootstrap_paired_ci, exact_paired_binary_test
from slm_training.harnesses.experiments.cap2_rate_distortion_sweep import (
    DISTORTION_THRESHOLD,
    FactorReconstructionModel,
    RateDistortionCell,
    _build_codec,
    train_factor_reconstruction,
)
from slm_training.models.latent_codec import LatentCodec
from slm_training.models.latent_codec_trainer import audit_no_bypass
from slm_training.models.mixed_radix_fsq import MixedRadixFSQCodec, MixedRadixFSQConfig
from slm_training.models.semantic_plan_factors import (
    PROGRAM_FACTOR_FAMILIES,
    ProgramFactorTensorizerConfig,
    concat_factor_tensors,
    factor_dims,
    tensorize_semantic_plan,
    total_feature_dim,
)

__all__ = [
    "CAUSAL_AUDIT_CONDITIONS",
    "NOISE_SIGMA_MULTIPLIERS",
    "ROBUST_RADIUS_TOLERANCE",
    "ROBUST_RADIUS_SWEEP",
    "CausalAuditArm",
    "ConditionResult",
    "NoiseRadiusResult",
    "ArmCausalAuditResult",
    "CausalAuditReport",
    "load_causal_audit_corpus",
    "default_causal_audit_arms",
    "evaluate_arm",
    "run_causal_audit",
]

# The base perturbation conditions run for every arm (noise is swept
# separately at NOISE_SIGMA_MULTIPLIERS + the finer ROBUST_RADIUS_SWEEP).
# "zero" doubles as the prompt-only control per the issue's own wiring note
# (this stage of the pipeline has no separate prompt channel -- the decoder's
# only input is the codec's declared latent output, so masking the latent
# entirely *is* the prompt-only condition; see ArmCausalAuditResult notes).
CAUSAL_AUDIT_CONDITIONS: tuple[str, ...] = (
    "correct",
    "zero",
    "random",
    "shuffled_between_example",
    "slot_permuted",
    "role_swapped",
    "quantized",
)

NOISE_SIGMA_MULTIPLIERS: tuple[float, ...] = (0.25, 0.5, 1.0)
ROBUST_RADIUS_SWEEP: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
ROBUST_RADIUS_TOLERANCE: float = 0.05


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _hash_run_id(parts: tuple[Any, ...]) -> str:
    import hashlib

    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


@dataclass(frozen=True)
class CausalAuditArm:
    """One (codec, capacity) arm of the CAP2-07 causal-necessity audit.

    ``codec="fsq_typed"`` is the one non-``_build_codec`` family: it builds a
    heterogeneous-radix ``MixedRadixFSQCodec`` (reusing AP-029's own
    ``plan_fsq_2_3_3_4_5`` radix vector unchanged) so ``role_swapped`` has at
    least one pair of same-typed slots to swap. Every other codec name
    (``uniform_scalar`` / ``fsq`` / ``lfq`` / ``vq`` / ``continuous``) is
    built via the exact AP-030 ``_build_codec`` unchanged.
    """

    arm_id: str
    codec: str
    num_latents: int = 2
    width: int = 64
    seed: int = 0
    train_steps: int = 400
    lr: float = 2e-2
    radixes: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ConditionResult:
    """Measured result for one perturbation condition, paired against ``correct``."""

    condition: str
    n: int
    strict_semantics_proxy_rate: float
    mean_mse_overall: float
    rate_ci_vs_correct: dict[str, Any]
    mse_ci_vs_correct: dict[str, Any]
    exact_paired_test_vs_correct: dict[str, Any] | None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "n": self.n,
            "strict_semantics_proxy_rate": self.strict_semantics_proxy_rate,
            "mean_mse_overall": self.mean_mse_overall,
            "rate_ci_vs_correct": self.rate_ci_vs_correct,
            "mse_ci_vs_correct": self.mse_ci_vs_correct,
            "exact_paired_test_vs_correct": self.exact_paired_test_vs_correct,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class NoiseRadiusResult:
    """One point on the fine noise-radius sweep used to find the robust radius."""

    sigma_multiplier: float
    strict_semantics_proxy_rate: float
    mean_mse_overall: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sigma_multiplier": self.sigma_multiplier,
            "strict_semantics_proxy_rate": self.strict_semantics_proxy_rate,
            "mean_mse_overall": self.mean_mse_overall,
        }


@dataclass(frozen=True)
class ArmCausalAuditResult:
    """Full causal-necessity audit for one (codec, capacity) arm."""

    arm_id: str
    codec: str
    num_latents: int
    width: int
    seed: int
    num_holdout: int
    no_bypass_ok: bool
    latent_dim: int
    effective_rank: float
    stable_rank: float
    spectral_entropy: float
    total_variance: float
    mean_variance: float
    conditions: tuple[ConditionResult, ...]
    noise_sweep: tuple[NoiseRadiusResult, ...]
    robust_semantic_radius: float | None
    robust_semantic_radius_censored: bool
    role_swap_applicable: bool
    weaker_decoder_note: str
    causally_necessary: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "codec": self.codec,
            "num_latents": self.num_latents,
            "width": self.width,
            "seed": self.seed,
            "num_holdout": self.num_holdout,
            "no_bypass_ok": self.no_bypass_ok,
            "latent_dim": self.latent_dim,
            "effective_rank": self.effective_rank,
            "stable_rank": self.stable_rank,
            "spectral_entropy": self.spectral_entropy,
            "total_variance": self.total_variance,
            "mean_variance": self.mean_variance,
            "conditions": [c.to_dict() for c in self.conditions],
            "noise_sweep": [r.to_dict() for r in self.noise_sweep],
            "robust_semantic_radius": self.robust_semantic_radius,
            "robust_semantic_radius_censored": self.robust_semantic_radius_censored,
            "role_swap_applicable": self.role_swap_applicable,
            "weaker_decoder_note": self.weaker_decoder_note,
            "causally_necessary": self.causally_necessary,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CausalAuditReport:
    """Full CAP2-07 causal-necessity/support/robustness/collapse report."""

    run_id: str
    fixture_count: int
    fixture_seed: int
    num_train: int
    num_holdout: int
    arms: tuple[ArmCausalAuditResult, ...]
    disposition: str
    disposition_reasons: tuple[str, ...]
    version: str = "cap2-07-v1"
    timestamp: str = field(default_factory=lambda: _utc_now())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "fixture_count": self.fixture_count,
            "fixture_seed": self.fixture_seed,
            "num_train": self.num_train,
            "num_holdout": self.num_holdout,
            "arms": [a.to_dict() for a in self.arms],
            "disposition": self.disposition,
            "disposition_reasons": list(self.disposition_reasons),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def load_causal_audit_corpus(
    fixture_count: int = 32,
    fixture_seed: int = 0,
) -> tuple[list[SemanticPlanV1], list[SemanticPlanV1]]:
    """Deterministic SLM-144 fixture corpus, 80/20 train/holdout split.

    The holdout (val) split is the "held-out evaluation sample" the issue
    asks for: the frozen decoder is trained only on the train split, and
    every causal-audit condition below is evaluated only on holdout plans.
    """
    corpus = build_fixture_plan_corpus(count=fixture_count, seed=fixture_seed)
    train_plans = [plan for _, plan in corpus["train"]]
    holdout_plans = [plan for _, plan in corpus["val"]]
    if len(holdout_plans) < 2:
        raise ValueError(
            f"holdout split has {len(holdout_plans)} record(s); need >= 2 for a "
            "between-example shuffle control. Increase --fixture-count."
        )
    return train_plans, holdout_plans


def default_causal_audit_arms(seed: int = 0) -> list[CausalAuditArm]:
    """Default arms: the AP-030 retained (smallest-passing, non-collapsed)
    configurations for two codec families, plus one heterogeneous-radix FSQ
    arm reused unchanged from AP-029's own ``plan_fsq_2_3_3_4_5`` so
    ``role_swapped`` has same-typed slots to swap.

    ``uniform_scalar``/``vq`` at minimum capacity are known (CAP2-06) to
    collapse the ``semantic_trace`` encoder on this tiny fixture corpus;
    they are omitted from the default set (still selectable via CLI/API) so
    the default run does not spend cycles on an arm expected to fail its own
    no-bypass gate.
    """
    return [
        CausalAuditArm("continuous_n2_w64", "continuous", num_latents=2, width=64, seed=seed, train_steps=800),
        CausalAuditArm("lfq_n2_w64", "lfq", num_latents=2, width=64, seed=seed, train_steps=800),
        CausalAuditArm(
            "fsq_typed_2_3_3_4_5",
            "fsq_typed",
            radixes=(2, 3, 3, 4, 5),
            width=64,
            seed=seed,
            train_steps=1000,
        ),
    ]


def _build_arm_codec(arm: CausalAuditArm, feature_dim: int) -> LatentCodec:
    if arm.codec == "fsq_typed":
        if arm.radixes is None:
            raise ValueError(f"fsq_typed arm {arm.arm_id} requires radixes")
        cfg = MixedRadixFSQConfig(
            num_states=1,
            levels=arm.radixes,
            hidden_dim=arm.width,
            feature_dim=feature_dim,
            mode="semantic_trace",
        )
        return MixedRadixFSQCodec(cfg)
    cell = RateDistortionCell(arm.codec, arm.num_latents, arm.width, seed=arm.seed, train_steps=arm.train_steps)
    return _build_codec(cell, feature_dim)


def _slot_levels(codec: LatentCodec, decoder_input_dim: int) -> list[int]:
    """Per-slot widths of the decoder input.

    For the concatenated-one-hot families (uniform_scalar / fsq / fsq_typed)
    this is exactly ``codec.spec.levels`` -- the same per-coordinate alphabet
    sizes ``_one_hot_mixed_radix`` already concatenates in that order. For
    element-wise families (lfq / vq / continuous) each raw coordinate is its
    own slot (width 1).
    """
    if codec.spec.name in ("uniform_scalar", "mixed_radix_fsq"):
        levels = list(codec.spec.levels)
        if sum(levels) == decoder_input_dim:
            return levels
    return [1] * decoder_input_dim


def _segment_ranges(levels: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for width in levels:
        ranges.append((start, start + width))
        start += width
    return ranges


def _permute_by_segments(decoder_input: torch.Tensor, ranges: list[tuple[int, int]], order: list[int]) -> torch.Tensor:
    return torch.cat([decoder_input[:, ranges[i][0] : ranges[i][1]] for i in order], dim=-1)


def _arbitrary_slot_permutation(num_slots: int, rng: random.Random) -> list[int]:
    order = list(range(num_slots))
    rng.shuffle(order)
    return order


def _role_preserving_permutation(levels: list[int], rng: random.Random) -> list[int] | None:
    """Permute slot order but only ever swap slots that share the same width
    ("role"). Returns ``None`` when the restriction would not actually
    restrict anything relative to ``slot_permuted``: either every slot
    shares one homogeneous width (a single group -- e.g. binary LFQ or any
    element-wise continuous/VQ latent, where every raw coordinate trivially
    has width 1), so role-preserving permutations are the *entire*
    permutation space, or every same-width group has exactly one member (no
    pair to swap within any group), so no permutation is possible at all. In
    both cases the result would be identical to or indistinguishable from
    ``slot_permuted``, so the arm is marked not-applicable rather than
    fabricating a separately distinguishable result.
    """
    groups: dict[int, list[int]] = {}
    for idx, width in enumerate(levels):
        groups.setdefault(width, []).append(idx)
    if len(groups) <= 1:
        return None
    swappable_groups = [g for g in groups.values() if len(g) >= 2]
    if not swappable_groups:
        return None
    order = list(range(len(levels)))
    for group in swappable_groups:
        shuffled = list(group)
        # Force a real derangement within the group when possible so the
        # permutation is never accidentally the identity.
        attempts = 0
        while attempts < 20 and (len(shuffled) < 2 or all(a == b for a, b in zip(shuffled, group))):
            rng.shuffle(shuffled)
            attempts += 1
        for src, dst in zip(group, shuffled):
            order[src] = dst
    return order


def _derangement(n: int, rng: random.Random) -> list[int]:
    """A random permutation of ``range(n)`` with no fixed points (n >= 2)."""
    if n < 2:
        raise ValueError("derangement requires n >= 2")
    attempts = 0
    order = list(range(n))
    while attempts < 200:
        rng.shuffle(order)
        if all(order[i] != i for i in range(n)):
            return order
        attempts += 1
    # Deterministic fallback: a full rotation is always a derangement for n >= 2.
    return [(i + 1) % n for i in range(n)]


def _latent_spectrum_stats(matrix: torch.Tensor) -> dict[str, float]:
    """SVD-based effective-rank / variance diagnostics over a retained-latent
    matrix ``[N, D]``.

    Follows the same von-Neumann-entropy effective-rank convention already
    used by the SLM-214/SLM-217 spectral-snapshot harnesses
    (``exp(entropy(normalized squared singular values))``), reimplemented
    locally rather than importing those modules' private helpers -- the same
    choice those two harnesses independently made from each other.
    """
    if matrix.numel() == 0 or matrix.shape[0] < 2:
        return {
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "spectral_entropy": 0.0,
            "total_variance": 0.0,
            "mean_variance": 0.0,
        }
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(centered.to(torch.float64))
    frob2 = float((s * s).sum())
    spec2 = float(s[0] * s[0]) if s.numel() else 0.0
    stable_rank = frob2 / spec2 if spec2 > 0 else 0.0
    if frob2 > 0:
        probs = (s * s) / frob2
        entropy = float(-(probs * torch.log(probs.clamp_min(1e-30))).sum())
        effective_rank = float(torch.exp(torch.tensor(entropy)))
    else:
        entropy = 0.0
        effective_rank = 0.0
    variance = matrix.var(dim=0, unbiased=False)
    return {
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "spectral_entropy": entropy,
        "total_variance": float(variance.sum()),
        "mean_variance": float(variance.mean()),
    }


def _strict_match_per_example(
    recon: torch.Tensor,
    target: torch.Tensor,
    dims: dict[str, int],
) -> torch.Tensor:
    """Per-example strict-semantics proxy: True iff every factor family's
    reconstruction MSE is at/under AP-030's ``DISTORTION_THRESHOLD``.
    """
    n = recon.shape[0]
    ok = torch.ones(n, dtype=torch.bool)
    offset = 0
    for name in PROGRAM_FACTOR_FAMILIES:
        width = dims[name]
        family_mse = ((recon[:, offset : offset + width] - target[:, offset : offset + width]) ** 2).mean(dim=-1)
        ok &= family_mse <= DISTORTION_THRESHOLD
        offset += width
    return ok


def _score_decoder_input(
    model: FactorReconstructionModel,
    decoder_input: torch.Tensor,
    target: torch.Tensor,
    dims: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen decoder on a (possibly perturbed) decoder input.

    Returns ``(per_example_mse_overall, per_example_strict_match)``. Never
    touches ``model.codec`` -- this is exactly the "hold the decoder fixed,
    perturb only the latent" design the audit requires.
    """
    model.eval()
    with torch.no_grad():
        recon = model.decoder(decoder_input)
        mse = ((recon - target) ** 2).mean(dim=-1)
        strict = _strict_match_per_example(recon, target, dims)
    return mse, strict


def _ci_low_is_positive(ci: dict[str, Any]) -> bool:
    low = ci.get("low", float("nan"))
    return isinstance(low, (int, float)) and not math.isnan(low) and low > 0.0


def _paired_metrics(
    condition: str,
    correct_mse: torch.Tensor,
    correct_strict: torch.Tensor,
    cond_mse: torch.Tensor,
    cond_strict: torch.Tensor,
    *,
    resamples: int,
    ci_seed: int,
    notes: tuple[str, ...] = (),
) -> ConditionResult:
    n = int(cond_mse.shape[0])
    rate_ci = bootstrap_paired_ci(
        correct_strict.float().tolist(),
        cond_strict.float().tolist(),
        metric=lambda left, right: float(sum(left) / len(left) - sum(right) / len(right)),
        seed=ci_seed,
        resamples=resamples,
    )
    mse_ci = bootstrap_paired_ci(
        correct_mse.tolist(),
        cond_mse.tolist(),
        metric=lambda left, right: float(sum(right) / len(right) - sum(left) / len(left)),
        seed=ci_seed,
        resamples=resamples,
    )
    exact_test: dict[str, Any] | None
    try:
        exact_test = exact_paired_binary_test(
            [int(v) for v in correct_strict.tolist()],
            [int(v) for v in cond_strict.tolist()],
        )
    except ValueError:
        exact_test = None
    return ConditionResult(
        condition=condition,
        n=n,
        strict_semantics_proxy_rate=float(cond_strict.float().mean().item()),
        mean_mse_overall=float(cond_mse.mean().item()),
        rate_ci_vs_correct=rate_ci,
        mse_ci_vs_correct=mse_ci,
        exact_paired_test_vs_correct=exact_test,
        notes=notes,
    )


def evaluate_arm(
    arm: CausalAuditArm,
    train_plans: list[SemanticPlanV1],
    holdout_plans: list[SemanticPlanV1],
    *,
    tensorizer_config: ProgramFactorTensorizerConfig | None = None,
    ci_resamples: int = 1000,
) -> ArmCausalAuditResult:
    """Train the frozen decoder on ``train_plans``, then run every causal-audit
    condition against the fixed held-out ``holdout_plans`` sample.
    """
    cfg = tensorizer_config or ProgramFactorTensorizerConfig()
    dims = factor_dims(cfg)
    feature_dim = total_feature_dim(cfg)
    features_train = torch.stack([concat_factor_tensors(tensorize_semantic_plan(p, cfg)) for p in train_plans])
    features_holdout = torch.stack([concat_factor_tensors(tensorize_semantic_plan(p, cfg)) for p in holdout_plans])

    # Decoder weights are trained once, then frozen for every condition below
    # -- the "decoder seed held fixed" requirement. There is no separate
    # prompt channel or mask schedule at this stage of the pipeline (the
    # decoder's only input is the codec's declared latent output), so both
    # are trivially held fixed by construction; see the module docstring's
    # scope decision.
    torch.manual_seed(arm.seed)
    codec = _build_arm_codec(arm, feature_dim)
    model = FactorReconstructionModel(codec, feature_dim)
    train_factor_reconstruction(model, features_train, steps=arm.train_steps, lr=arm.lr)
    model.eval()

    no_bypass_ok = audit_no_bypass(model, features_holdout)

    with torch.no_grad():
        encoding = codec.encode(features_holdout, hard=True)
        decoder_input_correct = codec.decode_input(encoding)
    n_holdout, latent_dim = decoder_input_correct.shape

    spectrum = _latent_spectrum_stats(decoder_input_correct)

    # A single generator seeded from arm.seed drives every stochastic
    # perturbation across every condition in this arm, in a fixed order --
    # a deterministic, reproducible "perturbation seed" held constant for
    # the whole paired comparison (never redrawn to favor one condition).
    rng = random.Random(arm.seed * 1_000_003 + 7919)

    mean_vec = decoder_input_correct.mean(dim=0)
    std_vec = decoder_input_correct.std(dim=0, unbiased=False).clamp_min(1e-6)

    correct_mse, correct_strict = _score_decoder_input(model, decoder_input_correct, features_holdout, dims)

    notes: list[str] = [
        "strict_semantics_proxy_rate is an MSE-threshold proxy (AP-030's "
        f"DISTORTION_THRESHOLD={DISTORTION_THRESHOLD}), not the real "
        "binding_aware_meaningful_v2 scorer -- no decoder from factors to "
        "full OpenUI program text exists yet in this repo (see module scope "
        "decision).",
        "the 'zero' condition doubles as the prompt-only control: this "
        "pipeline stage has no separate prompt input to the decoder.",
    ]
    if not no_bypass_ok:
        notes.append(
            "NO-BYPASS AUDIT FAILED for this arm: the semantic_trace encoder "
            "collapsed to a constant/dead code (same failure mode CAP2-06 "
            "documented at minimum capacity). Any 'causal necessity' result "
            "below is not trustworthy for this arm -- see disposition."
        )

    levels = _slot_levels(codec, latent_dim)
    ranges = _segment_ranges(levels)
    num_slots = len(levels)
    slot_order = _arbitrary_slot_permutation(num_slots, rng)
    role_order = _role_preserving_permutation(levels, rng)
    role_swap_applicable = role_order is not None

    conditions: list[ConditionResult] = []
    for condition in CAUSAL_AUDIT_CONDITIONS:
        if condition == "correct":
            continue
        if condition == "role_swapped" and not role_swap_applicable:
            continue
        if condition == "zero":
            perturbed = torch.zeros_like(decoder_input_correct)
            cond_notes: tuple[str, ...] = ("shares code with the prompt-only control",)
        elif condition == "random":
            perturbed = mean_vec.unsqueeze(0) + std_vec.unsqueeze(0) * torch.randn(
                n_holdout, latent_dim, generator=torch.Generator().manual_seed(rng.randrange(2**31))
            )
            cond_notes = ("fresh draw per-dimension N(mean, std) matched to the correct-latent marginal",)
        elif condition == "shuffled_between_example":
            order = _derangement(n_holdout, rng)
            perturbed = decoder_input_correct[order]
            cond_notes = ()
        elif condition == "slot_permuted":
            perturbed = _permute_by_segments(decoder_input_correct, ranges, slot_order)
            cond_notes = ()
        elif condition == "role_swapped":
            assert role_order is not None
            perturbed = _permute_by_segments(decoder_input_correct, ranges, role_order)
            cond_notes = ("permutation constrained to same-width ('same-role') slots only",)
        elif condition == "quantized":
            with torch.no_grad():
                fresh_encoding = codec.encode(features_holdout, hard=True)
                perturbed = codec.decode_input(fresh_encoding)
            cond_notes = (
                "sanity control: re-runs hard encode fresh with no perturbation; "
                "should reproduce 'correct' exactly (deterministic pipeline check).",
            )
        else:  # pragma: no cover - exhaustive above
            raise ValueError(f"unknown condition {condition!r}")

        cond_mse, cond_strict = _score_decoder_input(model, perturbed, features_holdout, dims)
        conditions.append(
            _paired_metrics(
                condition,
                correct_mse,
                correct_strict,
                cond_mse,
                cond_strict,
                resamples=ci_resamples,
                ci_seed=arm.seed,
                notes=cond_notes,
            )
        )

    if not role_swap_applicable:
        notes.append(
            "role_swapped omitted: this arm's codec has no >=2 distinct slot "
            "widths ('roles') with a same-width pair to swap -- homogeneous "
            "codecs (e.g. binary LFQ, element-wise continuous/VQ) would make "
            "a role-preserving permutation identical to slot_permuted; not "
            "fabricated as a separate result."
        )

    # Noise sweep at the 3 required radii, plus a finer sweep for the robust
    # radius search. Both reuse the same additive-Gaussian perturbation.
    noise_gen = torch.Generator().manual_seed(rng.randrange(2**31))

    def _noise_at(multiplier: float) -> tuple[torch.Tensor, torch.Tensor]:
        if multiplier == 0.0:
            return _score_decoder_input(model, decoder_input_correct, features_holdout, dims)
        noisy = decoder_input_correct + multiplier * std_vec.unsqueeze(0) * torch.randn(
            n_holdout, latent_dim, generator=noise_gen
        )
        return _score_decoder_input(model, noisy, features_holdout, dims)

    for multiplier in NOISE_SIGMA_MULTIPLIERS:
        cond_mse, cond_strict = _noise_at(multiplier)
        conditions.append(
            _paired_metrics(
                f"noise_perturbed_sigma{multiplier}",
                correct_mse,
                correct_strict,
                cond_mse,
                cond_strict,
                resamples=ci_resamples,
                ci_seed=arm.seed,
                notes=(f"additive Gaussian noise at {multiplier} x per-dimension std",),
            )
        )

    correct_rate = float(correct_strict.float().mean().item())
    noise_sweep: list[NoiseRadiusResult] = []
    robust_radius: float | None = None
    censored = True
    for multiplier in ROBUST_RADIUS_SWEEP:
        cond_mse, cond_strict = _noise_at(multiplier)
        rate = float(cond_strict.float().mean().item())
        noise_sweep.append(NoiseRadiusResult(multiplier, rate, float(cond_mse.mean().item())))
        if rate >= correct_rate - ROBUST_RADIUS_TOLERANCE:
            robust_radius = multiplier
        else:
            censored = False
            break

    # Causal necessity: correct must materially outperform zero/random/
    # shuffled with a paired-CI lower bound strictly above zero, and the
    # encoder must not have collapsed.
    destructive = {"zero", "random", "shuffled_between_example"}
    destructive_results = [c for c in conditions if c.condition in destructive]
    causally_necessary = (
        no_bypass_ok
        and bool(destructive_results)
        and all(_ci_low_is_positive(c.rate_ci_vs_correct) for c in destructive_results)
    )

    weaker_decoder_note = (
        "omitted: no smaller/weaker frozen-decoder checkpoint fixture exists "
        "in this repo (src/slm_training/resources/checkpoints/ has none for "
        "this factor-reconstruction stage) -- not fabricated."
    )

    return ArmCausalAuditResult(
        arm_id=arm.arm_id,
        codec=arm.codec,
        num_latents=arm.num_latents,
        width=arm.width,
        seed=arm.seed,
        num_holdout=n_holdout,
        no_bypass_ok=no_bypass_ok,
        latent_dim=latent_dim,
        effective_rank=spectrum["effective_rank"],
        stable_rank=spectrum["stable_rank"],
        spectral_entropy=spectrum["spectral_entropy"],
        total_variance=spectrum["total_variance"],
        mean_variance=spectrum["mean_variance"],
        conditions=tuple(conditions),
        noise_sweep=tuple(noise_sweep),
        robust_semantic_radius=robust_radius,
        robust_semantic_radius_censored=censored,
        role_swap_applicable=role_swap_applicable,
        weaker_decoder_note=weaker_decoder_note,
        causally_necessary=causally_necessary,
        notes=tuple(notes),
    )


def run_causal_audit(
    *,
    fixture_count: int = 32,
    fixture_seed: int = 0,
    arms: list[CausalAuditArm] | None = None,
    tensorizer_config: ProgramFactorTensorizerConfig | None = None,
    ci_resamples: int = 1000,
) -> CausalAuditReport:
    """Run the CAP2-07 causal-necessity/support/robustness/collapse audit."""
    train_plans, holdout_plans = load_causal_audit_corpus(fixture_count, fixture_seed)
    arm_configs = arms if arms is not None else default_causal_audit_arms()
    results = [
        evaluate_arm(a, train_plans, holdout_plans, tensorizer_config=tensorizer_config, ci_resamples=ci_resamples)
        for a in arm_configs
    ]

    reasons: list[str] = []
    trustworthy = [r for r in results if r.no_bypass_ok]
    if not trustworthy:
        disposition = "stop_ignore"
        reasons.append(
            "every evaluated arm failed the no-bypass audit (collapsed encoder); "
            "no trustworthy causal-necessity evidence was produced."
        )
    elif all(r.causally_necessary for r in trustworthy):
        disposition = "continue"
        reasons.append(
            "correct latents materially outperformed zero/random/shuffled "
            "controls (paired-CI lower bound > 0) for every trustworthy arm."
        )
    elif any(r.causally_necessary for r in trustworthy):
        disposition = "continue"
        reasons.append(
            "correct latents materially outperformed destructive controls for "
            "some but not all evaluated arms; see per-arm causally_necessary."
        )
    else:
        disposition = "stop_ignore"
        reasons.append(
            "no trustworthy arm showed a materially positive paired-CI effect "
            "of destroying the latent -- latent destruction had little "
            "measured effect on strict-semantics-proxy score."
        )
    collapsed = [r.arm_id for r in results if not r.no_bypass_ok]
    if collapsed:
        reasons.append(f"encoder collapse (no_bypass_ok=False) in arms: {', '.join(collapsed)}")

    run_id = _hash_run_id(
        ("cap2-07", fixture_count, fixture_seed, tuple(a.arm_id for a in arm_configs))
    )
    return CausalAuditReport(
        run_id=run_id,
        fixture_count=fixture_count,
        fixture_seed=fixture_seed,
        num_train=len(train_plans),
        num_holdout=len(holdout_plans),
        arms=tuple(results),
        disposition=disposition,
        disposition_reasons=tuple(reasons),
    )
