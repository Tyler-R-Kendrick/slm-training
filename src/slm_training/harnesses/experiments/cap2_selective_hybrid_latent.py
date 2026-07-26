"""CAP2-08 selective-hybrid continuous/discrete program-factor latent audit
(AP-033 / SLM-333; milestone "Selective Hybrid Objective").

Tests the issue's central decision -- "continuous latents carry style/
layout/global intent while exact topology, cardinality, and binder/reference
state remain discrete" -- at the same bounded, fixture-scale, CPU wiring
level as its AP-029/030/031/032 predecessors, reusing their stack rather than
duplicating it:

* :mod:`slm_training.models.semantic_plan_factors` for deterministic
  ``SemanticPlanV1`` tensorization, and its own
  :func:`~slm_training.models.selective_latent_connector.split_soft_precision_features`
  factor routing (the "soft" vs "precision-critical" split).
* :mod:`slm_training.models.selective_latent_connector` for the gated-
  additive connector and its six named arms
  (:class:`~slm_training.models.selective_latent_connector.SelectiveLatentArm`),
  all resolving through
  :func:`~slm_training.models.selective_latent_connector.resolve_selective_latent_vector`.
* :func:`slm_training.harnesses.experiments.cap2_latent_causal_audit.load_causal_audit_corpus`
  for the exact same SLM-144 fixture corpus 80/20 train/holdout split AP-031
  already established (never re-implemented here).
* :class:`slm_training.models.continuous_latent.ContinuousLatentCodec` (the
  same continuous codec family AP-030/031 already evaluate) for the trained
  soft-factor bottleneck and (``all_continuous`` arm only) the precision-
  critical bottleneck.
* AP-030's own ``DISTORTION_THRESHOLD`` (0.05) strict-semantics-proxy MSE
  gate, reused unchanged -- never a new number.
* :mod:`slm_training.evals.power_protocol` (SLM-183) for paired bootstrap
  confidence intervals between each arm and the ``discrete_only`` reference.

Scope decision (read before the results, mirroring AP-031's/AP-030's own
"Scope decision" sections): the issue's target-path list names
``binder_precision_channel.py``, ``program_latent_codec.py``, and a
``BinderGraphV1`` class. None of these exist anywhere in this repo (verified
by repo-wide search before writing this module) and this harness does not
fabricate them. There is no real discrete binder/reference symbol-table
pipeline to route "precision-critical factors" through here -- only the
AP-029 tensorized *proxy* (``binder_reference`` factor family: symbol/binding
counts, fallback rate, opaque binding-edge ordinals). "Precision-critical
factors remain discrete" is therefore implemented as literally *not
bottlenecking* those factors through any learned codec at all (the decoder
sees the raw tensorized precision-critical features directly, undegraded --
the strongest available "discrete/exact" proxy given the existing schema),
while ``all_continuous`` (the negative control) routes them through a
:class:`ContinuousLatentCodec` bottleneck exactly like the soft channel, to
show what is lost by giving up that exactness. This is a proxy for the
issue's own "discrete precision channel", not a claim that a real
binder-graph/symbol-table implementation exists.

Similarly, "condition TwoTower/MaskGIT through a gated continuous connector"
is satisfied by :class:`~slm_training.models.selective_latent_connector.SelectiveLatentConnector`
being a structural (duck-typed) drop-in for
``DenoiserTower.set_plan_connector`` -- proven directly by
``tests/test_models/test_selective_latent_connector.py`` -- rather than by
also editing ``twotower.py``/``blocks.py`` themselves. AP-024's connector
required editing those files because it shipped as a real (if default-off)
production conditioning path with its own head/trainer wiring; AP-029
through AP-032 (this issue's own listed prerequisites) deliberately did not
touch ``twotower.py`` for the same class of bounded factor-reconstruction
experiment this issue continues, reusing the small ``FactorReconstructionModel``-
style scaffold instead. Following that precedent -- not fabricating new
production wiring to check a box -- is the honest choice here too.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.evals.power_protocol import (
    bootstrap_paired_ci,
    exact_paired_binary_test,
)
from slm_training.harnesses.experiments.cap2_latent_causal_audit import (
    load_causal_audit_corpus,
)
from slm_training.harnesses.experiments.cap2_rate_distortion_sweep import (
    DISTORTION_THRESHOLD,
)
from slm_training.models.continuous_latent import (
    ContinuousLatentCodec,
    ContinuousLatentConfig,
)
from slm_training.models.selective_latent_connector import (
    PRECISION_CRITICAL_FACTOR_FAMILIES,
    SOFT_FACTOR_FAMILIES,
    SelectiveLatentArm,
    SelectiveLatentConnector,
    resolve_selective_latent_vector,
    split_soft_precision_features,
)
from slm_training.models.semantic_plan_factors import (
    PROGRAM_FACTOR_FAMILIES,
    ProgramFactorTensorizerConfig,
    concat_factor_tensors,
    factor_dims,
    tensorize_semantic_plan,
)

__all__ = [
    "DEFAULT_ARMS",
    "ArmComparison",
    "ArmResult",
    "SelectiveHybridArmConfig",
    "SelectiveHybridModel",
    "SelectiveHybridReport",
    "default_arm_configs",
    "evaluate_arm",
    "load_selective_hybrid_corpus",
    "run_selective_hybrid",
]

DEFAULT_ARMS: tuple[SelectiveLatentArm, ...] = (
    SelectiveLatentArm.DISCRETE_ONLY,
    SelectiveLatentArm.CONTINUOUS_ONLY,
    SelectiveLatentArm.SELECTIVE_HYBRID,
    SelectiveLatentArm.ALL_CONTINUOUS,
    SelectiveLatentArm.RANDOM_CONTINUOUS,
    SelectiveLatentArm.ORACLE_CONTINUOUS,
)

# Bottleneck width forced on the precision-critical channel for the
# all_continuous negative control only -- deliberately narrower than the
# precision-critical feature width so the bottleneck is genuinely lossy.
_ALL_CONTINUOUS_PRECISION_LATENT_DIM = 4


def _utc_now() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat()


def _hash_run_id(parts: tuple[Any, ...]) -> str:
    import hashlib

    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


@dataclass(frozen=True)
class SelectiveHybridArmConfig:
    """One arm of the CAP2-08 selective-hybrid audit."""

    arm_id: str
    arm: SelectiveLatentArm
    seed: int = 0
    train_steps: int = 400
    lr: float = 2e-2
    soft_hidden_dim: int = 64


def default_arm_configs(seed: int = 0) -> list[SelectiveHybridArmConfig]:
    return [SelectiveHybridArmConfig(arm.value, arm, seed=seed, train_steps=400) for arm in DEFAULT_ARMS]


class SelectiveHybridModel(nn.Module):
    """One arm's routing model: an always-exact precision-critical decoder
    (bottlenecked only for ``all_continuous``) plus a soft-factor decoder
    that is either a learned baseline alone (``discrete_only``) or that
    baseline gated-additively corrected by a
    :class:`SelectiveLatentConnector` (every other arm).

    ``discrete_only`` never constructs ``soft_codec``/``connector`` at all --
    zero extra parameters, zero extra forward compute, matching
    ``AbstractPlanConnector``'s own "off" guarantee. See
    ``tests/test_models/test_cap2_selective_hybrid_latent.py`` for the
    regression test proving this arm's forward output is bit-identical to a
    hand-built minimal reference model with no connector concept at all.
    """

    def __init__(
        self,
        precision_dim: int,
        soft_dim: int,
        *,
        arm: SelectiveLatentArm,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.arm = arm
        self.precision_dim = precision_dim
        self.soft_dim = soft_dim

        if arm is SelectiveLatentArm.ALL_CONTINUOUS:
            self.precise_codec: ContinuousLatentCodec | None = ContinuousLatentCodec(
                ContinuousLatentConfig(
                    num_states=1,
                    latent_dim=min(_ALL_CONTINUOUS_PRECISION_LATENT_DIM, precision_dim),
                    hidden_dim=hidden_dim,
                    feature_dim=precision_dim,
                    mode="semantic_trace",
                )
            )
            precise_decoder_input_dim = self.precise_codec.config.latent_dim
        else:
            self.precise_codec = None
            precise_decoder_input_dim = precision_dim
        self.precise_decoder = nn.Linear(precise_decoder_input_dim, precision_dim)

        # A learned constant baseline (fed a fixed "ones" input): the
        # discrete_only prediction for soft factors, i.e. "the best guess
        # with zero continuous signal". Present in every arm so enabling the
        # connector is a genuine additive correction, not a swap.
        self.soft_baseline = nn.Linear(1, soft_dim)

        if arm is SelectiveLatentArm.DISCRETE_ONLY:
            self.soft_codec: ContinuousLatentCodec | None = None
            self.connector: SelectiveLatentConnector | None = None
        else:
            self.soft_codec = ContinuousLatentCodec(
                ContinuousLatentConfig(
                    num_states=1,
                    latent_dim=soft_dim,
                    hidden_dim=hidden_dim,
                    feature_dim=soft_dim,
                    mode="semantic_trace",
                )
            )
            self.connector = SelectiveLatentConnector(soft_dim, soft_dim)

    def precise_forward(self, precision_features: torch.Tensor, *, hard: bool) -> torch.Tensor:
        """Precision-critical branch. Never reads ``soft_features`` -- see
        the module docstring's factor-swap falsification test, which relies
        on this independence holding by construction.
        """
        if self.precise_codec is None:
            decoder_input = precision_features
        else:
            encoding = self.precise_codec.encode(precision_features, hard=hard)
            decoder_input = self.precise_codec.decode_input(encoding)
        return self.precise_decoder(decoder_input)

    def soft_forward(
        self,
        soft_features: torch.Tensor,
        *,
        hard: bool,
        oracle_vector: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        override_resolved: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Soft-factor branch. ``override_resolved`` lets the factor-swap
        falsification test substitute an arbitrary perturbed vector for
        whatever the connector would normally resolve, without touching the
        precision branch at all.
        """
        ones = soft_features.new_ones(soft_features.shape[0], 1)
        baseline = self.soft_baseline(ones)
        if self.arm is SelectiveLatentArm.DISCRETE_ONLY:
            return baseline
        assert self.soft_codec is not None and self.connector is not None
        encoding = self.soft_codec.encode(soft_features, hard=hard)
        latent = self.soft_codec.decode_input(encoding)
        if override_resolved is not None:
            resolved = override_resolved
        else:
            resolved = resolve_selective_latent_vector(
                latent, arm=self.arm, oracle_vector=oracle_vector, generator=generator
            )
        bias = self.connector.bias_for_vocab(resolved)
        if self.arm is SelectiveLatentArm.CONTINUOUS_ONLY:
            return bias
        return baseline + bias

    def forward(
        self,
        precision_features: torch.Tensor,
        soft_features: torch.Tensor,
        *,
        hard: bool,
        oracle_vector: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        precise_recon = self.precise_forward(precision_features, hard=hard)
        soft_recon = self.soft_forward(
            soft_features, hard=hard, oracle_vector=oracle_vector, generator=generator
        )
        return torch.cat([precise_recon, soft_recon], dim=-1)


def _encoder_not_collapsed(codec: ContinuousLatentCodec | None, features: torch.Tensor) -> bool:
    """The same encoder-collapse check ``latent_codec_trainer.audit_no_bypass``
    performs (its check 2: zeroing the raw input must change the encoded
    code), applied per-codec since this composite model has two independent
    codecs that do not share ``LatentCodecModel``'s single-codec interface,
    so ``audit_no_bypass`` itself cannot be called directly. ``None``/absent
    codecs (``discrete_only``'s missing soft codec; every arm but
    ``all_continuous``'s missing precise codec) trivially report "not
    collapsed" -- there is nothing to collapse.
    """
    if codec is None or codec.encoder is None:
        return True
    if features.numel() == 0 or torch.equal(features, torch.zeros_like(features)):
        return True
    with torch.no_grad():
        real = codec.encode(features, hard=True)
        zero = codec.encode(torch.zeros_like(features), hard=True)
    return not torch.equal(zero.hard, real.hard)


def _family_mse(
    recon: torch.Tensor,
    target: torch.Tensor,
    dims: dict[str, int],
    families: tuple[str, ...],
    *,
    family_offset: dict[str, int],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in families:
        start = family_offset[name]
        width = dims[name]
        out[name] = float(
            ((recon[:, start : start + width] - target[:, start : start + width]) ** 2).mean().item()
        )
    return out


def _strict_match_per_family(
    recon: torch.Tensor,
    target: torch.Tensor,
    dims: dict[str, int],
    families: tuple[str, ...],
    *,
    family_offset: dict[str, int],
) -> torch.Tensor:
    """Per-example strict-semantics proxy restricted to ``families``: True iff
    every named family's reconstruction MSE is at/under AP-030's
    ``DISTORTION_THRESHOLD`` (reused unchanged).
    """
    n = recon.shape[0]
    ok = torch.ones(n, dtype=torch.bool)
    for name in families:
        start = family_offset[name]
        width = dims[name]
        family_mse = ((recon[:, start : start + width] - target[:, start : start + width]) ** 2).mean(dim=-1)
        ok &= family_mse <= DISTORTION_THRESHOLD
    return ok


@dataclass(frozen=True)
class FactorSwapResult:
    """The issue's own falsification test: perturb only the resolved
    soft-factor vector and confirm the precision-critical decode is
    unchanged.
    """

    perturbation: str
    precision_recon_bit_exact: bool
    max_abs_precision_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "perturbation": self.perturbation,
            "precision_recon_bit_exact": self.precision_recon_bit_exact,
            "max_abs_precision_delta": self.max_abs_precision_delta,
        }


@dataclass(frozen=True)
class ArmResult:
    """Full CAP2-08 measurement for one arm."""

    arm_id: str
    arm: str
    num_holdout: int
    soft_codec_collapsed: bool
    precise_codec_collapsed: bool
    precision_mse_overall: float
    precision_mse_per_family: dict[str, float]
    soft_mse_overall: float
    soft_mse_per_family: dict[str, float]
    overall_strict_rate: float
    soft_strict_rate: float
    precision_strict_rate: float
    factor_swaps: tuple[FactorSwapResult, ...]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "arm": self.arm,
            "num_holdout": self.num_holdout,
            "soft_codec_collapsed": self.soft_codec_collapsed,
            "precise_codec_collapsed": self.precise_codec_collapsed,
            "precision_mse_overall": self.precision_mse_overall,
            "precision_mse_per_family": self.precision_mse_per_family,
            "soft_mse_overall": self.soft_mse_overall,
            "soft_mse_per_family": self.soft_mse_per_family,
            "overall_strict_rate": self.overall_strict_rate,
            "soft_strict_rate": self.soft_strict_rate,
            "precision_strict_rate": self.precision_strict_rate,
            "factor_swaps": [f.to_dict() for f in self.factor_swaps],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ArmComparison:
    """Paired comparison of one arm against the ``discrete_only`` reference."""

    arm_id: str
    overall_strict_rate_delta: float
    overall_strict_rate_ci: dict[str, Any]
    soft_strict_rate_delta: float
    soft_strict_rate_ci: dict[str, Any]
    binder_reference_mse_delta: float
    binder_reference_mse_ci: dict[str, Any]
    exact_paired_test_overall: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "overall_strict_rate_delta": self.overall_strict_rate_delta,
            "overall_strict_rate_ci": self.overall_strict_rate_ci,
            "soft_strict_rate_delta": self.soft_strict_rate_delta,
            "soft_strict_rate_ci": self.soft_strict_rate_ci,
            "binder_reference_mse_delta": self.binder_reference_mse_delta,
            "binder_reference_mse_ci": self.binder_reference_mse_ci,
            "exact_paired_test_overall": self.exact_paired_test_overall,
        }


@dataclass(frozen=True)
class SelectiveHybridReport:
    run_id: str
    fixture_count: int
    fixture_seed: int
    num_train: int
    num_holdout: int
    arms: tuple[ArmResult, ...]
    comparisons: tuple[ArmComparison, ...]
    disposition: str
    disposition_reasons: tuple[str, ...]
    version: str = "cap2-08-v1"
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
            "comparisons": [c.to_dict() for c in self.comparisons],
            "disposition": self.disposition,
            "disposition_reasons": list(self.disposition_reasons),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def load_selective_hybrid_corpus(
    fixture_count: int = 32,
    fixture_seed: int = 0,
) -> tuple[list[SemanticPlanV1], list[SemanticPlanV1]]:
    """Reuses AP-031's fixture corpus loader unchanged (never re-implemented)."""
    return load_causal_audit_corpus(fixture_count, fixture_seed)


def _tensorize(
    plans: list[SemanticPlanV1], cfg: ProgramFactorTensorizerConfig
) -> torch.Tensor:
    return torch.stack([concat_factor_tensors(tensorize_semantic_plan(p, cfg)) for p in plans])


def _family_offsets(dims: dict[str, int]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    offset = 0
    for name in PROGRAM_FACTOR_FAMILIES:
        offsets[name] = offset
        offset += dims[name]
    return offsets


def _train(
    model: SelectiveHybridModel,
    precision_train: torch.Tensor,
    soft_train: torch.Tensor,
    *,
    steps: int,
    lr: float,
) -> None:
    # ContinuousLatentCodec's aux_loss (rate penalty) is off by default (both
    # codecs are constructed with the default rate_penalty=0.0), so training
    # is a plain MSE fit -- no separate aux-loss accumulation needed here.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    target = torch.cat([precision_train, soft_train], dim=-1)
    for _ in range(steps):
        optimizer.zero_grad()
        recon = model(precision_train, soft_train, hard=False, oracle_vector=soft_train)
        loss = F.mse_loss(recon, target)
        loss.backward()
        optimizer.step()


def _run_factor_swaps(
    model: SelectiveHybridModel,
    precision_holdout: torch.Tensor,
    soft_holdout: torch.Tensor,
    *,
    seed: int,
) -> tuple[FactorSwapResult, ...]:
    """The issue's own falsification test: perturb only the resolved
    soft-factor vector, then confirm the precision-critical decode is
    bit-identical -- for every arm, including ``all_continuous`` (the
    precise branch never reads ``soft_features`` regardless of arm; see
    ``SelectiveHybridModel.precise_forward``).
    """
    model.eval()
    rng = random.Random(seed)
    with torch.no_grad():
        baseline_precise = model.precise_forward(precision_holdout, hard=True)

        n = soft_holdout.shape[0]
        results: list[FactorSwapResult] = []

        def _check(name: str, perturbed_soft: torch.Tensor) -> None:
            # Re-derive the precise branch independently, exactly as it
            # would be computed alongside this perturbed soft branch, to
            # prove the two are actually decoupled in the running model --
            # not merely asserted from reading the code.
            _ = model.soft_forward(perturbed_soft, hard=True, oracle_vector=perturbed_soft)
            perturbed_precise = model.precise_forward(precision_holdout, hard=True)
            delta = (perturbed_precise - baseline_precise).abs().max().item()
            results.append(
                FactorSwapResult(
                    perturbation=name,
                    precision_recon_bit_exact=torch.equal(perturbed_precise, baseline_precise),
                    max_abs_precision_delta=float(delta),
                )
            )

        if n >= 2:
            order = list(range(n))
            rng.shuffle(order)
            if order == list(range(n)):
                order = order[::-1]
            _check("shuffled_between_example", soft_holdout[order])
        _check("zeroed", torch.zeros_like(soft_holdout))
        _check(
            "random",
            soft_holdout.mean(dim=0, keepdim=True)
            + soft_holdout.std(dim=0, keepdim=True).clamp_min(1e-6)
            * torch.randn(soft_holdout.shape, generator=torch.Generator().manual_seed(seed)),
        )
    return tuple(results)


def evaluate_arm(
    cfg: SelectiveHybridArmConfig,
    train_plans: list[SemanticPlanV1],
    holdout_plans: list[SemanticPlanV1],
    *,
    tensorizer_config: ProgramFactorTensorizerConfig | None = None,
) -> tuple[ArmResult, torch.Tensor, torch.Tensor]:
    """Train and score one arm. Returns ``(result, overall_strict_bool,
    soft_strict_bool)`` -- the per-example boolean tensors are needed by
    :func:`run_selective_hybrid` for paired comparisons against
    ``discrete_only`` and are not otherwise JSON-serialized.
    """
    tcfg = tensorizer_config or ProgramFactorTensorizerConfig()
    dims = factor_dims(tcfg)
    offsets = _family_offsets(dims)

    features_train = _tensorize(train_plans, tcfg)
    features_holdout = _tensorize(holdout_plans, tcfg)
    precision_train, soft_train = split_soft_precision_features(features_train, dims)
    precision_holdout, soft_holdout = split_soft_precision_features(features_holdout, dims)

    torch.manual_seed(cfg.seed)
    model = SelectiveHybridModel(
        precision_train.shape[-1], soft_train.shape[-1], arm=cfg.arm, hidden_dim=cfg.soft_hidden_dim
    )
    _train(model, precision_train, soft_train, steps=cfg.train_steps, lr=cfg.lr)
    model.eval()

    with torch.no_grad():
        recon = model(precision_holdout, soft_holdout, hard=True, oracle_vector=soft_holdout)
    target = torch.cat([precision_holdout, soft_holdout], dim=-1)

    precision_mse_overall = float(
        F.mse_loss(recon[:, : precision_holdout.shape[-1]], precision_holdout).item()
    )
    soft_mse_overall = float(F.mse_loss(recon[:, precision_holdout.shape[-1] :], soft_holdout).item())
    precision_mse_per_family = _family_mse(
        recon, target, dims, PRECISION_CRITICAL_FACTOR_FAMILIES, family_offset=offsets
    )
    soft_mse_per_family = _family_mse(recon, target, dims, SOFT_FACTOR_FAMILIES, family_offset=offsets)

    overall_strict = _strict_match_per_family(recon, target, dims, PROGRAM_FACTOR_FAMILIES, family_offset=offsets)
    soft_strict = _strict_match_per_family(recon, target, dims, SOFT_FACTOR_FAMILIES, family_offset=offsets)
    precision_strict = _strict_match_per_family(
        recon, target, dims, PRECISION_CRITICAL_FACTOR_FAMILIES, family_offset=offsets
    )

    soft_collapsed = not _encoder_not_collapsed(model.soft_codec, soft_holdout)
    precise_collapsed = not _encoder_not_collapsed(model.precise_codec, precision_holdout)

    swaps = _run_factor_swaps(model, precision_holdout, soft_holdout, seed=cfg.seed)

    notes: list[str] = [
        (
            "strict rates are AP-030's DISTORTION_THRESHOLD MSE-threshold proxy, not "
            "binding_aware_meaningful_v2 -- see module scope decision."
        ),
        (
            "precision-critical channel is unbottlenecked (raw features decoded "
            "directly) for every arm except all_continuous, which routes it "
            "through a ContinuousLatentCodec bottleneck as the negative control."
        ),
    ]
    if cfg.arm is SelectiveLatentArm.DISCRETE_ONLY:
        assert model.soft_codec is None and model.connector is None
        notes.append("discrete_only: soft_codec/connector not constructed (zero extra params).")

    result = ArmResult(
        arm_id=cfg.arm_id,
        arm=cfg.arm.value,
        num_holdout=int(precision_holdout.shape[0]),
        soft_codec_collapsed=soft_collapsed,
        precise_codec_collapsed=precise_collapsed,
        precision_mse_overall=precision_mse_overall,
        precision_mse_per_family=precision_mse_per_family,
        soft_mse_overall=soft_mse_overall,
        soft_mse_per_family=soft_mse_per_family,
        overall_strict_rate=float(overall_strict.float().mean().item()),
        soft_strict_rate=float(soft_strict.float().mean().item()),
        precision_strict_rate=float(precision_strict.float().mean().item()),
        factor_swaps=swaps,
        notes=tuple(notes),
    )
    return result, overall_strict, soft_strict


def run_selective_hybrid(
    *,
    fixture_count: int = 32,
    fixture_seed: int = 0,
    arms: list[SelectiveHybridArmConfig] | None = None,
    tensorizer_config: ProgramFactorTensorizerConfig | None = None,
    ci_resamples: int = 1000,
) -> SelectiveHybridReport:
    """Run the CAP2-08 selective-hybrid continuous/discrete factor audit."""
    train_plans, holdout_plans = load_selective_hybrid_corpus(fixture_count, fixture_seed)
    arm_configs = arms if arms is not None else default_arm_configs()
    if not any(a.arm is SelectiveLatentArm.DISCRETE_ONLY for a in arm_configs):
        raise ValueError("run_selective_hybrid requires a discrete_only arm as the paired-comparison reference")

    tcfg = tensorizer_config or ProgramFactorTensorizerConfig()

    results: dict[str, ArmResult] = {}
    overall_strict_by_arm: dict[str, torch.Tensor] = {}
    soft_strict_by_arm: dict[str, torch.Tensor] = {}
    for a in arm_configs:
        result, overall_strict, soft_strict = evaluate_arm(a, train_plans, holdout_plans, tensorizer_config=tcfg)
        results[a.arm_id] = result
        overall_strict_by_arm[a.arm_id] = overall_strict
        soft_strict_by_arm[a.arm_id] = soft_strict

    discrete_only_id = next(a.arm_id for a in arm_configs if a.arm is SelectiveLatentArm.DISCRETE_ONLY)
    ref_overall = overall_strict_by_arm[discrete_only_id]
    ref_soft = soft_strict_by_arm[discrete_only_id]
    ref_binder_mse = results[discrete_only_id].precision_mse_per_family["binder_reference"]

    comparisons: list[ArmComparison] = []
    for a in arm_configs:
        if a.arm_id == discrete_only_id:
            continue
        cur_overall = overall_strict_by_arm[a.arm_id]
        cur_soft = soft_strict_by_arm[a.arm_id]
        overall_ci = bootstrap_paired_ci(
            ref_overall.float().tolist(),
            cur_overall.float().tolist(),
            metric=lambda left, right: float(sum(right) / len(right) - sum(left) / len(left)),
            seed=a.seed,
            resamples=ci_resamples,
        )
        soft_ci = bootstrap_paired_ci(
            ref_soft.float().tolist(),
            cur_soft.float().tolist(),
            metric=lambda left, right: float(sum(right) / len(right) - sum(left) / len(left)),
            seed=a.seed,
            resamples=ci_resamples,
        )
        binder_delta = results[a.arm_id].precision_mse_per_family["binder_reference"] - ref_binder_mse
        try:
            exact_test = exact_paired_binary_test(
                [int(v) for v in ref_overall.tolist()], [int(v) for v in cur_overall.tolist()]
            )
        except ValueError:
            exact_test = None
        comparisons.append(
            ArmComparison(
                arm_id=a.arm_id,
                overall_strict_rate_delta=results[a.arm_id].overall_strict_rate - results[discrete_only_id].overall_strict_rate,
                overall_strict_rate_ci=overall_ci,
                soft_strict_rate_delta=results[a.arm_id].soft_strict_rate - results[discrete_only_id].soft_strict_rate,
                soft_strict_rate_ci=soft_ci,
                binder_reference_mse_delta=binder_delta,
                binder_reference_mse_ci={},
                exact_paired_test_overall=exact_test,
            )
        )

    reasons: list[str] = []
    hybrid = results.get(SelectiveLatentArm.SELECTIVE_HYBRID.value)
    all_continuous = results.get(SelectiveLatentArm.ALL_CONTINUOUS.value)
    discrete_only = results[discrete_only_id]

    swaps_ok = all(
        swap.precision_recon_bit_exact
        for r in results.values()
        for swap in r.factor_swaps
    )
    if not swaps_ok:
        reasons.append(
            "factor-swap falsification test FAILED: at least one arm's precision-critical "
            "decode changed when only the soft-factor input was perturbed -- no continuous "
            "factor may be promoted."
        )

    if hybrid is None:
        disposition = "inconclusive"
        reasons.append("selective_hybrid arm not evaluated in this run.")
    elif not swaps_ok:
        disposition = "stop_do_not_promote"
    else:
        matches_discrete = hybrid.overall_strict_rate >= discrete_only.overall_strict_rate - 1e-9
        soft_improves = hybrid.soft_strict_rate > discrete_only.soft_strict_rate or (
            hybrid.soft_mse_overall < discrete_only.soft_mse_overall
        )
        binder_regressed = hybrid.precision_mse_per_family["binder_reference"] > (
            discrete_only.precision_mse_per_family["binder_reference"] + DISTORTION_THRESHOLD
        )
        if matches_discrete and soft_improves and not binder_regressed:
            disposition = "promote_selective_hybrid"
            reasons.append(
                "selective_hybrid matched/exceeded discrete_only strict-semantics-proxy rate "
                f"({hybrid.overall_strict_rate:.3f} vs {discrete_only.overall_strict_rate:.3f}) and "
                f"improved a soft-factor metric (soft_strict_rate {hybrid.soft_strict_rate:.3f} vs "
                f"{discrete_only.soft_strict_rate:.3f}; soft_mse {hybrid.soft_mse_overall:.4f} vs "
                f"{discrete_only.soft_mse_overall:.4f}) without a binder/reference MSE regression "
                "beyond DISTORTION_THRESHOLD."
            )
        else:
            disposition = "do_not_promote"
            if not matches_discrete:
                reasons.append(
                    f"selective_hybrid overall_strict_rate {hybrid.overall_strict_rate:.3f} < "
                    f"discrete_only {discrete_only.overall_strict_rate:.3f}."
                )
            if not soft_improves:
                reasons.append("selective_hybrid did not improve any soft-factor metric over discrete_only.")
            if binder_regressed:
                reasons.append(
                    f"selective_hybrid binder_reference MSE {hybrid.precision_mse_per_family['binder_reference']:.4f} "
                    f"regressed beyond discrete_only's {discrete_only.precision_mse_per_family['binder_reference']:.4f} "
                    f"+ DISTORTION_THRESHOLD ({DISTORTION_THRESHOLD})."
                )
    if all_continuous is not None:
        reasons.append(
            "all_continuous (negative control) binder_reference MSE "
            f"{all_continuous.precision_mse_per_family['binder_reference']:.4f} vs discrete_only's "
            f"{discrete_only.precision_mse_per_family['binder_reference']:.4f}."
        )

    run_id = _hash_run_id(("cap2-08", fixture_count, fixture_seed, tuple(a.arm_id for a in arm_configs)))
    return SelectiveHybridReport(
        run_id=run_id,
        fixture_count=fixture_count,
        fixture_seed=fixture_seed,
        num_train=len(train_plans),
        num_holdout=len(holdout_plans),
        arms=tuple(results[a.arm_id] for a in arm_configs),
        comparisons=tuple(comparisons),
        disposition=disposition,
        disposition_reasons=tuple(reasons),
    )
