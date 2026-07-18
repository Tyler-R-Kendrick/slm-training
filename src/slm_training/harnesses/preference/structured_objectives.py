"""OpenUI-native structured local-preference objectives (LDI3-01 / SLM-128).

Three architecture-neutral objectives, each a pure function of exact decision logits +
materialized action sets, so a causal or TwoTower trainer calls the *same* implementation
(the trainer extracts the logits row; this module owns the math — nothing is duplicated in
architecture-specific files):

* **Legal-Set FTPO** (Objective A) — two variants over multiple independently verified good
  and bad legal actions: an explicitly pair-weighted `G × B` margin, and a legal
  probability-mass margin in the grammar-legal simplex.
* **TAB-PO-inspired barrier** (Objective B) — an additive, separately-metered SFT anchor
  `-log p(g)` for under-confident, verifier-critical good actions; zero for confident ones;
  structural punctuation gets default-low weight unless marked critical.
* **TBPO-inspired ratio control** (Objective C) — a bounded good-vs-bad log-ratio control
  against a reference at the *same* exact state, with an advantage-centered or a small
  serializable learned-scalar state baseline.

These are **Adapted**, not reproductions of the cited papers. No model update, no matrix
run, no hidden eval gold: this module computes losses/metrics and a no-update fixture
report only. Existing `local_train` objectives (`ce_margin`/`unlikelihood`/`ftpo_*`) are
untouched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

from slm_training.lineage.records import content_sha

STRUCTURED_OBJECTIVE_SCHEMA_VERSION = 1

StructuredObjectiveName = Literal[
    "legal_set_ftpo",
    "tab_po_inspired_barrier",
    "tbpo_inspired_ratio",
]
LegalSetVariant = Literal["pairwise_margin", "mass_margin"]
BaselineType = Literal["none", "advantage", "learned_scalar"]

__all__ = [
    "STRUCTURED_OBJECTIVE_SCHEMA_VERSION",
    "StateBaseline",
    "StructuredObjectiveConfig",
    "StructuredObjectiveError",
    "StructuredObjectiveInput",
    "legal_probability_masses",
    "structured_objective_batch_loss",
    "structured_objective_loss",
    "structured_objective_report",
    "token_erosion_rate",
]


class StructuredObjectiveError(ValueError):
    """Raised when a structured objective input or config is invalid."""


@dataclass(frozen=True)
class StructuredObjectiveConfig:
    """Typed, versioned config for a structured objective; part of run/adapter identity.

    ``from_dict`` fails closed on unknown fields and ``fingerprint`` is a deterministic
    content hash, so a config change is always visible in lineage.
    """

    name: StructuredObjectiveName
    variant: LegalSetVariant = "mass_margin"
    epsilon: float = 2.0
    tau: float = 1.0
    temperature: float = 1.0
    margin: float = 0.0
    state_normalized: bool = True
    evidence_clip: float = 10.0
    barrier_p: float = 0.1
    barrier_strength: float = 1.0
    default_role_weight: float = 1.0
    structural_role_weight: float = 0.0
    role_weights: tuple[tuple[str, float], ...] = ()
    structural_roles: tuple[str, ...] = ("punctuation", "structural")
    non_target_tether: float = 0.0
    target_tether: float = 0.0
    target_grace: float = 1.0
    baseline_type: BaselineType = "none"
    numerical_eps: float = 1e-8
    schema_version: int = STRUCTURED_OBJECTIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STRUCTURED_OBJECTIVE_SCHEMA_VERSION:
            raise StructuredObjectiveError(
                f"unsupported structured objective schema version {self.schema_version}"
            )
        if self.name not in (
            "legal_set_ftpo",
            "tab_po_inspired_barrier",
            "tbpo_inspired_ratio",
        ):
            raise StructuredObjectiveError(f"unknown structured objective {self.name!r}")
        if self.name == "legal_set_ftpo" and self.variant not in (
            "pairwise_margin",
            "mass_margin",
        ):
            raise StructuredObjectiveError(f"unknown legal-set variant {self.variant!r}")
        for name in ("epsilon", "tau", "temperature"):
            if float(getattr(self, name)) <= 0.0:
                raise StructuredObjectiveError(f"{name} must be positive")
        if not 0.0 <= self.barrier_p <= 1.0:
            raise StructuredObjectiveError("barrier_p must be in [0, 1]")
        for name in (
            "barrier_strength",
            "evidence_clip",
            "non_target_tether",
            "target_tether",
            "target_grace",
            "numerical_eps",
        ):
            if float(getattr(self, name)) < 0.0:
                raise StructuredObjectiveError(f"{name} must be non-negative")
        if self.baseline_type not in ("none", "advantage", "learned_scalar"):
            raise StructuredObjectiveError(f"unknown baseline_type {self.baseline_type!r}")

    def role_weight(self, role: str | None) -> float:
        """Semantic weight for an action ``role`` (structural roles default low)."""
        table = dict(self.role_weights)
        if role in table:
            return float(table[role])
        if role in self.structural_roles:
            return float(self.structural_role_weight)
        return float(self.default_role_weight)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON view (lists, not tuples) round-trippable via ``from_dict``."""
        return {
            "name": self.name,
            "variant": self.variant,
            "epsilon": self.epsilon,
            "tau": self.tau,
            "temperature": self.temperature,
            "margin": self.margin,
            "state_normalized": self.state_normalized,
            "evidence_clip": self.evidence_clip,
            "barrier_p": self.barrier_p,
            "barrier_strength": self.barrier_strength,
            "default_role_weight": self.default_role_weight,
            "structural_role_weight": self.structural_role_weight,
            "role_weights": [list(pair) for pair in self.role_weights],
            "structural_roles": list(self.structural_roles),
            "non_target_tether": self.non_target_tether,
            "target_tether": self.target_tether,
            "target_grace": self.target_grace,
            "baseline_type": self.baseline_type,
            "numerical_eps": self.numerical_eps,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredObjectiveConfig:
        """Rebuild from ``to_dict`` output, rejecting unknown fields (fail closed)."""
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise StructuredObjectiveError(
                f"unknown structured objective config fields: {sorted(unknown)}"
            )
        payload = dict(data)
        if "role_weights" in payload and payload["role_weights"] is not None:
            payload["role_weights"] = tuple(
                (str(role), float(weight)) for role, weight in payload["role_weights"]
            )
        if "structural_roles" in payload and payload["structural_roles"] is not None:
            payload["structural_roles"] = tuple(payload["structural_roles"])
        return cls(**payload)

    def fingerprint(self) -> str:
        """Deterministic content hash of the config (part of run/adapter identity)."""
        return content_sha(self.to_dict())


@dataclass(frozen=True)
class StructuredObjectiveInput:
    """One exact decision: logits plus its materialized, verifier-legal action sets.

    Architecture-neutral — the caller (causal or TwoTower trainer, or a test with mock
    logits) supplies the 1-D logit row and the action sets. Good/bad must be non-empty and
    every referenced id must be legal; the good/bad/ambiguous/unobserved sets must be
    disjoint.
    """

    logits: torch.Tensor
    legal_ids: tuple[int, ...]
    good_ids: tuple[int, ...]
    bad_ids: tuple[int, ...]
    ambiguous_ids: tuple[int, ...] = ()
    unobserved_ids: tuple[int, ...] = ()
    good_weights: tuple[float, ...] | None = None
    bad_weights: tuple[float, ...] | None = None
    good_roles: tuple[str, ...] | None = None
    good_critical: tuple[bool, ...] | None = None
    evidence_confidence: float = 1.0
    reference_logits: torch.Tensor | None = None
    state_id: str = ""

    def __post_init__(self) -> None:
        if self.logits.ndim != 1:
            raise StructuredObjectiveError("decision logits must be one-dimensional")
        if not self.good_ids or not self.bad_ids:
            raise StructuredObjectiveError("good and bad action sets must be non-empty")
        legal = set(self.legal_ids)
        if not legal:
            raise StructuredObjectiveError("legal_ids must be non-empty")
        partitions = {
            "good": self.good_ids,
            "bad": self.bad_ids,
            "ambiguous": self.ambiguous_ids,
            "unobserved": self.unobserved_ids,
        }
        for label, ids in partitions.items():
            if not legal.issuperset(ids):
                raise StructuredObjectiveError(
                    f"{label} action ids must be a subset of the legal set"
                )
        pooled = [tok for ids in partitions.values() for tok in ids]
        if len(pooled) != len(set(pooled)):
            raise StructuredObjectiveError(
                "good/bad/ambiguous/unobserved action sets must be disjoint"
            )
        vocab = int(self.logits.shape[0])
        if any(not 0 <= tok < vocab for tok in legal):
            raise StructuredObjectiveError("legal ids must index the logits")
        for name, ids in (("good_weights", self.good_ids), ("bad_weights", self.bad_ids)):
            weights = getattr(self, name)
            if weights is not None and len(weights) != len(ids):
                raise StructuredObjectiveError(f"{name} must align with its action set")
        for name in ("good_roles", "good_critical"):
            values = getattr(self, name)
            if values is not None and len(values) != len(self.good_ids):
                raise StructuredObjectiveError(f"{name} must align with good_ids")
        if self.reference_logits is not None and (
            self.reference_logits.shape != self.logits.shape
        ):
            raise StructuredObjectiveError("reference logits must match the logits shape")


def _select(logits: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    index = torch.tensor(tuple(ids), dtype=torch.long, device=logits.device)
    return logits.index_select(0, index)


def legal_probability_masses(
    inp: StructuredObjectiveInput, *, temperature: float = 1.0
) -> dict[str, torch.Tensor]:
    """Softmax over **only** the legal set, summed per partition (differentiable).

    Returns ``good``/``bad``/``ambiguous``/``unobserved`` mass tensors plus the full legal
    probability vector and its entropy — mass never leaks to out-of-legal tokens.
    """
    legal_logits = _select(inp.logits, inp.legal_ids) / temperature
    legal_probs = F.softmax(legal_logits, dim=-1)
    position = {tok: i for i, tok in enumerate(inp.legal_ids)}

    def _mass(ids: Sequence[int]) -> torch.Tensor:
        if not ids:
            return inp.logits.new_zeros(())
        idx = torch.tensor([position[t] for t in ids], dtype=torch.long, device=inp.logits.device)
        return legal_probs.index_select(0, idx).sum()

    entropy = -(legal_probs.clamp_min(1e-12) * legal_probs.clamp_min(1e-12).log()).sum()
    return {
        "legal_probs": legal_probs,
        "good": _mass(inp.good_ids),
        "bad": _mass(inp.bad_ids),
        "ambiguous": _mass(inp.ambiguous_ids),
        "unobserved": _mass(inp.unobserved_ids),
        "entropy": entropy,
    }


def _evidence_weights(
    weights: tuple[float, ...] | None, count: int, clip: float, device: torch.device
) -> torch.Tensor:
    if weights is None:
        return torch.ones(count, dtype=torch.float32, device=device)
    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    if clip > 0:
        tensor = tensor.clamp(0.0, clip)
    return tensor


def _legal_set_ftpo_pairwise(
    inp: StructuredObjectiveInput, cfg: StructuredObjectiveConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    """Explicitly pair-weighted ``G × B`` set margin, normalized per state."""
    good = _select(inp.logits, inp.good_ids)
    bad = _select(inp.logits, inp.bad_ids)
    deltas = good[:, None] - bad[None, :]
    good_w = _evidence_weights(inp.good_weights, len(inp.good_ids), cfg.evidence_clip, inp.logits.device)
    bad_w = _evidence_weights(inp.bad_weights, len(inp.bad_ids), cfg.evidence_clip, inp.logits.device)
    pair_evidence = good_w[:, None] * bad_w[None, :]
    margin_weight = ((cfg.epsilon - deltas) / cfg.epsilon).clamp(0.0, 1.0)
    weight = pair_evidence * margin_weight
    penalty = F.softplus((cfg.epsilon - deltas) / cfg.tau)
    total_weight = weight.sum()
    if float(total_weight) <= cfg.numerical_eps:
        # All pairs already satisfy the margin: a zero-weight, differentiable zero loss.
        loss = (weight * penalty).sum()
    else:
        loss = (weight * penalty).sum() / total_weight
    loss = loss * inp.evidence_confidence
    metrics = {
        "preference_loss": float(loss.detach()),
        "mean_margin": float(deltas.mean().detach()),
        "chosen_win": float((deltas > 0).float().mean().detach()),
        "active_pair_weight": float(weight.mean().detach()),
        "pair_count": float(deltas.numel()),
    }
    return loss, metrics


def _legal_set_ftpo_mass_margin(
    inp: StructuredObjectiveInput, cfg: StructuredObjectiveConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    """Legal probability-mass margin ``softplus(margin + log P_B - log P_G)``."""
    masses = legal_probability_masses(inp, temperature=cfg.temperature)
    eps = cfg.numerical_eps
    good_mass = masses["good"]
    bad_mass = masses["bad"]
    loss = F.softplus(
        cfg.margin + torch.log(bad_mass + eps) - torch.log(good_mass + eps)
    ) * inp.evidence_confidence
    metrics = {
        "preference_loss": float(loss.detach()),
        "good_mass": float(good_mass.detach()),
        "bad_mass": float(bad_mass.detach()),
        "ambiguous_mass": float(masses["ambiguous"].detach()),
        "unobserved_mass": float(masses["unobserved"].detach()),
        "legal_entropy": float(masses["entropy"].detach()),
    }
    return loss, metrics


def _tab_po_barrier(
    inp: StructuredObjectiveInput, cfg: StructuredObjectiveConfig
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Additive SFT anchor for under-confident, verifier-critical good actions.

    ``-log p_legal(g)`` weighted by evidence × semantic role, applied only where a good
    action is flagged critical and its legal-space probability is below ``barrier_p``.
    Zero for confident good actions; structural roles carry ``structural_role_weight``.
    """
    masses = legal_probability_masses(inp, temperature=cfg.temperature)
    legal_probs = masses["legal_probs"]
    position = {tok: i for i, tok in enumerate(inp.legal_ids)}
    good_w = _evidence_weights(inp.good_weights, len(inp.good_ids), cfg.evidence_clip, inp.logits.device)
    total = inp.logits.new_zeros(())
    active = 0
    anchored_probs: dict[int, float] = {}
    loss_by_role: dict[str, float] = {}
    for slot, token in enumerate(inp.good_ids):
        critical = True if inp.good_critical is None else bool(inp.good_critical[slot])
        role = None if inp.good_roles is None else inp.good_roles[slot]
        role_weight = cfg.role_weight(role)
        prob = legal_probs[position[token]]
        if not critical or role_weight <= 0.0 or float(prob) >= cfg.barrier_p:
            continue
        anchor = -torch.log(prob.clamp_min(cfg.numerical_eps))
        contribution = cfg.barrier_strength * float(good_w[slot]) * role_weight * anchor
        total = total + contribution
        active += 1
        anchored_probs[int(token)] = float(prob.detach())
        loss_by_role[str(role)] = loss_by_role.get(str(role), 0.0) + float(contribution.detach())
    if active:
        total = total / active
    metrics = {
        "barrier_loss": float(total.detach()),
        "barrier_active_fraction": active / max(1, len(inp.good_ids)),
        "barrier_active_count": active,
        "anchored_probabilities": anchored_probs,
        "loss_by_role": loss_by_role,
        "legal_entropy": float(masses["entropy"].detach()),
    }
    return total, metrics


def _tbpo_ratio_control(
    inp: StructuredObjectiveInput,
    cfg: StructuredObjectiveConfig,
    *,
    baseline: StateBaseline | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bounded good-vs-bad log-ratio control against a same-state reference.

    Log-ratios ``z[a] - reference_z[a]`` are advantage-centered (by the mean legal
    log-ratio) or by a serializable learned scalar baseline; the control is disabled and
    reported when there is no reference (inadequate state support).
    """
    if inp.reference_logits is None:
        return inp.logits.new_zeros(()), {
            "ratio_loss": 0.0,
            "control_active": False,
            "reason": "no reference logits; state support inadequate for ratio control",
        }
    diff = inp.logits - inp.reference_logits.detach()
    legal_ratio = _select(diff, inp.legal_ids)
    if cfg.baseline_type == "learned_scalar" and baseline is not None:
        center = baseline.value(inp.logits.device)
    elif cfg.baseline_type == "none":
        center = inp.logits.new_zeros(())
    else:  # advantage: center by the mean legal log-ratio at this exact state
        center = legal_ratio.mean().detach()
    good_ratio = _select(diff, inp.good_ids) - center
    bad_ratio = _select(diff, inp.bad_ids) - center
    loss = F.softplus(cfg.margin + bad_ratio.mean() - good_ratio.mean()) * inp.evidence_confidence
    metrics = {
        "ratio_loss": float(loss.detach()),
        "control_active": True,
        "good_log_ratio": float(good_ratio.mean().detach()),
        "bad_log_ratio": float(bad_ratio.mean().detach()),
        "baseline": float(center) if center.ndim == 0 else None,
        "baseline_type": cfg.baseline_type,
    }
    return loss, metrics


def _tether(
    inp: StructuredObjectiveInput, cfg: StructuredObjectiveConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    """Target/non-target MSE locality tethers against the reference (separately metered)."""
    if (cfg.non_target_tether <= 0 and cfg.target_tether <= 0) or inp.reference_logits is None:
        zero = inp.logits.new_zeros(())
        return zero, {"non_target_logit_mse": 0.0, "target_excess_logit_mse": 0.0}
    diff = inp.logits - inp.reference_logits.detach()
    target_mask = torch.zeros_like(inp.logits, dtype=torch.bool)
    target_index = torch.tensor(
        tuple(set(inp.good_ids) | set(inp.bad_ids)), dtype=torch.long, device=inp.logits.device
    )
    target_mask[target_index] = True
    non_target_mse = inp.logits.new_zeros(())
    target_excess_mse = inp.logits.new_zeros(())
    if cfg.non_target_tether > 0 and (~target_mask).any():
        non_target_mse = diff[~target_mask].pow(2).mean()
    if cfg.target_tether > 0:
        excess = (diff[target_mask].abs() - cfg.target_grace).clamp(min=0.0)
        target_excess_mse = excess.pow(2).mean()
    loss = cfg.non_target_tether * non_target_mse + cfg.target_tether * target_excess_mse
    return loss, {
        "non_target_logit_mse": float(non_target_mse.detach()),
        "target_excess_logit_mse": float(target_excess_mse.detach()),
    }


def structured_objective_loss(
    inp: StructuredObjectiveInput,
    cfg: StructuredObjectiveConfig,
    *,
    baseline: StateBaseline | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """One decision's structured-objective loss + component metrics.

    Composes the preference term (per ``cfg.name``/``variant``), the optional additive
    TAB barrier, and the locality tethers — each separately metered so barrier and target
    tether never double-count.
    """
    if cfg.name == "legal_set_ftpo":
        if cfg.variant == "pairwise_margin":
            preference, metrics = _legal_set_ftpo_pairwise(inp, cfg)
        else:
            preference, metrics = _legal_set_ftpo_mass_margin(inp, cfg)
        barrier, barrier_metrics = inp.logits.new_zeros(()), {"barrier_loss": 0.0}
    elif cfg.name == "tab_po_inspired_barrier":
        preference, metrics = _legal_set_ftpo_mass_margin(inp, cfg)
        barrier, barrier_metrics = _tab_po_barrier(inp, cfg)
    else:  # tbpo_inspired_ratio
        preference, metrics = _tbpo_ratio_control(inp, cfg, baseline=baseline)
        barrier, barrier_metrics = inp.logits.new_zeros(()), {"barrier_loss": 0.0}
    tether, tether_metrics = _tether(inp, cfg)
    loss = preference + barrier + tether
    report = {
        "objective": cfg.name,
        "objective_only_loss": float(preference.detach()),
        "loss": float(loss.detach()),
        **metrics,
        **barrier_metrics,
        **tether_metrics,
    }
    return loss, report


def structured_objective_batch_loss(
    inputs: Sequence[StructuredObjectiveInput],
    cfg: StructuredObjectiveConfig,
    *,
    baseline: StateBaseline | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Mean of per-state losses (state-normalized) so large action sets cannot dominate.

    With ``cfg.state_normalized`` the batch loss is the arithmetic mean of the per-state
    losses regardless of each state's ``G × B`` size; without it, states are weighted by
    their pair count (the un-normalized behavior, kept for ablation).
    """
    if not inputs:
        raise StructuredObjectiveError("structured objective batch requires >= 1 state")
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    per_state: list[dict[str, Any]] = []
    for inp in inputs:
        loss, report = structured_objective_loss(inp, cfg, baseline=baseline)
        losses.append(loss)
        weights.append(float(len(inp.good_ids) * len(inp.bad_ids)))
        per_state.append(report)
    stacked = torch.stack(losses)
    if cfg.state_normalized:
        batch_loss = stacked.mean()
    else:
        weight = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
        batch_loss = (stacked * weight).sum() / weight.sum().clamp_min(cfg.numerical_eps)
    aggregate = {
        "states": len(inputs),
        "state_normalized": cfg.state_normalized,
        "batch_loss": float(batch_loss.detach()),
        "mean_state_loss": float(stacked.mean().detach()),
        "config_fingerprint": cfg.fingerprint(),
    }
    return batch_loss, {"aggregate": aggregate, "per_state": per_state}


class StateBaseline:
    """A tiny serializable scalar state baseline for the ratio control (Objective C).

    Fit strictly from **train** states as the mean legal log-ratio, then frozen and
    serialized as part of the artifact. It is not a model update; it centers advantages.
    """

    def __init__(self, value: float = 0.0, *, fitted: bool = False) -> None:
        """Hold a scalar baseline value and whether it was fit from train states."""
        self._value = float(value)
        self._fitted = bool(fitted)

    @classmethod
    def fit(
        cls, train_inputs: Sequence[StructuredObjectiveInput]
    ) -> StateBaseline:
        """Fit the scalar as the mean legal log-ratio over train states with a reference."""
        ratios: list[float] = []
        for inp in train_inputs:
            if inp.reference_logits is None:
                continue
            diff = inp.logits.detach() - inp.reference_logits.detach()
            ratios.append(float(_select(diff, inp.legal_ids).mean()))
        if not ratios:
            return cls(0.0, fitted=False)
        return cls(sum(ratios) / len(ratios), fitted=True)

    def value(self, device: torch.device | None = None) -> torch.Tensor:
        """The baseline as a detached scalar tensor (never carries gradient)."""
        return torch.tensor(self._value, dtype=torch.float32, device=device)

    @property
    def fitted(self) -> bool:
        """Whether this baseline was fit from train states (vs the default zero)."""
        return self._fitted

    def to_dict(self) -> dict[str, Any]:
        """Serialize the baseline scalar and its fitted flag."""
        return {"value": self._value, "fitted": self._fitted, "kind": "state_scalar_baseline"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateBaseline:
        """Rebuild a baseline from :meth:`to_dict` output."""
        return cls(float(data["value"]), fitted=bool(data.get("fitted", False)))


def token_erosion_rate(
    before: dict[int, float], after: dict[int, float], *, preference_improved: bool
) -> dict[str, Any]:
    """Fraction of verified good actions whose absolute probability dropped.

    Erosion is the TAB-PO failure mode: relative preference improves while the good
    action's own likelihood falls. Reports the eroded fraction and the eroded tokens; the
    ``preference_improved`` flag records whether relative preference actually rose.
    """
    shared = sorted(set(before) & set(after))
    if not shared:
        return {"eroded_fraction": 0.0, "eroded_tokens": [], "preference_improved": preference_improved}
    eroded = [tok for tok in shared if after[tok] < before[tok]]
    return {
        "eroded_fraction": len(eroded) / len(shared),
        "eroded_tokens": eroded,
        "preference_improved": bool(preference_improved),
        "considered": len(shared),
    }


def structured_objective_report(
    inputs: Sequence[StructuredObjectiveInput],
    cfg: StructuredObjectiveConfig,
    *,
    baseline: StateBaseline | None = None,
) -> dict[str, Any]:
    """No-model-update objective report over a frozen fixture corpus.

    Records objective/component values, barrier + legal/full-vocab masses, and the config
    fingerprint — the LDI3-01 evidence artifact. Computes nothing that would update a
    model and makes no quality claim.
    """
    _batch_loss, batch = structured_objective_batch_loss(inputs, cfg, baseline=baseline)
    full_vocab_good = []
    for inp in inputs:
        probs = F.softmax(inp.logits.detach(), dim=-1)
        full_vocab_good.append(float(_select(probs, inp.good_ids).sum()))
    return {
        "kind": "structured_objective_report",
        "adaptation": "Adapted (not reproduced/SOTA): Legal-Set FTPO, TAB-PO-inspired "
        "barrier, TBPO-inspired ratio control",
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint(),
        "baseline": None if baseline is None else baseline.to_dict(),
        "aggregate": batch["aggregate"],
        "per_state": batch["per_state"],
        "full_vocab_good_mass": full_vocab_good,
    }
