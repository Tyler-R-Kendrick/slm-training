"""SLM-241 (RSC-A05): named recursive-control architecture arms + resource
accounting.

Linear SLM-241 (RSC-A05) asks for eight architecture control arms (A-H) that
isolate recurrence, weight sharing, explicit z-state capacity, unshared
depth, gradient-through-time, and extra parameters from each other, so a
future matched recursive-depth quality campaign (SLM-233, out of scope here)
can attribute a result to one resource dimension at a time instead of the
historical fixture's confounded 64,994-vs-74,242-parameter comparison (see
``slm_training.models.recursive_denoiser`` module docstring / SLM-240).

This module builds the subset actually implemented so far and reports
resource accounting for it -- never a quality/efficiency claim (``claim_class
== "wiring"`` always). See ``docs/design/iter-rsc-a05-*`` for the exact
formulas, residual matching errors, and an explicit built-vs-deferred split.

Built this iteration (SLM-241):
  - **A** -- stacked baseline (``denoiser_arch="stacked"``).
  - **B** -- shared recursive V1 (``denoiser_arch="shared_recursive"``,
    ``z_state_mode="full"``).
  - **C** -- shared y-only recurrence (``denoiser_arch=
    "shared_recursive_y_only"``, ``z_state_mode="y_only"``).
  - **D** -- recursive no-extra-capacity control (``denoiser_arch=
    "shared_recursive_no_extra_capacity"``, ``z_state_mode="parameter_free"``).
  - **G** -- R=1 shared architecture control: arm B's constructor with
    ``recursive_steps=1``. Interface-compatible, not behaviorally equivalent
    to stacked (SLM-240 already established this framing for R=1 -- reused,
    not re-derived, here).

Explicitly deferred (see the dated design note for why):
  - **E** -- stacked + matched state capacity (a learned target-position
    state + context projection injected once, no recurrence).
  - **F** -- unshared depth-matched tower (an ordinary unshared tower with
    the same total block-evaluation count as the recursive arm).
  - **H** -- stop-gradient recurrence (same forward recurrence, y/z detached
    between steps).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from slm_training.models.blocks import DenoiserTower
from slm_training.models.recursive_denoiser import (
    SharedRecursiveDenoiserTower,
    active_parameter_count,
    checkpoint_state_dict_bytes,
    estimate_transformer_block_flops,
)

RECURSIVE_CONTROL_ARM_REPORT_VERSION = "RecursiveControlArmReportV1"

#: Arm ids from Linear SLM-241 (RSC-A05).
ALL_ARM_IDS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")
BUILT_ARM_IDS: tuple[str, ...] = ("A", "B", "C", "D", "G")
DEFERRED_ARM_IDS: tuple[str, ...] = ("E", "F", "H")

ARM_LABELS: dict[str, str] = {
    "A": "stacked baseline (no z state, unshared blocks)",
    "B": "shared recursive V1 (shared blocks, explicit z state)",
    "C": "shared y-only recurrence (shared blocks, no distinct z state)",
    "D": "recursive no-extra-capacity control (shared blocks, parameter-free z)",
    "E": "stacked + matched state capacity (deferred)",
    "F": "unshared depth-matched tower (deferred)",
    "G": "R=1 shared architecture control (interface-, not behavior-, compatible)",
    "H": "stop-gradient recurrence (deferred)",
}

#: Canonical ``denoiser_arch`` value each built arm maps onto -- the same
#: field ``ModelBuildConfig``/``TwoTowerConfig`` already use, never an ad hoc
#: parallel selector. Arm G reuses B's denoiser_arch; ``recursive_steps=1``
#: is what distinguishes it.
ARM_DENOISER_ARCH: dict[str, str] = {
    "A": "stacked",
    "B": "shared_recursive",
    "C": "shared_recursive_y_only",
    "D": "shared_recursive_no_extra_capacity",
    "G": "shared_recursive",
}

#: ``z_state_mode`` each built (non-stacked) arm maps onto.
ARM_Z_STATE_MODE: dict[str, str] = {
    "B": "full",
    "C": "y_only",
    "D": "parameter_free",
    "G": "full",
}

#: Declared parameter-matching target + tolerance (parameters) for each built
#: arm relative to arm A's stacked baseline, per the issue's requirement to
#: "never call a pair matched without naming the target and residual".
ARM_MATCHING_TARGET: dict[str, str] = {
    "A": "baseline (self); no matching target",
    "B": (
        "none declared -- V1 always adds z_latent+ctx_proj on top of the "
        "shared transition blocks (see recursive_zstate_parameter_delta); "
        "this arm is not intended to parameter-match A"
    ),
    "C": (
        "match A's total parameters exactly when recursive_transition_layers "
        "== A's n_layers (no z-state parameter bank of any kind)"
    ),
    "D": (
        "match A's total parameters exactly when recursive_transition_layers "
        "== A's n_layers (z-state made parameter-free instead of removed)"
    ),
    "G": (
        "none declared -- architecture-change control at R=1, not a "
        "parameter-matching arm (same parameter profile as B)"
    ),
}


def construct_arm_tower(
    arm_id: str,
    *,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    max_len: int,
    dropout: float = 0.0,
    kind_ids: list[int] | None = None,
    n_kinds: int = 0,
    recursive_steps: int = 2,
    recursive_transition_layers: int | None = None,
) -> nn.Module:
    """Construct one control arm's denoiser tower via the same constructors
    ``TwoTowerModel`` uses for ``denoiser_arch``/``z_state_mode`` -- never a
    parallel/ad hoc implementation. Fails closed for deferred (E/F/H) or
    unknown arm ids.
    """
    if arm_id in DEFERRED_ARM_IDS:
        raise NotImplementedError(
            f"control arm {arm_id!r} ({ARM_LABELS[arm_id]}) is explicitly "
            "deferred by SLM-241 (RSC-A05) -- see docs/design/iter-rsc-a05-*; "
            "it is not constructed by this module."
        )
    if arm_id not in ARM_DENOISER_ARCH:
        raise ValueError(f"unknown control arm {arm_id!r}; known arms are {ALL_ARM_IDS!r}")

    transition_layers = (
        recursive_transition_layers if recursive_transition_layers is not None else n_layers
    )
    arch = ARM_DENOISER_ARCH[arm_id]
    if arch == "stacked":
        return DenoiserTower(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len,
            dropout=dropout,
            kind_ids=kind_ids,
            n_kinds=n_kinds,
        )
    steps = 1 if arm_id == "G" else recursive_steps
    return SharedRecursiveDenoiserTower(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_len=max_len,
        dropout=dropout,
        kind_ids=kind_ids,
        n_kinds=n_kinds,
        recursive_steps=steps,
        recursive_transition_layers=transition_layers,
        z_state_mode=ARM_Z_STATE_MODE[arm_id],
    )


@dataclass(frozen=True)
class RecursiveControlArmReportV1:
    """Per-arm resource accounting -- extends SLM-240's
    ``ArchitectureComparisonReportV1`` idea (independent, falsifiable
    dimensions, never a single ``parity``/winner field) to an N-arm table.

    Built only via :func:`build_arm_report`, from a real constructed module
    and one real forward pass -- nothing here is a hard-coded literal.
    """

    contract_version: str
    claim_class: str
    arm_id: str
    label: str
    denoiser_arch: str
    z_state_mode: str | None
    recursive_steps: int
    recursive_transition_layers: int
    d_model: int
    max_len: int
    parameter_count_total: int
    parameter_count_denoiser: int
    active_parameter_count: int
    checkpoint_bytes: int
    undeclared_zstate_parameter_names: tuple[str, ...]
    block_evaluations_per_forward: int
    self_attn_calls_per_forward: int
    cross_attn_calls_per_forward: int
    mlp_calls_per_forward: int
    estimated_forward_flops: float
    matching_target: str
    matching_tolerance_params: int
    parameter_count_delta_vs_baseline: int
    parameter_count_delta_vs_baseline_pct: float
    residual_matching_error_params: int
    within_matching_tolerance: bool
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.contract_version != RECURSIVE_CONTROL_ARM_REPORT_VERSION:
            raise ValueError(
                f"contract_version={self.contract_version!r} does not match "
                f"{RECURSIVE_CONTROL_ARM_REPORT_VERSION!r}."
            )
        if self.claim_class != "wiring":
            raise ValueError(
                "RecursiveControlArmReportV1 is wiring-only evidence; "
                f"claim_class={self.claim_class!r} is not a defined value "
                "(no quality/perf claim is ever encoded here)."
            )
        if "parity" in self.as_dict() or "winner" in self.as_dict():
            raise ValueError(
                "RecursiveControlArmReportV1 must never carry a 'parity' or "
                "'winner' field -- each arm is reported independently."
            )
        expected_within = (
            abs(self.residual_matching_error_params) <= self.matching_tolerance_params
        )
        if expected_within != self.within_matching_tolerance:
            raise ValueError(
                "within_matching_tolerance "
                f"({self.within_matching_tolerance}) does not reflect "
                f"abs(residual_matching_error_params={self.residual_matching_error_params}) "
                f"<= matching_tolerance_params={self.matching_tolerance_params}."
            )
        if self.arm_id in ("C", "D") and self.undeclared_zstate_parameter_names:
            raise ValueError(
                f"arm {self.arm_id!r} declares z-state parameters "
                f"{self.undeclared_zstate_parameter_names!r} -- arms C/D must "
                "have no z-shaped parameter bank at all."
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_arm_report(
    arm_id: str,
    tower: nn.Module,
    *,
    baseline_tower: nn.Module,
    noisy_ids: torch.Tensor,
    context: torch.Tensor,
    pad_id: int,
    matching_tolerance_params: int = 0,
) -> RecursiveControlArmReportV1:
    """Build a real, measured :class:`RecursiveControlArmReportV1` for one
    already-constructed arm tower, relative to ``baseline_tower`` (arm A)."""
    named = dict(tower.named_parameters())
    total = int(sum(p.numel() for p in named.values()))
    denoiser_layers = list(getattr(tower, "layers", []))
    denoiser_params = int(sum(p.numel() for layer in denoiser_layers for p in layer.parameters()))
    baseline_total = int(sum(p.numel() for p in baseline_tower.parameters()))

    undeclared_zstate = tuple(
        sorted(name for name in named if name.split(".")[0] in {"z_latent", "ctx_proj"})
    )

    checkpoint_bytes = checkpoint_state_dict_bytes(tower)

    fresh_out = tower(noisy_ids, context, pad_id)
    active = active_parameter_count(tower, fresh_out)

    recursive_steps = int(getattr(tower, "recursive_steps", 1))
    transition_layers = int(
        getattr(tower, "recursive_transition_layers", len(denoiser_layers)) or len(denoiser_layers)
    )
    is_recursive = hasattr(tower, "recursive_steps")
    block_evals = recursive_steps * transition_layers if is_recursive else len(denoiser_layers)

    seq_len = int(noisy_ids.shape[1])
    ctx_len = int(context.shape[1])
    d_model = int(getattr(tower, "d_model", context.shape[-1]))
    per_block_flops = estimate_transformer_block_flops(
        seq_len=seq_len, ctx_len=ctx_len, d_model=d_model
    )
    flops = per_block_flops * block_evals

    delta = total - baseline_total
    delta_pct = (delta / baseline_total * 100.0) if baseline_total else float("nan")
    residual = delta
    within_tol = abs(residual) <= matching_tolerance_params

    notes: list[str] = []
    if arm_id in ("C", "D"):
        notes.append(
            "no z-state parameter bank declared"
            if not undeclared_zstate
            else "UNEXPECTED: z-state parameter names present"
        )
    if arm_id == "G":
        notes.append(
            "R=1 architecture-change control -- interface-compatible with "
            "the stacked baseline, not behaviorally equivalent (SLM-240)."
        )

    z_state_mode = getattr(tower, "z_state_mode", None)

    return RecursiveControlArmReportV1(
        contract_version=RECURSIVE_CONTROL_ARM_REPORT_VERSION,
        claim_class="wiring",
        arm_id=arm_id,
        label=ARM_LABELS[arm_id],
        denoiser_arch=ARM_DENOISER_ARCH[arm_id],
        z_state_mode=z_state_mode,
        recursive_steps=recursive_steps if is_recursive else 0,
        recursive_transition_layers=transition_layers,
        d_model=d_model,
        max_len=int(getattr(tower, "max_len", 0)),
        parameter_count_total=total,
        parameter_count_denoiser=denoiser_params,
        active_parameter_count=active,
        checkpoint_bytes=checkpoint_bytes,
        undeclared_zstate_parameter_names=undeclared_zstate,
        block_evaluations_per_forward=block_evals,
        self_attn_calls_per_forward=block_evals,
        cross_attn_calls_per_forward=block_evals,
        mlp_calls_per_forward=block_evals,
        estimated_forward_flops=flops,
        matching_target=ARM_MATCHING_TARGET[arm_id],
        matching_tolerance_params=matching_tolerance_params,
        parameter_count_delta_vs_baseline=delta,
        parameter_count_delta_vs_baseline_pct=delta_pct,
        residual_matching_error_params=residual,
        within_matching_tolerance=within_tol,
        notes=tuple(notes),
    )


def build_control_arm_table(
    arm_ids: tuple[str, ...],
    *,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    max_len: int,
    recursive_steps: int = 2,
    recursive_transition_layers: int | None = None,
    noisy_ids: torch.Tensor,
    context: torch.Tensor,
    pad_id: int,
    matching_tolerance_params: int = 0,
) -> list[RecursiveControlArmReportV1]:
    """Build a real resource-accounting table across ``arm_ids`` -- requirement
    #11 ("the fixture emits a complete comparison table for every arm you
    built"). Never includes a raw loss or a winner; every field is a
    structural/resource measurement of one concrete constructed tower."""
    if "A" not in arm_ids:
        raise ValueError(
            "build_control_arm_table requires arm A (stacked baseline) so "
            "every other arm's parameter delta has a declared reference"
        )
    baseline = construct_arm_tower(
        "A",
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_len=max_len,
        recursive_steps=recursive_steps,
        recursive_transition_layers=recursive_transition_layers,
    )
    reports = []
    for arm_id in arm_ids:
        tower = (
            baseline
            if arm_id == "A"
            else construct_arm_tower(
                arm_id,
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                max_len=max_len,
                recursive_steps=recursive_steps,
                recursive_transition_layers=recursive_transition_layers,
            )
        )
        reports.append(
            build_arm_report(
                arm_id,
                tower,
                baseline_tower=baseline,
                noisy_ids=noisy_ids,
                context=context,
                pad_id=pad_id,
                matching_tolerance_params=matching_tolerance_params,
            )
        )
    return reports
