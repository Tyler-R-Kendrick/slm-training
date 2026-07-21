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

Built (SLM-241, A/B/C/D/G; F and E added in follow-up SLM-241 iterations):
  - **A** -- stacked baseline (``denoiser_arch="stacked"``).
  - **B** -- shared recursive V1 (``denoiser_arch="shared_recursive"``,
    ``z_state_mode="full"``).
  - **C** -- shared y-only recurrence (``denoiser_arch=
    "shared_recursive_y_only"``, ``z_state_mode="y_only"``).
  - **D** -- recursive no-extra-capacity control (``denoiser_arch=
    "shared_recursive_no_extra_capacity"``, ``z_state_mode="parameter_free"``).
  - **E** -- stacked + matched state capacity (``denoiser_arch=
    "stacked_matched_state"``): the mirror image of D. A plain, unshared,
    non-recursive tower (:class:`~slm_training.models.recursive_denoiser.
    StackedMatchedStateDenoiserTower`, ``n_layers`` blocks, each called
    exactly once, same as arm A) plus a learned ``state``/``state_ctx_proj``
    pair shape-matched to B's ``z_latent``/``ctx_proj`` and injected once
    before any transition block runs (never recurrently re-applied). Its
    total parameter count equals a same-``n_layers`` arm A plus
    ``recursive_zstate_parameter_delta(d_model, max_len)`` exactly.
  - **F** -- unshared depth-matched tower (``denoiser_arch=
    "stacked_depth_matched"``): a plain, unshared ``DenoiserTower`` -- the
    exact same class as arm A, no new tower code -- built with
    ``recursive_steps * recursive_transition_layers`` independent transition
    blocks instead of ``n_layers``, so its per-forward block-evaluation count
    matches arm B's exactly. This necessarily costs MORE parameters than B
    (nothing is shared); :func:`build_arm_f_dual_view` reports both the
    block-evaluation-matched view (the primary construction
    ``construct_arm_tower("F", ...)`` returns) and a separate
    parameter-nearest view, with the honest residual on whichever dimension
    isn't exact -- never a bare "matched" claim.
  - **G** -- R=1 shared architecture control: arm B's constructor with
    ``recursive_steps=1``. Interface-compatible, not behaviorally equivalent
    to stacked (SLM-240 already established this framing for R=1 -- reused,
    not re-derived, here).
  - **H** -- stop-gradient recurrence (SLM-241/RSC-A05 second follow-up):
    arm B's exact constructor (``denoiser_arch="shared_recursive"`` -- reused,
    not a new arch string, the same convention arm G already established) with
    ``SharedRecursiveDenoiserTower(detach_between_steps=True)``. Identical
    forward recurrence to B (same shared blocks, same ``y``/``z`` update
    equations, same shapes, byte-identical forward values for identical
    weights/inputs) -- only the backward graph differs: ``.detach()`` is
    called on the carried-forward ``y``/``z`` between recursion steps, so a
    later step's loss cannot backpropagate through the recurrent chain into
    an earlier step's block application (though it still reaches the shared
    weights through that later step's own application of them). No new
    parameter, no new block-evaluation -- same params/block-evals as B
    exactly. See ``docs/design/iter-rsc-a05-*`` for the forward-identity and
    gradient-divergence evidence.

All eight arms (A-H) are built as of this iteration -- none remain deferred.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from slm_training.models.blocks import DenoiserTower
from slm_training.models.recursive_denoiser import (
    SharedRecursiveDenoiserTower,
    StackedMatchedStateDenoiserTower,
    active_parameter_count,
    checkpoint_state_dict_bytes,
    estimate_transformer_block_flops,
    recursive_zstate_parameter_delta,
)

RECURSIVE_CONTROL_ARM_REPORT_VERSION = "RecursiveControlArmReportV1"

#: Arm ids from Linear SLM-241 (RSC-A05).
ALL_ARM_IDS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")
BUILT_ARM_IDS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")
DEFERRED_ARM_IDS: tuple[str, ...] = ()

ARM_LABELS: dict[str, str] = {
    "A": "stacked baseline (no z state, unshared blocks)",
    "B": "shared recursive V1 (shared blocks, explicit z state)",
    "C": "shared y-only recurrence (shared blocks, no distinct z state)",
    "D": "recursive no-extra-capacity control (shared blocks, parameter-free z)",
    "E": (
        "stacked + matched state capacity (unshared, non-recursive blocks; "
        "same-shaped state/state_ctx_proj as B's z_latent/ctx_proj injected "
        "once before the blocks run -- mirror image of D)"
    ),
    "F": (
        "unshared depth-matched tower (block-evaluation-matched against B; "
        "no z state, no weight sharing -- MORE parameters than B, see "
        "build_arm_f_dual_view for the honest parameter-nearest residual)"
    ),
    "G": "R=1 shared architecture control (interface-, not behavior-, compatible)",
    "H": (
        "stop-gradient recurrence (arm B's exact construction; y/z detached "
        "between recursive steps -- identical forward values, no cross-step "
        "backprop through the recurrent state)"
    ),
}

#: Canonical ``denoiser_arch`` value each built arm maps onto -- the same
#: field ``ModelBuildConfig``/``TwoTowerConfig`` already use, never an ad hoc
#: parallel selector. Arm G reuses B's denoiser_arch; ``recursive_steps=1``
#: is what distinguishes it. Arm F reuses arm A's tower *class*
#: (``DenoiserTower``) but under its own ``denoiser_arch`` string
#: (``"stacked_depth_matched"``) because its layer count is derived from
#: ``recursive_steps * recursive_transition_layers`` rather than ``n_layers``
#: -- a distinct, named, discoverable config choice, not a shadow path. Arm E
#: (``"stacked_matched_state"``) is a genuinely new tower class
#: (``StackedMatchedStateDenoiserTower``), same convention. Arm H
#: (stop-gradient recurrence) reuses B's exact ``denoiser_arch`` string too --
#: same convention as G -- because H is not a different tower architecture,
#: only a different gradient-flow rule on the identical construction
#: (``SharedRecursiveDenoiserTower(detach_between_steps=True)``); the
#: orthogonal ``detach_between_steps`` flag, not a new ``denoiser_arch``
#: value, is what distinguishes it (see ``construct_arm_tower`` below).
ARM_DENOISER_ARCH: dict[str, str] = {
    "A": "stacked",
    "B": "shared_recursive",
    "C": "shared_recursive_y_only",
    "D": "shared_recursive_no_extra_capacity",
    "E": "stacked_matched_state",
    "F": "stacked_depth_matched",
    "G": "shared_recursive",
    "H": "shared_recursive",
}

#: ``z_state_mode`` each built (non-stacked) arm maps onto. E/F have no z
#: state (same as A -- E's matched capacity lives in ``state``/
#: ``state_ctx_proj``, a distinct tensor pair, not a z-state mode), so both
#: are deliberately absent from this dict, same as A.
ARM_Z_STATE_MODE: dict[str, str] = {
    "B": "full",
    "C": "y_only",
    "D": "parameter_free",
    "G": "full",
    "H": "full",
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
    "E": (
        "matches arm B's total parameter DELTA over a same-n_layers arm A "
        "exactly -- recursive_zstate_parameter_delta(d_model, max_len) -- "
        "via same-shaped state+state_ctx_proj injected once; NOT intended to "
        "match A's raw total parameter count (that is C's/D's kind of "
        "target, not E's -- E is the mirror image of D)"
    ),
    "F": (
        "matches arm B's block_evaluations_per_forward exactly (this row's "
        "construction); does NOT match A's or B's parameter count -- see "
        "build_arm_f_dual_view for the real measured parameter-nearest "
        "alternative construction and its block-evaluation residual instead"
    ),
    "G": (
        "none declared -- architecture-change control at R=1, not a "
        "parameter-matching arm (same parameter profile as B)"
    ),
    "H": (
        "matches arm B's total parameter count and "
        "block_evaluations_per_forward EXACTLY at every config -- identical "
        "construction to B (same denoiser_arch, same z_state_mode='full'), "
        "only detach_between_steps=True differs, which changes no parameter "
        "and no block evaluation, only the backward graph"
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
    parallel/ad hoc implementation. Fails closed for any (currently none)
    deferred or unknown arm ids.
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
    if arch == "stacked_matched_state":
        # Arm E: an unshared, non-recursive DenoiserTower variant (n_layers
        # blocks, each called once -- same block-evaluation count as arm A)
        # with a state/state_ctx_proj pair shape-matched to B's
        # z_latent/ctx_proj, injected once before the blocks run. n_layers
        # here is the plain requested n_layers (never
        # recursive_steps * transition_layers -- that is arm F's dial, not
        # E's), so build_arm_report's generic len(tower.layers) block-eval
        # measurement comes out exactly equal to arm A's automatically.
        return StackedMatchedStateDenoiserTower(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len,
            dropout=dropout,
            kind_ids=kind_ids,
            n_kinds=n_kinds,
        )
    if arch == "stacked_depth_matched":
        # Arm F: same DenoiserTower class as arm A, no weight sharing, no z
        # state -- just recursive_steps * transition_layers independent
        # blocks instead of n_layers, so block_evaluations_per_forward (a
        # generic hasattr(tower, "recursive_steps")-gated measurement in
        # build_arm_report) comes out exactly equal to B's without any
        # special-casing there. This is the block-evaluation-matched
        # primary view; see build_arm_f_dual_view for the paired
        # parameter-nearest view and its (necessarily nonzero) residual.
        return DenoiserTower(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=recursive_steps * transition_layers,
            n_heads=n_heads,
            max_len=max_len,
            dropout=dropout,
            kind_ids=kind_ids,
            n_kinds=n_kinds,
        )
    steps = 1 if arm_id == "G" else recursive_steps
    # Arm H: identical construction to B (same z_state_mode="full"), except
    # detach_between_steps=True -- a pure backward-graph flag, never a
    # parameter/shape/block-evaluation change (see recursive_denoiser.py's
    # module docstring for the exact semantics).
    detach_between_steps = arm_id == "H"
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
        detach_between_steps=detach_between_steps,
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
    if arm_id == "H":
        notes.append(
            "gradient-flow-only variant of arm B: identical construction "
            "(same denoiser_arch, same z_state_mode='full', same parameters, "
            "same block-evaluations) with detach_between_steps=True -- y/z "
            "are detached between recursive steps so a later step's loss "
            "cannot backpropagate through the recurrent state into an "
            "earlier step's block application; the shared weights still "
            "receive gradient from every step's own (same-step) "
            "contribution. See tests/test_models/test_recursive_denoiser.py "
            "for the forward-identity (bit-identical to B) and "
            "gradient-divergence (hook-based mechanism) evidence."
        )
    if arm_id == "F":
        notes.append(
            "block-evaluation-matched against arm B by construction "
            f"(block_evaluations_per_forward={block_evals}); no z-state "
            "parameter bank (same as A); MORE parameters than B because "
            "nothing is shared -- see build_arm_f_dual_view for the "
            "measured parameter-nearest alternative and its block-"
            "evaluation residual."
        )
    if arm_id == "E":
        formula_delta = recursive_zstate_parameter_delta(
            d_model=d_model, max_len=int(getattr(tower, "max_len", 0))
        )
        notes.append(
            "unshared, non-recursive tower (mirror image of D); "
            f"parameter_count_delta_vs_baseline={delta} vs "
            f"recursive_zstate_parameter_delta(d_model={d_model}, "
            f"max_len={getattr(tower, 'max_len', 0)})={formula_delta} -- "
            f"matches_formula={delta == formula_delta}; state/state_ctx_proj "
            "consumed once before the transition blocks run, never "
            "recurrently re-applied."
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


def build_arm_f_dual_view(
    *,
    vocab_size: int,
    d_model: int,
    n_heads: int,
    max_len: int,
    dropout: float = 0.0,
    kind_ids: list[int] | None = None,
    n_kinds: int = 0,
    recursive_steps: int,
    recursive_transition_layers: int,
    noisy_ids: torch.Tensor,
    context: torch.Tensor,
    pad_id: int,
) -> dict[str, Any]:
    """Arm F's two honest matching views against arm B, target + residual.

    ``DenoiserTower`` (arm F's tower -- unshared, no z state) has exactly one
    free dial, ``n_layers``, so it cannot simultaneously match arm B's
    block-evaluation count *and* its parameter count at fixture scale (B
    shares ``recursive_transition_layers`` blocks across ``recursive_steps``
    passes plus a z-state delta; F pays full unshared params per block). This
    function builds and measures **both** constructions instead of asserting
    one is "matched":

    - ``block_evaluation_matched``: ``n_layers = recursive_steps *
      recursive_transition_layers`` -- the construction
      ``construct_arm_tower("F", ...)`` returns and the row
      :func:`build_control_arm_table` reports. Its block-evaluation count is
      exactly B's; its parameter count is real, measured, and reported as a
      surplus over B, never hidden.
    - ``parameter_nearest``: the integer ``n_layers`` whose real measured
      total parameter count is closest to arm B's real measured total
      parameter count. The per-layer parameter cost is derived from two real
      constructed 1-layer/2-layer towers (never a hard-coded constant), so
      the candidate layer count is a formula, not a guess. This construction
      is real and measured but is NOT block-evaluation-matched; its residual
      block-evaluation count vs B is reported honestly.

    Neither view is asserted to be simultaneously matched on both dimensions
    -- this function's whole purpose is reporting the target + residual for
    whichever dimension is not exact, per this repo's C/D-established
    convention.
    """
    baseline_tower = construct_arm_tower(
        "A",
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=recursive_transition_layers,
        n_heads=n_heads,
        max_len=max_len,
        dropout=dropout,
        kind_ids=kind_ids,
        n_kinds=n_kinds,
    )
    b_tower = construct_arm_tower(
        "B",
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=recursive_transition_layers,
        n_heads=n_heads,
        max_len=max_len,
        dropout=dropout,
        kind_ids=kind_ids,
        n_kinds=n_kinds,
        recursive_steps=recursive_steps,
        recursive_transition_layers=recursive_transition_layers,
    )
    b_total = int(sum(p.numel() for p in b_tower.parameters()))
    b_block_evals = recursive_steps * recursive_transition_layers

    def _make(n_layers: int) -> DenoiserTower:
        return DenoiserTower(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=max(1, n_layers),
            n_heads=n_heads,
            max_len=max_len,
            dropout=dropout,
            kind_ids=kind_ids,
            n_kinds=n_kinds,
        )

    def _total_params(tower: nn.Module) -> int:
        return int(sum(p.numel() for p in tower.parameters()))

    one_layer_total = _total_params(_make(1))
    two_layer_total = _total_params(_make(2))
    per_layer_params = two_layer_total - one_layer_total
    if per_layer_params <= 0:
        raise ValueError(
            "measured per-layer parameter delta must be positive; got "
            f"{per_layer_params} (one_layer_total={one_layer_total}, "
            f"two_layer_total={two_layer_total})"
        )
    common_params = one_layer_total - per_layer_params

    block_matched_layers = b_block_evals
    block_matched_tower = _make(block_matched_layers)
    block_matched_report = build_arm_report(
        "F",
        block_matched_tower,
        baseline_tower=baseline_tower,
        noisy_ids=noisy_ids,
        context=context,
        pad_id=pad_id,
    )

    raw_estimate = (b_total - common_params) / per_layer_params
    candidates = sorted(
        {max(1, int(raw_estimate)), max(1, int(raw_estimate) + 1)}
    )
    nearest_layers = candidates[0]
    nearest_tower = _make(nearest_layers)
    nearest_delta = _total_params(nearest_tower) - b_total
    for cand in candidates[1:]:
        cand_tower = _make(cand)
        cand_delta = _total_params(cand_tower) - b_total
        if abs(cand_delta) < abs(nearest_delta):
            nearest_layers, nearest_tower, nearest_delta = cand, cand_tower, cand_delta
    nearest_report = build_arm_report(
        "F",
        nearest_tower,
        baseline_tower=baseline_tower,
        noisy_ids=noisy_ids,
        context=context,
        pad_id=pad_id,
    )

    return {
        "target_arm": "B",
        "target_total_parameters": b_total,
        "target_block_evaluations_per_forward": b_block_evals,
        "per_layer_parameter_cost_formula": {
            "note": (
                "measured from real constructed 1-layer/2-layer DenoiserTower "
                "instances, never hard-coded"
            ),
            "common_parameters": common_params,
            "per_layer_parameters": per_layer_params,
        },
        "block_evaluation_matched": {
            "matching_dimension": "block_evaluations_per_forward",
            "residual_dimension": "parameter_count_total",
            "n_layers": block_matched_layers,
            "report": block_matched_report.as_dict(),
            "parameter_count_delta_vs_target_arm_b": (
                block_matched_report.parameter_count_total - b_total
            ),
        },
        "parameter_nearest": {
            "matching_dimension": "parameter_count_total",
            "residual_dimension": "block_evaluations_per_forward",
            "n_layers": nearest_layers,
            "report": nearest_report.as_dict(),
            "parameter_count_delta_vs_target_arm_b": nearest_delta,
            "block_evaluations_delta_vs_target_arm_b": (
                nearest_layers - b_block_evals
            ),
        },
    }
