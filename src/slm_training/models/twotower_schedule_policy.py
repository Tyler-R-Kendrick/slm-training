"""TwoTower numeric weight/schedule capability matrix (RSC-A06 / SLM-242).

Generalizes the SLM-138 recursive deep-supervision length/zero-sum fix into a
fail-closed gate for every TwoTower numeric vector/schedule config field this
audit covers. See docs/design/rsc-a06-numeric-schedule-validation-20260721.md
for the field inventory, what is covered, and what is explicitly deferred.

This module is deliberately torch-free (pure stdlib + ``harness_core``
primitives) so both ``ModelBuildConfig`` (``slm_training.harnesses.model_build.config``,
plain dataclass, no torch import) and ``TwoTowerConfig``
(``slm_training.models.twotower``) can validate against the *same* rules
without ``ModelBuildConfig`` paying for a torch import. Every entry point
accepts a config-like object and reads fields with ``getattr(..., default)``
so it works against either dataclass (and tolerates older configs that
predate a given field — see the migration note in each rule).

Non-goals: this module never changes an intended default weight/schedule; it
only decides whether a supplied value is well-formed and whether the active
config gives it a capability to act on.
"""

from __future__ import annotations

from typing import Any

from slm_training.data.diffusion.adapter import POLICIES as DIFFUSION_POLICIES
from slm_training.harness_core.schedule_validation import (
    ScheduleValidationError,
    exact_length_vector,
    non_negative_scalar,
    paired_equal_length_sequences,
    positive_sum_vector,
    strictly_increasing_sequence,
    supported_capability_requirement,
    unique_enum_sequence,
)

__all__ = [
    "ScheduleValidationError",
    "validate_recursive_depth_supervision_weights",
    "validate_grammar_ltr_stages",
    "validate_diffusion_policies",
    "validate_diffusion_length_buckets",
    "validate_diffusion_length_loss_weight",
    "validate_mask_range",
    "validate_targeted_margin_family_weights",
    "validate_slot_component_class_weights",
    "validate_slot_component_priors",
    "validate_twotower_numeric_schedule",
]


# RSC-A06 migration contract: an object that lacks a field entirely (a
# pre-field legacy checkpoint/config dict, or a plain test namespace) must
# validate exactly as if the field were present at *its actual TwoTowerConfig
# default* -- never a blanket ()/0, which would be wrong for fields whose
# real default is non-empty (e.g. diffusion_policies, mask_min/mask_max) and
# would reject a legacy config that never touched this field.
_FIELD_DEFAULTS: dict[str, Any] = {
    "recursive_depth_supervision_weights": (),
    "recursive_steps": 1,
    "denoiser_arch": "stacked",
    "grammar_ltr_stages": None,
    "diffusion_policies": DIFFUSION_POLICIES,
    "diffusion_length_buckets": (32, 64, 96, 128, 192, 256, 384, 512),
    "diffusion_length_loss_weight": 0.1,
    "mask_pattern": "random",
    "mask_min": 0.15,
    "mask_max": 0.85,
    "targeted_margin_family_weights": (),
    "slot_component_class_weights": (),
    "slot_component_loss_weight": 0.0,
    "slot_component_decode_weight": 0.0,
    "slot_component_lexeme_priors": (),
    "slot_component_lexeme_prior_weight": 0.0,
    "slot_component_span_priors": (),
    "slot_component_span_prior_weight": 0.0,
}


def _get(cfg: Any, name: str, default: Any) -> Any:
    fallback = _FIELD_DEFAULTS.get(name, default)
    value = getattr(cfg, name, fallback)
    return fallback if value is None else value


def validate_recursive_depth_supervision_weights(cfg: Any) -> None:
    """SLM-138 deep-supervision weights: the field this issue generalizes from.

    - empty tuple == feature off (always valid; new/legacy configs alike).
    - non-empty: every weight finite & >= 0, sum > 0 (rejects all-zero
      erasure), length exactly ``recursive_steps`` (rejects the
      ``min(len(depth_logits), len(weights))`` silent-truncation pattern —
      the recursive denoiser always emits exactly ``recursive_steps`` depth
      logits, see ``SharedRecursiveDenoiserTower.recursive_outputs``).
    - capability: only ``denoiser_arch="shared_recursive"`` implements
      ``recursive_outputs``; a non-empty vector under any other arch was
      previously silently ignored (non-recursive capability ignore).
    """
    field = "recursive_depth_supervision_weights"
    weights = tuple(_get(cfg, field, ()))
    if not weights:
        return
    positive_sum_vector(weights, field=field)
    recursive_steps = int(_get(cfg, "recursive_steps", 1) or 1)
    exact_length_vector(weights, recursive_steps, field=field)
    denoiser_arch = str(_get(cfg, "denoiser_arch", "stacked") or "stacked")
    supported_capability_requirement(
        condition=True,
        capability_ok=denoiser_arch == "shared_recursive",
        field=field,
        reason=(
            f"non-empty {field} requires denoiser_arch='shared_recursive' "
            f"(the only arch implementing recursive_outputs); got "
            f"denoiser_arch={denoiser_arch!r}"
        ),
    )


def validate_grammar_ltr_stages(cfg: Any) -> None:
    """Progressive LTR canvas stages.

    ``None`` means "use the built-in default" and is always valid. When
    provided, every stage must be an int > 1 (``_ltr_canvas_stages`` filters
    ``s <= 1`` out silently — a stage of 1 would be dead weight configured but
    never applied) and stages must be strictly increasing (no duplicates, no
    unsorted values for the downstream dedup pass to silently reorder).
    """
    field = "grammar_ltr_stages"
    stages = getattr(cfg, field, None)
    if stages is None:
        return
    stages = tuple(stages)
    if not stages:
        raise ScheduleValidationError(field, "must be non-empty when provided")
    for s in stages:
        if not isinstance(s, int) or isinstance(s, bool):
            raise ScheduleValidationError(field, f"every stage must be an int, got {s!r}")
        if s <= 1:
            raise ScheduleValidationError(
                field, f"every stage must be > 1 (stages <= 1 are silently dropped), got {s!r}"
            )
    strictly_increasing_sequence(stages, field=field)


def validate_diffusion_policies(cfg: Any) -> None:
    """Online corruption policy mixture: non-empty, unique, from the known set."""
    field = "diffusion_policies"
    policies = tuple(_get(cfg, field, ()))
    if not policies:
        raise ScheduleValidationError(field, "must be non-empty")
    unique_enum_sequence(policies, field=field, allowed=frozenset(DIFFUSION_POLICIES))


def validate_diffusion_length_buckets(cfg: Any) -> None:
    """Length-bucket boundaries for the diffusion length head: positive, strictly increasing."""
    field = "diffusion_length_buckets"
    buckets = tuple(_get(cfg, field, ()))
    if not buckets:
        raise ScheduleValidationError(field, "must be non-empty")
    for b in buckets:
        if not isinstance(b, int) or isinstance(b, bool) or b <= 0:
            raise ScheduleValidationError(field, f"every bucket must be a positive int, got {b!r}")
    strictly_increasing_sequence(buckets, field=field)


def validate_diffusion_length_loss_weight(cfg: Any) -> None:
    """Shape-only check for ``diffusion_length_loss_weight`` (finite, >= 0).

    NOT gated on ``mask_pattern == "diffusion"`` even though
    ``TwoTowerModel`` only builds ``self.length_head`` (and only ever applies
    this weight) in that mode: both ``TwoTowerConfig()`` and
    ``ModelBuildConfig()`` ship a *default* ``diffusion_length_loss_weight =
    0.1`` under the *default* ``mask_pattern = "random"``, i.e. the shipped
    default is itself in the "configured but currently inert" state. A
    capability rule here would reject the bare default, which the issue's
    non-goals forbid working around by changing the default. This is exactly
    the fail-open shape (a coefficient with no effect unless a separate
    switch is also flipped) the audit is meant to catch, so it is recorded as
    a found defect in the design note rather than silently accepted or papered
    over with a rule that only fires on non-default values.
    """
    field = "diffusion_length_loss_weight"
    non_negative_scalar(_get(cfg, field, 0.0), field=field)


def validate_mask_range(cfg: Any) -> None:
    """``0 <= mask_min <= mask_max <= 1``.

    ``_mask_targets``/``DiffusionConfig`` sample ``uniform(mask_min, mask_max)``
    per row; an inverted or out-of-range pair is a silent, statistically wrong
    masking schedule rather than a crash.
    """
    mask_min = non_negative_scalar(_get(cfg, "mask_min", 0.0), field="mask_min")
    mask_max = non_negative_scalar(_get(cfg, "mask_max", 0.0), field="mask_max")
    if mask_min > 1.0:
        raise ScheduleValidationError("mask_min", f"must be <= 1, got {mask_min!r}")
    if mask_max > 1.0:
        raise ScheduleValidationError("mask_max", f"must be <= 1, got {mask_max!r}")
    if mask_min > mask_max:
        raise ScheduleValidationError(
            "mask_min", f"must be <= mask_max, got mask_min={mask_min!r} mask_max={mask_max!r}"
        )


def validate_targeted_margin_family_weights(cfg: Any) -> None:
    """Per-family confusion-margin weight table: unique family keys, weights >= 0.

    NOTE: as of this audit, ``targeted_margin_family_weights`` is threaded
    through ``ModelBuildConfig``/``TwoTowerConfig``/the factory but has no
    reader in ``twotower.py`` — it is dead config (see the design note's
    found-defects section). Only shape validation is enforced here; there is
    no downstream consumer to write a capability rule against.
    """
    field = "targeted_margin_family_weights"
    pairs = tuple(_get(cfg, field, ()))
    if not pairs:
        return
    seen: set[str] = set()
    for key, weight in pairs:
        if key in seen:
            raise ScheduleValidationError(field, f"duplicate family key {key!r}")
        seen.add(key)
        non_negative_scalar(weight, field=field)


def validate_slot_component_class_weights(cfg: Any) -> None:
    """Slot-component per-class CE weights (``TwoTowerConfig``-only, model-derived).

    Non-empty vectors are currently only ever produced internally by
    ``TwoTowerModel.from_records`` (class-balanced weighting) and are not on
    the ``ModelBuildConfig``/CLI/``apply_runtime_overrides`` surface, so a
    user cannot directly mis-size this vector today. Validation still guards
    checkpoint round-trips and any future exposure: finite & >= 0, sum > 0,
    and the slot-component head must be enabled. The exact
    length-vs-class-count check needs the built tokenizer's component
    inventory and is out of scope for config-time validation (documented as
    deferred in the design note).
    """
    field = "slot_component_class_weights"
    weights = tuple(_get(cfg, field, ()))
    if not weights:
        return
    positive_sum_vector(weights, field=field)
    loss_w = non_negative_scalar(_get(cfg, "slot_component_loss_weight", 0.0), field=field)
    decode_w = non_negative_scalar(_get(cfg, "slot_component_decode_weight", 0.0), field=field)
    supported_capability_requirement(
        condition=True,
        capability_ok=(loss_w > 0.0 or decode_w > 0.0),
        field=field,
        reason=(
            f"non-empty {field} requires slot_component_loss_weight > 0 or "
            "slot_component_decode_weight > 0 (the slot-component head must be enabled)"
        ),
    )


def validate_slot_component_priors(cfg: Any) -> None:
    """Lexeme/span prior tables (``TwoTowerConfig``-only, model-derived).

    Each table is a list of ``(key, score_vector)`` pairs; every vector must
    line up with the same class ordering (equal length across keys), keys
    unique.

    NOTE: unlike ``recursive_depth_supervision_weights``/``diffusion_length_loss_weight``,
    ``slot_component_{lexeme,span}_prior_weight > 0`` with an empty table is
    NOT flagged as a capability violation here: both tables are mined from the
    training corpus in ``TwoTowerModel.from_records`` and legitimately come
    back empty for small/degenerate corpora (no token/span pair meets the
    minimum-count threshold) even though the weight is configured; the
    runtime (`_slot_component_logits`) already treats ``weight > 0 and not
    priors`` as an intentional no-op, not an error. Flagging it here would be
    a false positive against normal small-corpus training, not a
    misconfiguration -- see the design note's found-defects section.
    """
    for field, weight_field in (
        ("slot_component_lexeme_priors", "slot_component_lexeme_prior_weight"),
        ("slot_component_span_priors", "slot_component_span_prior_weight"),
    ):
        pairs = tuple(_get(cfg, field, ()))
        non_negative_scalar(_get(cfg, weight_field, 0.0), field=weight_field)
        if pairs:
            paired_equal_length_sequences(pairs, field=field)


def validate_twotower_numeric_schedule(cfg: Any) -> None:
    """Run every in-scope TwoTower numeric vector/schedule rule against ``cfg``.

    Accepts any object exposing the relevant attributes (``TwoTowerConfig``,
    ``ModelBuildConfig``, or a plain namespace in tests) — missing attributes
    fall back to the same defaults the field declares, so legacy configs that
    predate a field validate exactly as if the field were absent (the
    documented migration behavior: default, never hard-fail, for genuinely
    missing/legacy fields).
    """
    validate_recursive_depth_supervision_weights(cfg)
    validate_grammar_ltr_stages(cfg)
    validate_diffusion_policies(cfg)
    validate_diffusion_length_buckets(cfg)
    validate_diffusion_length_loss_weight(cfg)
    validate_mask_range(cfg)
    validate_targeted_margin_family_weights(cfg)
    validate_slot_component_class_weights(cfg)
    validate_slot_component_priors(cfg)
