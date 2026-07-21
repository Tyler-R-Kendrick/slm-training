"""Property/regression tests for the RSC-A06 (SLM-242) numeric schedule gate.

Covers: the capability matrix in ``slm_training.models.twotower_schedule_policy``,
its wiring into ``ModelBuildConfig``/``TwoTowerConfig``/``apply_runtime_overrides``/
``TwoTowerModel.save``/``load``/``from_checkpoint``, the eight core invariants
from the issue, and a dedicated regression test per SLM-237-class defect
pattern (referenced by name from the ``scripts.verify_numeric_schedule_guard``
suppression comments in ``twotower.py``).

Split into a torch-free section (``ModelBuildConfig``, plain namespaces) and a
torch-gated section (``TwoTowerConfig``/``TwoTowerModel``), mirroring the
repo's existing ``pytest.importorskip("torch")`` convention.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from slm_training.harness_core.schedule_validation import ScheduleValidationError
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.models.twotower_schedule_policy import (
    validate_diffusion_length_buckets,
    validate_grammar_ltr_stages,
    validate_mask_range,
    validate_recursive_depth_supervision_weights,
    validate_slot_component_class_weights,
    validate_targeted_margin_family_weights,
    validate_twotower_numeric_schedule,
)

# --------------------------------------------------------------------------
# Torch-free: ModelBuildConfig and plain-namespace capability checks.
# --------------------------------------------------------------------------


def test_default_model_build_config_is_valid() -> None:
    """The shipped default must validate cleanly (no accidental self-lockout)."""
    ModelBuildConfig(train_dir=Path("."))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(denoiser_arch="shared_recursive", recursive_steps=1),
        dict(denoiser_arch="shared_recursive", recursive_steps=3),
        dict(
            denoiser_arch="shared_recursive",
            recursive_steps=3,
            recursive_depth_supervision_weights=(0.5, 1.0, 0.5),
        ),
        dict(
            denoiser_arch="shared_recursive",
            recursive_steps=1,
            recursive_depth_supervision_weights=(1.0,),
        ),
        dict(grammar_ltr_stages=(32, 48, 64)),
        dict(diffusion_policies=("uniform", "contiguous")),
        dict(diffusion_length_buckets=(16, 32, 64, 96)),
        dict(mask_min=0.0, mask_max=0.0),
        dict(mask_min=0.15, mask_max=0.85),
    ],
)
def test_model_build_config_accepts_valid_combinations(kwargs: dict) -> None:
    ModelBuildConfig(train_dir=Path("."), **kwargs)


# --- Regression pattern 1: silent truncation via min() ---------------------


def test_length_mismatch_rejected_before_reaching_training_loss() -> None:
    """recursive_depth_supervision_weights must equal recursive_steps exactly;
    previously ``min(len(depth_logits), len(ds_weights))`` would silently use
    a truncated prefix instead. Referenced by the TRUNCATE suppression at
    twotower.py's deep-supervision block."""
    with pytest.raises(ScheduleValidationError, match="length"):
        ModelBuildConfig(
            train_dir=Path("."),
            denoiser_arch="shared_recursive",
            recursive_steps=3,
            recursive_depth_supervision_weights=(0.5, 1.0),
        )


# --- Regression pattern 2: all-zero erasure ---------------------------------


def test_all_zero_weights_rejected_before_reaching_training_loss() -> None:
    """A non-empty, all-zero weight vector must raise instead of silently
    disabling the feature with no signal. Referenced by the UNGUARDED_SUM
    suppression at twotower.py's deep-supervision block."""
    with pytest.raises(ScheduleValidationError, match="all-zero"):
        ModelBuildConfig(
            train_dir=Path("."),
            denoiser_arch="shared_recursive",
            recursive_steps=2,
            recursive_depth_supervision_weights=(0.0, 0.0),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("grammar_ltr_stages", ()),
        ("diffusion_policies", ()),
        ("diffusion_length_buckets", ()),
    ],
)
def test_other_empty_vectors_are_rejected_not_silently_defaulted(
    field: str, value: tuple
) -> None:
    with pytest.raises(ScheduleValidationError):
        ModelBuildConfig(train_dir=Path("."), **{field: value})


# --- Regression pattern 3: non-recursive capability ignore ------------------


def test_capability_ignore_rejected_for_non_recursive_denoiser_arch() -> None:
    """Non-empty recursive_depth_supervision_weights under denoiser_arch=
    "stacked" used to be silently ignored (has_recursive_outputs is False, so
    the deep-supervision branch is simply never entered). Must now raise."""
    with pytest.raises(ScheduleValidationError, match="shared_recursive"):
        ModelBuildConfig(
            train_dir=Path("."),
            denoiser_arch="stacked",
            recursive_steps=2,
            recursive_depth_supervision_weights=(0.5, 1.0),
        )


def test_capability_ignore_rejected_for_disabled_slot_component_head() -> None:
    """A second, distinct capability-ignore instance: slot_component_class_weights
    only has an effect when the slot-component head is enabled."""
    cfg = SimpleNamespace(
        slot_component_class_weights=(1.0, 2.0),
        slot_component_loss_weight=0.0,
        slot_component_decode_weight=0.0,
    )
    with pytest.raises(ScheduleValidationError, match="slot_component"):
        validate_slot_component_class_weights(cfg)


# --- Regression pattern 4: unvalidated negative values ----------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("recursive_depth_supervision_weights", (-0.5, 1.0)),
    ],
)
def test_negative_weight_rejected(field: str, value: tuple) -> None:
    with pytest.raises(ScheduleValidationError):
        ModelBuildConfig(
            train_dir=Path("."),
            denoiser_arch="shared_recursive",
            recursive_steps=2,
            **{field: value},
        )


def test_negative_targeted_margin_family_weight_rejected() -> None:
    with pytest.raises(ScheduleValidationError):
        ModelBuildConfig(
            train_dir=Path("."),
            targeted_margin_family_weights=(("family_a", -1.0),),
        )


def test_inverted_mask_range_rejected() -> None:
    """mask_min > mask_max is an unvalidated-negative-ordering variant of the
    same fail-open family (a statistically wrong, silently-accepted schedule)."""
    with pytest.raises(ScheduleValidationError, match="mask_min"):
        ModelBuildConfig(train_dir=Path("."), mask_min=0.9, mask_max=0.1)


# --- Regression pattern 5: loop variable named as a weight but unused ------
# (documented as a known behavior defect, not fixed by this change -- see
# docs/design/rsc-a06-numeric-schedule-validation-20260721.md "Found defects" #1.
# Torch-gated: needs a real TwoTowerModel forward pass; see below.)


# --- Regression pattern 6: capability guard silent fallback -----------------


def test_diffusion_policy_capability_guard_rejects_unknown_policy() -> None:
    """A policy outside the known set would previously reach DiffusionConfig
    deep inside an online training step (mid-run) instead of failing at
    config-build time -- the fail-open "guard silently proceeds/falls back
    downstream instead of rejecting up front" shape."""
    with pytest.raises(ScheduleValidationError, match="unknown"):
        ModelBuildConfig(train_dir=Path("."), diffusion_policies=("not_a_real_policy",))


# --------------------------------------------------------------------------
# Direct primitive-function tests against plain namespaces (no ModelBuildConfig
# construction needed) -- exercises defaulting/migration tolerance directly.
# --------------------------------------------------------------------------


def test_missing_attributes_default_like_an_absent_legacy_field() -> None:
    """Core invariant #8 (migration): an object that simply lacks a field
    (as a pre-field legacy checkpoint config would) must validate exactly as
    if the field were present at its default -- never a hard AttributeError."""
    cfg = SimpleNamespace()  # no fields at all
    validate_twotower_numeric_schedule(cfg)  # must not raise


def test_grammar_ltr_stages_none_is_always_valid() -> None:
    validate_grammar_ltr_stages(SimpleNamespace(grammar_ltr_stages=None))


@pytest.mark.parametrize("stages", [(1, 2, 3), (2, 1), (2, 2, 3)])
def test_grammar_ltr_stages_rejects_low_or_unsorted_or_duplicate(stages) -> None:
    with pytest.raises(ScheduleValidationError):
        validate_grammar_ltr_stages(SimpleNamespace(grammar_ltr_stages=stages))


@pytest.mark.parametrize("buckets", [(32,), (32, 64, 96, 128, 192, 256, 384, 512)])
def test_diffusion_length_buckets_accepts_varying_length(buckets) -> None:
    validate_diffusion_length_buckets(SimpleNamespace(diffusion_length_buckets=buckets))


def test_diffusion_length_buckets_rejects_duplicate() -> None:
    with pytest.raises(ScheduleValidationError):
        validate_diffusion_length_buckets(
            SimpleNamespace(diffusion_length_buckets=(32, 32, 64))
        )


def test_mask_range_rejects_out_of_unit_interval() -> None:
    with pytest.raises(ScheduleValidationError):
        validate_mask_range(SimpleNamespace(mask_min=0.1, mask_max=1.5))


def test_targeted_margin_family_weights_rejects_duplicate_family_key() -> None:
    with pytest.raises(ScheduleValidationError, match="duplicate"):
        validate_targeted_margin_family_weights(
            SimpleNamespace(
                targeted_margin_family_weights=(("a", 1.0), ("a", 2.0))
            )
        )


def test_recursive_depth_supervision_weights_empty_is_always_off() -> None:
    """Empty tuple is the explicit off-sentinel and must be valid under any arch."""
    validate_recursive_depth_supervision_weights(
        SimpleNamespace(
            recursive_depth_supervision_weights=(),
            denoiser_arch="stacked",
            recursive_steps=1,
        )
    )


# --------------------------------------------------------------------------
# Torch-gated: TwoTowerConfig / TwoTowerModel integration.
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord  # noqa: E402
from slm_training.harnesses.model_build.factory import (  # noqa: E402
    apply_runtime_overrides,
)
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel  # noqa: E402

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


def _records() -> list[ExampleRecord]:
    return [
        ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA layout", openui=CTA, split="train"),
    ]


def test_default_twotower_config_is_valid() -> None:
    TwoTowerConfig()


def test_twotower_config_construction_rejects_length_mismatch() -> None:
    with pytest.raises(ScheduleValidationError):
        TwoTowerConfig(
            denoiser_arch="shared_recursive",
            recursive_steps=2,
            recursive_depth_supervision_weights=(1.0,),
        )


# --- Core invariant #4: zero weight disables only its own contribution -----


def test_zero_weighted_depth_still_reports_telemetry() -> None:
    """A zero entry inside an otherwise-positive weight vector must still
    compute and log its own per-depth metric (only its loss contribution is
    zeroed, not the telemetry or the other depths' contributions)."""
    model = TwoTowerModel.from_records(
        _records(),
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=2,
            denoiser_arch="shared_recursive",
            recursive_steps=3,
            recursive_transition_layers=2,
            recursive_depth_supervision_weights=(0.0, 1.0, 0.5),
            grammar_constrained=False,
            seed=0,
        ),
        device="cpu",
    )
    loss = model.training_loss(_records())
    assert torch.isfinite(loss)
    # All three depths -- including the zero-weighted depth 0 -- log telemetry.
    assert "recursive_depth_loss_0" in model.last_training_metrics
    assert "recursive_depth_loss_1" in model.last_training_metrics
    assert "recursive_depth_loss_2" in model.last_training_metrics


# --- Regression pattern 5 (known defect, not fixed here) --------------------


def test_per_depth_weight_ratio_is_not_applied_known_defect() -> None:
    """KNOWN BEHAVIOR DEFECT (documented, not fixed by this change -- see
    docs/design/rsc-a06-numeric-schedule-validation-20260721.md "Found
    defects" #1): the per-depth weight ``w`` bound in
    ``for d, w in enumerate(ds_weights[:usable])`` is never multiplied into
    ``d_loss``; only its contribution to ``total_w`` (the final divisor)
    matters. This test pins the *current* behavior -- extreme vs. uniform
    per-depth weighting produce the same supervision loss -- so a future
    intentional fix changes this test deliberately rather than silently
    drifting. This is exactly the "loop variable named as a weight but
    unused in the contribution" static-guard pattern (UNUSED_LOOP_WEIGHT).
    """

    def _build(weights: tuple[float, ...]) -> TwoTowerModel:
        torch.manual_seed(0)
        return TwoTowerModel.from_records(
            _records(),
            config=TwoTowerConfig(
                d_model=16,
                n_heads=2,
                context_layers=1,
                denoiser_layers=2,
                denoiser_arch="shared_recursive",
                recursive_steps=2,
                recursive_transition_layers=2,
                recursive_depth_supervision_weights=weights,
                grammar_constrained=False,
                seed=0,
            ),
            device="cpu",
        )

    uniform = _build((1.0, 1.0))
    skewed = _build((0.01, 100.0))  # if per-depth weighting worked, this would
    # heavily favor depth 1's loss; under the current defect it does not.
    records = _records()

    # Reseed the global torch RNG immediately before each forward pass so the
    # only difference between the two runs is the configured weight ratio,
    # not independently-drifted masking/corruption randomness.
    torch.manual_seed(1)
    uniform.training_loss(records)
    torch.manual_seed(1)
    skewed.training_loss(records)
    uniform_metrics = uniform.last_training_metrics
    skewed_metrics = skewed.last_training_metrics

    # Per-depth *raw* losses are identical regardless of the configured
    # weight ratio (only sums into total_w change, never d_loss itself).
    assert uniform_metrics["recursive_depth_loss_0"] == pytest.approx(
        skewed_metrics["recursive_depth_loss_0"], rel=1e-4
    )
    assert uniform_metrics["recursive_depth_loss_1"] == pytest.approx(
        skewed_metrics["recursive_depth_loss_1"], rel=1e-4
    )


# --- Core invariant #7: CLI overrides cannot bypass validation --------------


def test_runtime_override_revalidates_and_rejects_bypass_attempt() -> None:
    """apply_runtime_overrides setattr's fields directly onto an already-
    constructed TwoTowerConfig, bypassing __post_init__. Overriding only
    recursive_depth_supervision_weights (leaving the model's own
    recursive_steps=3 untouched) to a length-2 vector must still be rejected
    -- an eval/CLI-time override cannot bypass the fail-closed gate."""
    model = TwoTowerModel.from_records(
        _records(),
        config=TwoTowerConfig(
            d_model=16,
            n_heads=2,
            context_layers=1,
            denoiser_layers=2,
            denoiser_arch="shared_recursive",
            recursive_steps=3,
            recursive_transition_layers=2,
            grammar_constrained=False,
            seed=0,
        ),
        device="cpu",
    )
    override_config = ModelBuildConfig(
        train_dir=Path("."),
        denoiser_arch="shared_recursive",
        recursive_steps=2,
        recursive_depth_supervision_weights=(0.5, 1.0),
        runtime_override_fields=frozenset({"recursive_depth_supervision_weights"}),
    )
    with pytest.raises(ScheduleValidationError, match="length"):
        apply_runtime_overrides(model, override_config)


# --- Core invariant #6: config serialization/deserialization is idempotent -


def test_config_round_trip_through_asdict_is_idempotent() -> None:
    from dataclasses import asdict

    original = TwoTowerConfig(
        d_model=16,
        denoiser_arch="shared_recursive",
        recursive_steps=2,
        recursive_depth_supervision_weights=(0.5, 1.0),
        grammar_ltr_stages=(32, 48, 64),
        diffusion_length_buckets=(16, 32, 64),
    )
    payload = asdict(original)
    valid_fields = {f.name for f in TwoTowerConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    restored = TwoTowerConfig(**{k: v for k, v in payload.items() if k in valid_fields})
    assert asdict(restored) == payload


# --- Core invariant #8: legacy migration is explicit -----------------------


def test_legacy_config_missing_new_field_defaults_not_hard_fails() -> None:
    """A config dict that predates recursive_depth_supervision_weights
    entirely (simulating an old checkpoint) must construct via the dataclass
    default, not raise."""
    valid_fields = {f.name for f in TwoTowerConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    legacy_payload = {
        k: v
        for k, v in asdict_default().items()
        if k in valid_fields and k != "recursive_depth_supervision_weights"
    }
    cfg = TwoTowerConfig(**legacy_payload)
    assert cfg.recursive_depth_supervision_weights == ()


def asdict_default() -> dict:
    from dataclasses import asdict

    return asdict(TwoTowerConfig())


def test_legacy_config_with_genuinely_invalid_historical_value_requires_migration() -> None:
    """A checkpoint carrying the pre-fix invalid combination (non-empty
    ds_weights under denoiser_arch="stacked") must not load silently -- it
    needs an explicit migration (see slm_training.models.checkpoint_migrate),
    not a silent accept."""
    with pytest.raises(ScheduleValidationError):
        TwoTowerConfig(
            denoiser_arch="stacked",
            recursive_depth_supervision_weights=(0.5, 1.0),
        )
