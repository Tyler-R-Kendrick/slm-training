"""SLM-434 (LAR0-07): regression tests for the SLM-308/310 model-knob port.

The ``value_label_mode`` and ``stop_slot_accounting`` knobs were ported from
the never-merged SLM-305/308/310 branch forks so the SLM-317 powered rerun's
historical/improved arms can construct their configs. Both knobs are
default-off: when unset, training and decode behavior is byte-identical to
pre-port main (mutation-count value labels, legacy STOP-slot accounting).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.tree_edit_diffusion import (
    TreeEditDiffusionConfig,
    TreeEditDiffusionModel,
    pairwise_progress_loss,
    parse_statements,
)

RECORDS = [
    ExampleRecord(
        id="a",
        prompt="render the cta",
        openui='root = Stack([cta], "column")\ncta = Button(":cta.label")',
        placeholders=[":cta.label"],
    ),
    ExampleRecord(
        id="b",
        prompt="render the body",
        openui='root = Stack([n0], "column")\nn0 = TextContent(":content.body")',
        placeholders=[":content.body"],
    ),
]

FIXTURE_OVERRIDES = {
    "d_model": 64,
    "denoiser_layers": 2,
    "context_layers": 1,
    "max_chain": 3,
    "max_search_steps": 3,
    "beam_width": 2,
    "expand_per_state": 2,
}

# Metrics keys emitted by the pre-port mutation-count loss path. The port must
# not add bounded-mode keys unless bounded mode is actually selected.
LEGACY_METRIC_KEYS = {"action", "stmt", "comp", "slot", "value", "skipped"}


def _model(**config_overrides) -> TreeEditDiffusionModel:
    torch.manual_seed(0)
    config = TreeEditDiffusionConfig(seed=0, **{**FIXTURE_OVERRIDES, **config_overrides})
    return TreeEditDiffusionModel.from_records(RECORDS, config=config, device="cpu")


def test_default_config_preserves_mutation_count_labels() -> None:
    """Absent knobs = historical behavior: mutation-count value labels, no
    bounded-mode metrics, and the legacy value-label formula."""
    model = _model()
    loss = model.training_loss(RECORDS)
    assert torch.isfinite(loss)
    assert set(model.last_training_metrics) == LEGACY_METRIC_KEYS

    # Spot-check the historical formula on a deterministic single-record run:
    # every non-clean training row's value label is 1 - applied/(max_chain+1).
    torch.manual_seed(0)
    model = _model()
    labels: list[float] = []
    original_mse = torch.nn.functional.mse_loss
    import slm_training.models.tree_edit_diffusion as ted

    def spy(input, target, **kwargs):
        labels.extend(float(v) for v in target)
        return original_mse(input, target, **kwargs)

    original = ted.F.mse_loss
    ted.F.mse_loss = spy
    try:
        model.training_loss(RECORDS)
    finally:
        ted.F.mse_loss = original
    assert labels  # value loss ran
    max_chain = FIXTURE_OVERRIDES["max_chain"]
    for value in labels:
        assert 0.0 <= value <= 1.0
        if value != 1.0:  # clean-state STOP rows keep full value in both modes
            applied = round((1.0 - value) * (max_chain + 1))
            assert value == pytest.approx(1.0 - applied / float(max_chain + 1))


def test_default_decode_matches_explicit_legacy_stop_accounting() -> None:
    """Unset stop_slot_accounting decodes byte-identically to explicit legacy."""
    default_model = _model()
    legacy_model = _model(stop_slot_accounting="legacy")
    default_out = default_model.generate("render the cta", gold=RECORDS[0])
    legacy_out = legacy_model.generate("render the cta", gold=RECORDS[0])
    assert default_out == legacy_out
    assert (
        default_model._generation_evidence == legacy_model._generation_evidence
    )


def test_bounded_distance_mode_uses_oracle_labels_and_pairs() -> None:
    model = _model(value_label_mode="bounded_distance")
    loss = model.training_loss(RECORDS)
    assert torch.isfinite(loss)
    metrics = model.last_training_metrics
    assert "value_bounded" in metrics
    assert "value_unknown_excluded" in metrics
    # Fixture states are within the oracle budget: at least one row got a
    # usable (masked-in) value label.
    assert metrics["value"] >= 0.0
    assert metrics["value_unknown_excluded"] == 0.0


def test_unknown_states_are_excluded_from_value_loss(monkeypatch) -> None:
    """UNKNOWN oracle labels are masked out of the value loss, never coerced."""
    from slm_training.harnesses.experiments import slm308_distance_oracle as oracle

    model = _model(value_label_mode="bounded_distance")

    def unknown_label(*args, **kwargs):
        return oracle.DistanceLabel(
            kind=oracle.DistanceKind.UNKNOWN, reason=oracle.REASON_BUDGET
        )

    monkeypatch.setattr(
        "slm_training.models.tree_edit_diffusion.TreeEditDiffusionModel._distance_label",
        lambda self, *args: unknown_label(),
    )
    loss = model.training_loss(RECORDS)
    assert torch.isfinite(loss)
    metrics = model.last_training_metrics
    assert metrics["value_unknown_excluded"] >= 1.0
    # With every mutated row excluded, only clean-state STOP rows (value 1.0)
    # can contribute to the value loss, if any were sampled this run.
    if "value" in metrics:
        assert metrics["value"] >= 0.0


def test_unknown_knob_values_fail_closed() -> None:
    model = _model(value_label_mode="guess")
    with pytest.raises(ValueError, match="value_label_mode"):
        model.training_loss(RECORDS)
    model = _model(stop_slot_accounting="lenient")
    with pytest.raises(ValueError, match="stop_slot_accounting"):
        model.training_loss(RECORDS)
    model = _model(stop_slot_accounting="lenient")
    with pytest.raises(ValueError, match="stop_slot_accounting"):
        model.generate("render the cta", gold=RECORDS[0])


def test_checkpoint_roundtrip_and_old_payloads_default_off(tmp_path) -> None:
    """New fields round-trip; payloads written before the fields existed load
    with the legacy defaults (pre-port behavior preserved)."""
    model = _model(
        value_label_mode="bounded_distance", stop_slot_accounting="corrected"
    )
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = TreeEditDiffusionModel.from_checkpoint(path)
    assert loaded.config.value_label_mode == "bounded_distance"
    assert loaded.config.stop_slot_accounting == "corrected"

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"].pop("value_label_mode")
    payload["config"].pop("stop_slot_accounting")
    payload["config"].pop("pairwise_progress_margin")
    old_path = tmp_path / "old.pt"
    import shutil

    shutil.copy(path.with_suffix(".tokenizer.json"), old_path.with_suffix(".tokenizer.json"))
    torch.save(payload, old_path)
    loaded_old = TreeEditDiffusionModel.from_checkpoint(old_path)
    assert loaded_old.config.value_label_mode == "mutation_count"
    assert loaded_old.config.stop_slot_accounting == "legacy"


def test_pairwise_progress_loss_margins() -> None:
    parent = torch.tensor([0.5, 0.5])
    improving_child = torch.tensor([0.8, 0.5])
    loss = pairwise_progress_loss(parent, improving_child, margin=0.1)
    # First pair clears the margin (0), second violates it (0.1).
    assert float(loss) == pytest.approx(0.05)
    empty = pairwise_progress_loss(
        torch.zeros(0), torch.zeros(0), margin=0.1
    )
    assert float(empty) == 0.0


def test_distance_oracle_exact_small_case() -> None:
    """The ported SLM-308 oracle runs against main's evolved edit space."""
    from slm_training.harnesses.experiments import slm308_distance_oracle as oracle
    from slm_training.models.tree_edit_diffusion import TreeEditSpace

    oracle.clear_caches()
    seed = parse_statements('root = Stack([], "column")')
    gold = parse_statements(
        'root = Stack([cta], "column")\ncta = Button(":cta.label")'
    )
    label = oracle.distance_to_target(
        seed, gold, space=TreeEditSpace(), inventory=[":cta.label"], node_budget=64
    )
    assert label.kind is oracle.DistanceKind.EXACT
    assert label.distance == 1
    assert label.value_target(8) == pytest.approx(1.0 - 1.0 / 8.0)
    oracle.clear_caches()
