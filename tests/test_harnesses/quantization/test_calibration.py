"""Tests for CAP3-02 grammar-stratified calibration and low-bit adaptation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.distill.grammar_trace import (
    GrammarTraceRecorder,
)
from slm_training.harnesses.distill.trace_store import TraceStore
from slm_training.harnesses.quantization.calibration import (
    PRIMARY_STRATEGIES,
    CalibrationSample,
    assign_production_weights,
    build_calibration_corpus,
    calibrate_scales_ptq,
    load_grammar_decision_traces,
    mixed_task_distillation_objective,
    qat_reconstruct_local_scorer,
    sample_active_counterexample,
    sample_low_margin,
    sample_random_production,
    sample_sensitivity_weighted,
)
from slm_training.models.local_action_head import LocalFlatHead, StateContext
from slm_training.models.quantization import ternary_format


ACTIONS = ["a", "b", "c", "d", "e"]


def _make_synthetic_records(
    n: int = 32,
    *,
    seed: int = 0,
    include_counterexamples: bool = False,
) -> list[dict[str, Any]]:
    import random

    rng = random.Random(seed)
    recorder = GrammarTraceRecorder(
        run_id="r1",
        checkpoint_id="ckpt",
        dataset_id="ds",
        example_id="ex",
        seed=seed,
        capture_logits=True,
        capture_sensitivity="grad_norm",
    )
    for i in range(n):
        legal = rng.sample(ACTIONS, k=rng.randint(2, len(ACTIONS)))
        selected = rng.choice(legal)
        logits = [rng.gauss(0, 1) for _ in legal]
        idx = legal.index(selected)
        margin = rng.choice([0.1, 0.5, 2.0])
        logits[idx] += margin
        recorder.record(
            state_fingerprint=f"state-{i % 6}",
            state_signature_version="1",
            legal_action_ids=legal,
            selected_action_id=selected,
            compiler_coverage="complete",
            logits_or_energies=logits,
            scope_signature=f"scope-{i % 3}",
            template_signature=f"tpl-{i % 2}",
            sensitivity={"grad_norm": rng.random()},
            verification_outcome="counterexample" if include_counterexamples and i % 7 == 0 else None,
        )
    records = recorder.finalize()
    for record in records:
        record["kind"] = "grammar_decision"
    return records


def _write_records(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    store = TraceStore(tmp_path, run_id="r1")
    for record in records:
        record["kind"] = "grammar_decision"
        store.append(record)
    return tmp_path


def _records_to_samples(records: list[dict[str, Any]]) -> list[CalibrationSample]:
    # Use the loader on a temporary store so we exercise the real path.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_records(root, records)
        samples, _, _ = load_grammar_decision_traces(root)
    return assign_production_weights(samples)


def test_load_traces_filters_by_kind(tmp_path: Path) -> None:
    store = TraceStore(tmp_path, run_id="r1")
    records = _make_synthetic_records(n=5)
    for record in records:
        store.append(record)
    # Append a decode trace that should be ignored.
    store.append({"version": 3, "kind": "decode", "meta": {}})
    samples, coverage, violations = load_grammar_decision_traces(tmp_path)
    assert len(samples) == 5
    assert coverage["n"] == 5
    assert not violations


def test_sampling_deterministic_and_hashes_stable() -> None:
    records = _make_synthetic_records(n=24)
    samples = _records_to_samples(records)
    manifest1, _ = build_calibration_corpus(
        samples,
        "uniform_state",
        12,
        seed=42,
        test_split_hashes=["never-used"],
    )
    manifest2, _ = build_calibration_corpus(
        samples,
        "uniform_state",
        12,
        seed=42,
        test_split_hashes=["never-used"],
    )
    assert manifest1.calibration_split_hashes == manifest2.calibration_split_hashes
    assert manifest1.sample_count == 12


def test_primary_strategies_return_equal_sample_counts() -> None:
    records = _make_synthetic_records(n=64)
    samples = _records_to_samples(records)
    counts: dict[str, int] = {}
    for strategy in sorted(PRIMARY_STRATEGIES):
        manifest, _ = build_calibration_corpus(
            samples,
            strategy,
            16,
            seed=7,
            test_split_hashes=[],
        )
        counts[strategy] = manifest.sample_count
    assert len(set(counts.values())) == 1
    assert list(counts.values())[0] == 16


def test_no_test_leakage_assertion_raises_on_overlap() -> None:
    records = _make_synthetic_records(n=8)
    samples = _records_to_samples(records)
    manifest, selected = build_calibration_corpus(
        samples,
        "random_production",
        4,
        seed=1,
        test_split_hashes=[],
    )
    # Inject an overlap.
    manifest.test_split_hashes.append(manifest.calibration_split_hashes[0])
    with pytest.raises(ValueError, match="overlap"):
        manifest.assert_no_test_leakage()


def test_uniform_state_covers_states() -> None:
    records = _make_synthetic_records(n=64)
    samples = _records_to_samples(records)
    manifest, selected = build_calibration_corpus(
        samples,
        "uniform_state",
        20,
        seed=3,
        test_split_hashes=[],
    )
    unique_states = {s.state_fingerprint for s in selected}
    # There are only 6 synthetic states; uniform_state should reach all of them.
    assert len(unique_states) == 6


def test_uniform_state_action_covers_pairs() -> None:
    records = _make_synthetic_records(n=64)
    samples = _records_to_samples(records)
    manifest, selected = build_calibration_corpus(
        samples,
        "uniform_state_action",
        24,
        seed=3,
        test_split_hashes=[],
    )
    pairs = {(s.state_fingerprint, s.selected_action_id) for s in selected}
    assert len(pairs) >= min(24, 6 * len(ACTIONS))


def test_scope_template_stratified_covers_bins() -> None:
    records = _make_synthetic_records(n=64)
    samples = _records_to_samples(records)
    manifest, selected = build_calibration_corpus(
        samples,
        "scope_template_stratified",
        24,
        seed=3,
        test_split_hashes=[],
    )
    bins = {(s.scope_signature, s.template_signature) for s in selected}
    assert len(bins) >= 2


def test_low_margin_prefers_small_margins() -> None:
    records = _make_synthetic_records(n=64)
    samples = _records_to_samples(records)
    random_indices = sample_random_production(samples, 20, random.Random(4))
    low_indices = sample_low_margin(samples, 20, random.Random(4))
    random_margins = [samples[i].top1_margin for i in random_indices if samples[i].top1_margin is not None]
    low_margins = [samples[i].top1_margin for i in low_indices if samples[i].top1_margin is not None]
    assert low_margins
    assert sum(low_margins) / len(low_margins) < sum(random_margins) / len(random_margins)


def test_sensitivity_weighted_prefers_high_sensitivity() -> None:
    records = _make_synthetic_records(n=64)
    samples = _records_to_samples(records)
    random_indices = sample_random_production(samples, 20, random.Random(5))
    sens_indices = sample_sensitivity_weighted(samples, 20, random.Random(5))
    random_sens = [samples[i].sensitivity_score for i in random_indices if samples[i].sensitivity_score is not None]
    sens_sens = [samples[i].sensitivity_score for i in sens_indices if samples[i].sensitivity_score is not None]
    assert sens_sens
    assert sum(sens_sens) / len(sens_sens) > sum(random_sens) / len(random_sens)


def test_active_counterexample_does_not_read_test_data() -> None:
    records = _make_synthetic_records(n=16, include_counterexamples=False)
    samples = _records_to_samples(records)
    selected = sample_active_counterexample(samples, 10, random.Random(6))
    assert selected == []

    records_ce = _make_synthetic_records(n=16, include_counterexamples=True)
    samples_ce = _records_to_samples(records_ce)
    selected_ce = sample_active_counterexample(samples_ce, 10, random.Random(6))
    assert all(samples_ce[i].verification_outcome == "counterexample" for i in selected_ce)


def test_calibration_manifest_fields() -> None:
    records = _make_synthetic_records(n=16)
    samples = _records_to_samples(records)
    manifest, _ = build_calibration_corpus(
        samples,
        "hybrid_coverage_margin",
        8,
        seed=0,
        checkpoint_id="ckpt",
        teacher_id="teacher",
        test_split_hashes=[],
        inclusion_rules={"min_legal_actions": 2},
        exclusion_rules={"coverage": "none"},
    )
    assert manifest.schema_version is not None
    assert manifest.checkpoint_id == "ckpt"
    assert manifest.teacher_id == "teacher"
    assert manifest.sampling_strategy == "hybrid_coverage_margin"
    assert manifest.no_test_leakage_asserted
    assert manifest.bin_edges is not None
    assert manifest.raw_production_frequency_weights
    assert "n" in manifest.coverage_fields


def test_calibrate_scales_ptq_changes_weight() -> None:
    w = torch.randn(32, 32)
    q, scale, _ = calibrate_scales_ptq(w, ternary_format())
    assert not torch.equal(q, w)
    assert scale.numel() >= 1


def test_qat_updates_shadow_embeddings_and_gradients_flow() -> None:
    head = LocalFlatHead(hidden_dim=8)
    fmt = ternary_format()
    legal = ["x", "y", "z"]
    hidden = torch.randn(1, 8)
    # Warm embeddings.
    with torch.no_grad():
        head.score(hidden, StateContext(state_family_id="synthetic"), legal)
    originals = {a: head.action_embeddings[a].data.clone() for a in legal}
    teacher_logits = torch.tensor([[0.5, 1.0, 0.2]])
    result = qat_reconstruct_local_scorer(
        head,
        fmt,
        [(hidden, legal, teacher_logits)],
        steps=5,
        lr=0.5,
    )
    assert result["status"] == "ok"
    assert result["final_loss"] is not None
    changed = sum(
        1
        for a in legal
        if not torch.equal(head.action_embeddings[a].data, originals[a])
    )
    assert changed > 0


def test_qat_does_not_touch_other_model_parameters() -> None:
    class ToyWithHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.local_head = LocalFlatHead(8)
            self.unrelated = torch.nn.Linear(8, 8)

    model = ToyWithHead()
    unrelated_before = model.unrelated.weight.data.clone()
    fmt = ternary_format()
    legal = ["x", "y"]
    hidden = torch.randn(1, 8)
    with torch.no_grad():
        model.local_head.score(hidden, StateContext(state_family_id="synthetic"), legal)
    teacher_logits = torch.tensor([[0.0, 1.0]])
    qat_reconstruct_local_scorer(
        model.local_head,
        fmt,
        [(hidden, legal, teacher_logits)],
        steps=3,
        lr=0.5,
    )
    assert torch.equal(model.unrelated.weight.data, unrelated_before)


def test_mixed_task_distillation_loss_is_scalar() -> None:
    logits = torch.randn(2, 4, requires_grad=True)
    teacher = torch.softmax(torch.randn(2, 4), dim=-1)
    target = torch.tensor([1, 2])
    loss = mixed_task_distillation_objective(logits, teacher, target)
    assert loss.dim() == 0
    loss.backward()
    assert logits.grad is not None


def test_build_calibration_corpus_rejects_unknown_strategy() -> None:
    sample = CalibrationSample(
        trace_id="t0",
        state_fingerprint="fp",
        state_signature_version="1",
        legal_action_ids=("a", "b"),
        selected_action_id="a",
        target_action_ids=(),
        top1_margin=1.0,
        posterior_entropy_bits=0.5,
        scope_signature="s",
        template_signature=None,
        production_weight=1.0,
        bin_id=None,
        sensitivity_score=None,
        verification_outcome=None,
    )
    with pytest.raises(ValueError):
        build_calibration_corpus([sample], "not_a_strategy", 1)
