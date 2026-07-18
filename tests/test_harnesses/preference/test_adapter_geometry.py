"""Torch-gated tests for the LDI2-02 adapter-subspace geometry profiler."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord  # noqa: E402
from slm_training.harnesses.preference.adapter_geometry import (  # noqa: E402
    PROTECTED_QUANTITIES,
    legal_space_quantities,
    profile_adapter_objective_geometry,
    profile_adapter_solvers,
    profile_rank_matrix,
    write_geometry_evidence,
)
from slm_training.harnesses.preference.local_decisions import (  # noqa: E402
    DecisionEventV1,
    split_for_group,
)
from slm_training.harnesses.preference.local_train import _event_logits  # noqa: E402
from slm_training.models.adapters import TwoTowerAdapterSpec  # noqa: E402
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel  # noqa: E402

_HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)


def _model() -> TwoTowerModel:
    model = TwoTowerModel.from_records(
        [ExampleRecord(id="a", prompt="Hero", openui=_HERO, split="train")],
        config=TwoTowerConfig(d_model=32, n_heads=4, context_layers=1, denoiser_layers=1),
        device="cpu",
    )
    model.eval()  # deterministic forward for finite differences
    return model


def _attach(model: TwoTowerModel) -> None:
    model.attach_adapter(
        TwoTowerAdapterSpec(
            method="low_rank",
            rank=2,
            alpha=4.0,
            dropout=0.0,
            target_modules=("attn_q", "attn_v"),
            base_compatibility_fingerprint=model.compatibility_fingerprint(),
            base_checkpoint_sha="ckpt",
            tokenizer_sha=model.artifact_identity()["tokenizer_sha"],
        )
    )


def _event(group: str = "grp") -> DecisionEventV1:
    while split_for_group(group) != "train":
        group += "x"
    return DecisionEventV1(
        event_id="e",
        group_id=group,
        context_text="Generate a card",
        canvas_ids=(1, 0, 0, 0),
        position=1,
        good_token_ids=(2,),
        bad_token_ids=(3,),
        legal_token_ids=(2, 3, 4),
        evidence_kind="counterfactual",
        evidence_confidence=1.0,
        decision_kind="component",
        split="train",
        policy_checkpoint_sha="p",
        tokenizer_sha="t",
        decode_config_hash="d",
        seed=0,
        trajectory_id="tr",
    )


def test_profile_reports_adapter_subspace_geometry() -> None:
    model = _model()
    _attach(model)
    report = profile_adapter_objective_geometry(model, _event())
    assert report.parameter_dim > 0
    assert set(report.gradient_norms) == set(PROTECTED_QUANTITIES)
    for value in report.cosine_alignment.values():
        assert -1.0001 <= value <= 1.0001
    assert isinstance(report.common_descent, bool)


def test_profile_requires_an_attached_adapter() -> None:
    with pytest.raises(ValueError, match="attached adapter"):
        profile_adapter_objective_geometry(_model(), _event())


def test_parent_parameters_are_never_differentiated() -> None:
    model = _model()
    _attach(model)
    profile_adapter_objective_geometry(model, _event())
    parent_grads = [
        param.grad
        for name, param in model.named_parameters()
        if "lora_" not in name.lower()
    ]
    assert all(grad is None for grad in parent_grads)


def test_good_mass_gradient_matches_finite_differences() -> None:
    model = _model()
    _attach(model)
    event = _event()
    report = profile_adapter_objective_geometry(model, event)
    # descent sign for good_mass is -1, so the raw quantity gradient is -descent.
    raw = -report.descent_gradients["good_mass"]
    params = [p for p in model.adapter_parameters() if p.requires_grad]

    # Validate the element carrying the largest analytic gradient (nonzero at the
    # zero-init adapter), mapping the flat index back to its parameter tensor.
    flat_index = int(torch.argmax(raw.abs()))
    analytic = float(raw[flat_index])
    offset = flat_index
    target = None
    local = 0
    for param in params:
        if offset < param.numel():
            target, local = param, offset
            break
        offset -= param.numel()
    assert target is not None

    def good_mass() -> float:
        with torch.no_grad():
            logits = _event_logits(model, event)
            return float(legal_space_quantities(logits, event)["good_mass"])

    step = 1e-2
    with torch.no_grad():
        target.view(-1)[local] += step
    plus = good_mass()
    with torch.no_grad():
        target.view(-1)[local] -= 2 * step
    minus = good_mass()
    with torch.no_grad():
        target.view(-1)[local] += step  # restore
    finite_difference = (plus - minus) / (2 * step)
    assert abs(finite_difference - analytic) < 3e-2


def test_solver_panel_reports_pcgrad_mgda_and_is_deterministic() -> None:
    model = _model()
    _attach(model)
    event = _event()
    report = profile_adapter_solvers(model, event)
    assert report.pcgrad["task_count"] == len(PROTECTED_QUANTITIES)
    assert isinstance(report.common_descent_certified, bool)
    assert set(report.pairwise_cosine) == {
        "loss|good_mass",
        "loss|bad_mass",
        "loss|margin",
        "good_mass|bad_mass",
        "good_mass|margin",
        "bad_mass|margin",
    }
    # Optimizer first-step transforms are reported with in-range cosines.
    assert set(report.optimizer_first_step) == {
        "sgd_min_task_cosine",
        "adam_vs_mean_cosine",
        "adam_norm",
    }
    assert -1.0001 <= report.optimizer_first_step["adam_vs_mean_cosine"] <= 1.0001
    assert report.optimizer_first_step["adam_norm"] >= 0.0
    # Deterministic for a fixed model + event (same seed/order).
    again = profile_adapter_solvers(model, event)
    assert again.mgda["weights"] == report.mgda["weights"]
    assert again.pcgrad == report.pcgrad
    assert again.optimizer_first_step == report.optimizer_first_step


def test_mgda_weights_form_a_convex_combination() -> None:
    model = _model()
    _attach(model)
    report = profile_adapter_solvers(model, _event())
    weights = report.mgda["weights"]
    total = sum(weights)
    # A non-degenerate problem yields a convex combination summing to one.
    assert abs(total - 1.0) < 1e-6 or total == 0.0
    assert all(w >= -1e-9 for w in weights)


def _spec_at_rank(model: TwoTowerModel, rank: int) -> TwoTowerAdapterSpec:
    return TwoTowerAdapterSpec(
        method="low_rank",
        rank=rank,
        alpha=2.0 * rank,
        dropout=0.0,
        target_modules=("attn_q", "attn_v"),
        base_compatibility_fingerprint=model.compatibility_fingerprint(),
        base_checkpoint_sha="ckpt",
        tokenizer_sha=model.artifact_identity()["tokenizer_sha"],
    )


def test_rank_matrix_completes_small_ranks() -> None:
    record = profile_rank_matrix(
        _model, _spec_at_rank, _event(), ranks=(2, 4), max_wall_seconds=120.0
    )
    assert record["status"] == "complete"
    assert record["ranks_profiled"] == [2, 4]
    assert set(record["results"]) == {"2", "4"}
    # A larger rank spans a strictly larger adapter subspace.
    assert record["results"]["4"]["parameter_dim"] > record["results"]["2"]["parameter_dim"]


def test_rank_matrix_emits_expired_record_over_budget() -> None:
    record = profile_rank_matrix(
        _model, _spec_at_rank, _event(), ranks=(2, 4), max_wall_seconds=0.0
    )
    assert record["status"] == "expired"
    assert record["results"] == {}  # no valid geometry survives expiry
    assert record["ranks_profiled"] == []
    assert record["wall_seconds"] >= 0.0


def test_geometry_evidence_writes_deterministic_json(tmp_path) -> None:
    model = _model()
    _attach(model)
    out = tmp_path / "evidence.json"
    evidence = write_geometry_evidence(out, model, _event())
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == evidence
    assert loaded["schema"] == "adapter-geometry-evidence-v1"
    # Honesty marker: fixture evidence is never a training/quality claim.
    assert "no training" in loaded["claim"] and "quality claim" in loaded["claim"]
    assert loaded["parameter_dim"] > 0
    assert "geometry" in loaded and "solvers" in loaded
    assert isinstance(loaded["solvers"]["mgda"]["converged"], bool)


def test_geometry_evidence_requires_an_adapter() -> None:
    with pytest.raises(ValueError, match="attached adapter"):
        write_geometry_evidence("unused.json", _model(), _event())
