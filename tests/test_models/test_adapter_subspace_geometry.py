"""Tests for the TwoTower adapter-subspace objective-geometry profiler (LDI2-02 / SLM-125)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord  # noqa: E402
from slm_training.harnesses.preference import decision_diagnostics as dd  # noqa: E402
from slm_training.harnesses.preference.adapter_subspace_geometry import (  # noqa: E402
    OBJECTIVE_SENSE,
    PROTECTED_OBJECTIVES,
    SubspaceGeometryError,
    _solve,
    _state_view_event,
    profile_adapter_subspace_geometry,
    profile_corpus_cell,
)
from slm_training.harnesses.preference.decision_diagnostics import DiagnosticBudget  # noqa: E402
from slm_training.harnesses.preference.decision_events_v2 import (  # noqa: E402
    DecisionStateV2,
    ObjectiveView,
)
from slm_training.harnesses.preference.local_decisions import split_for_group  # noqa: E402
from slm_training.harnesses.preference.local_train import _guard_objective_tensors  # noqa: E402
from slm_training.models.adapters import TwoTowerAdapterSpec  # noqa: E402
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel  # noqa: E402


def _model() -> TwoTowerModel:
    record = ExampleRecord(
        id="a",
        prompt="Card",
        openui='root = TextContent(":card.title")',
        split="train",
        placeholders=[":card.title"],
    )
    return TwoTowerModel.from_records(
        [record],
        config=TwoTowerConfig(
            d_model=16, n_heads=4, context_layers=1, denoiser_layers=1,
            max_target_len=8, seed=0,
        ),
        device="cpu",
    )


def _group(split: str, seed: str) -> str:
    group = seed
    while split_for_group(group) != split:
        group += "x"
    return group


def _state(group: str, *, kind: str = "component", role: str = "component_slot") -> DecisionStateV2:
    return DecisionStateV2(
        group_id=group,
        architecture="twotower",
        context_text="root=Stack([",
        canvas_ids=(1, 2, 3),
        decision_position=1,
        legal_action_ids=(4, 9, 10),
        decision_kind=kind,
        abstract_state_role=role,
        grammar_state_hash="gsh",
        policy_checkpoint_sha="pcs",
        tokenizer_sha="tsha",
        decode_config_hash="dch",
        verifier_bundle_hash="vbh",
        split=split_for_group(group),
    )


def _view(
    *, good: tuple[int, ...] = (4,), bad: tuple[int, ...] = (9,),
    materializer_id: str = "pareto", trainable: bool = True,
) -> ObjectiveView:
    return ObjectiveView(
        good_action_ids=good,
        bad_action_ids=bad,
        ambiguous_action_ids=(),
        unobserved_action_ids=(),
        weights=(),
        materializer_id=materializer_id,
        materializer_config_hash="cfg",
        trainable=trainable,
    )


def _corpus() -> list[tuple[DecisionStateV2, ObjectiveView]]:
    train_a = _group("train", "obj-a")
    train_b = _group("train", "obj-b")
    held = _group("held_out", "obj-h")
    return [
        (_state(train_a, kind="component", role="component_slot"), _view(good=(4,), bad=(9,))),
        (_state(train_b, kind="grammar_comma", role="grammar_slot"), _view(good=(9,), bad=(4,))),
        (_state(held, kind="component", role="component_slot"), _view(good=(4,), bad=(9,))),
    ]


def _spec_factory(model: TwoTowerModel, cell: dict) -> TwoTowerAdapterSpec:
    rank = int(cell["rank"])
    return TwoTowerAdapterSpec(
        method="low_rank",
        rank=rank,
        alpha=float(2 * rank),
        dropout=0.0,
        target_modules=cell["target_modules"],
        base_compatibility_fingerprint=model.compatibility_fingerprint(),
        base_checkpoint_sha="ckpt",
        tokenizer_sha=model.artifact_identity()["tokenizer_sha"],
    )


_RESTRICTED = ("attn_q", "attn_v")
_BROADER = ("attn_q", "attn_k", "attn_v", "attn_out")


def _attach(model: TwoTowerModel, rank: int = 4, modules=_RESTRICTED):
    model.attach_adapter(_spec_factory(model, {"rank": rank, "target_modules": modules}))
    return list(model.adapter_parameters())


# --- End-to-end profiling -------------------------------------------------------------


def test_profile_cell_keeps_parent_grad_free_and_reports_active_objectives() -> None:
    model = _model()
    params = _attach(model)
    cell = profile_corpus_cell(model, _corpus(), params, objective="ftpo_set")
    assert cell["status"] == "profiled"
    # Frozen-parent invariant: the wrapped base weight never receives a gradient.
    base = model.denoiser.layers[0].self_attn.q_proj.base.weight
    assert base.grad is None
    # lora_B carries gradient at the fresh (zero-delta) point, so objectives are active.
    variants = cell["pooled"]["gradient_variants"]
    assert variants["unit_norm"]["status"] == "solved"
    assert set(variants) == {"raw", "unit_norm"}


def test_support_is_reported_before_gradients() -> None:
    model = _model()
    params = _attach(model)
    cell = profile_corpus_cell(model, _corpus(), params)
    support = cell["support"]
    assert "held_out_coverage" in support
    assert cell["profiled_states"] == 3  # two train + one held-out, all trainable pairs


def test_non_trainable_and_empty_partitions_are_excluded_with_reason() -> None:
    model = _model()
    params = _attach(model)
    corpus = _corpus() + [
        (_state(_group("train", "obj-shadow")), _view(trainable=False)),
    ]
    cell = profile_corpus_cell(model, corpus, params)
    reasons = {entry["reason"] for entry in cell["excluded_views"]}
    assert "non_trainable_view" in reasons
    assert cell["profiled_states"] == 3  # the shadow view is excluded


# --- Legal-space gradient correctness -------------------------------------------------


def test_legal_space_good_mass_gradient_matches_finite_difference() -> None:
    state, view = _corpus()[0]
    event = _state_view_event(state, view)  # good=(4,), bad=(9,), legal=(4,9,10)
    logits = torch.randn(16, dtype=torch.float64, requires_grad=True)
    tensors = _guard_objective_tensors(
        logits, event, objective="ftpo_set", probability_space="legal_tokens"
    )
    good_mass = -tensors["good_probability_mass"]  # un-negate the minimization form
    (grad,) = torch.autograd.grad(good_mass, logits)

    eps = 1e-6
    index = 4  # a good, legal token
    with torch.no_grad():
        plus = logits.clone()
        plus[index] += eps
        minus = logits.clone()
        minus[index] -= eps

        def _good_mass(vec: torch.Tensor) -> float:
            t = _guard_objective_tensors(
                vec, event, objective="ftpo_set", probability_space="legal_tokens"
            )
            return float(-t["good_probability_mass"])

        finite = (_good_mass(plus) - _good_mass(minus)) / (2 * eps)
    assert abs(finite - float(grad[index])) < 1e-4


def test_sign_conventions_reproduce_toy_directions() -> None:
    state, view = _corpus()[0]
    event = _state_view_event(state, view)
    logits = torch.zeros(16, requires_grad=True)
    tensors = _guard_objective_tensors(
        logits, event, objective="ftpo_set", probability_space="legal_tokens"
    )
    # good_probability_mass is stored as -good_mass (minimize == raise good mass):
    # its gradient w.r.t. the good-token logit must be negative.
    (g_good,) = torch.autograd.grad(
        tensors["good_probability_mass"], logits, retain_graph=True
    )
    assert g_good[4] < 0
    # bad_probability_mass is +bad_mass (minimize): gradient w.r.t. the bad logit positive.
    (g_bad,) = torch.autograd.grad(tensors["bad_probability_mass"], logits)
    assert g_bad[9] > 0
    assert OBJECTIVE_SENSE["good_probability_mass"].startswith("minimize(-good_mass)")


# --- Solvers / transforms -------------------------------------------------------------


def test_solve_marks_zero_gradient_objectives_inactive() -> None:
    # Two active, one all-zero objective; the zero one must be excluded explicitly.
    active_a = [torch.tensor([1.0, 0.0])]
    active_b = [torch.tensor([0.0, 1.0])]
    dead = [torch.zeros(2)]
    result = _solve(
        {"loss": active_a, "bad_probability_mass": active_b, "mean_margin": dead},
        [torch.nn.Parameter(torch.zeros(2))],
    )
    assert result["status"] == "solved"
    assert "mean_margin" in result["inactive_objectives"]
    assert result["active_objectives"] == ["bad_probability_mass", "loss"]


def test_solve_reports_all_transforms_and_is_deterministic() -> None:
    grads = {
        "loss": [torch.tensor([1.0, 0.5])],
        "bad_probability_mass": [torch.tensor([-0.5, 1.0])],
    }
    params = [torch.nn.Parameter(torch.zeros(2))]
    first = _solve(grads, params)
    second = _solve(
        {k: [t.clone() for t in v] for k, v in grads.items()},
        [torch.nn.Parameter(torch.zeros(2))],
    )
    for key in ("weighted_mean", "pcgrad", "mgda", "sgd_first_step", "adamw_first_step", "adam_first_step"):
        assert key in first
    # MGDA mixing weights and PCGrad conflict counts are deterministic for fixed input.
    assert first["mgda"]["weights"] == second["mgda"]["weights"]
    assert first["pcgrad"]["report"]["conflict_count"] == second["pcgrad"]["report"]["conflict_count"]


def test_unit_norm_variant_differs_from_raw() -> None:
    model = _model()
    params = _attach(model)
    cell = profile_corpus_cell(model, _corpus(), params)
    variants = cell["pooled"]["gradient_variants"]
    # Both variants are solved; unit-normalization changes the MGDA duality/norm geometry.
    assert variants["raw"]["status"] == "solved"
    assert variants["unit_norm"]["status"] == "solved"
    assert variants["raw"]["mgda"]["report"]["norm_sq"] != variants["unit_norm"]["mgda"]["report"]["norm_sq"]


# --- Strata ---------------------------------------------------------------------------


def test_exact_objective_signature_strata_split_by_partition() -> None:
    model = _model()
    params = _attach(model)
    cell = profile_corpus_cell(model, _corpus(), params)
    strata = cell["strata"]
    assert set(strata) == {"decision_kind", "abstract_state_role", "objective_signature"}
    # The two train states have swapped good/bad partitions -> distinct objective signatures.
    assert len(strata["objective_signature"]) >= 2
    assert set(strata["decision_kind"]) == {"component", "grammar_comma"}


# --- Matrix + wall + telemetry --------------------------------------------------------


def test_matrix_profiles_complete_and_carry_telemetry() -> None:
    matrix = [
        {"rank": 2, "target_modules": _RESTRICTED},
        {"rank": 4, "target_modules": _RESTRICTED},
        {"rank": 8, "target_modules": _BROADER},
    ]
    report = profile_adapter_subspace_geometry(
        _model, _corpus(), _spec_factory, matrix, budget=DiagnosticBudget(max_wall_minutes=3.0)
    )
    assert report["status"] == "completed"
    assert report["kind"] == "adapter_subspace_geometry"
    assert set(report["result"]) == {"rank2:attn_q+attn_v", "rank4:attn_q+attn_v", "rank8:attn_k+attn_out+attn_q+attn_v"}
    assert "result_content_sha" in report
    cell = report["result"]["rank2:attn_q+attn_v"]
    telemetry = cell["telemetry"]
    assert telemetry["forward_passes"] > 0 and telemetry["backward_passes"] > 0
    assert cell["adapter_parameter_dimensions"] > 0
    assert cell["peak_memory_bytes"] > 0


def test_expired_budget_yields_stopped_record_with_no_partial_result(monkeypatch) -> None:
    # Advance the shared monotonic clock past the deadline before the first stage runs.
    ticks = iter([0.0] + [10_000.0] * 50)
    monkeypatch.setattr(dd.time, "monotonic", lambda: next(ticks))
    report = profile_adapter_subspace_geometry(
        _model,
        _corpus(),
        _spec_factory,
        [{"rank": 2, "target_modules": _RESTRICTED}],
        budget=DiagnosticBudget(max_wall_minutes=1.0),
    )
    assert report["status"] == "expired"
    assert report["result"] is None  # never a partial artifact
    assert "result_content_sha" not in report


def test_empty_corpus_is_not_authorized_and_empty_matrix_raises() -> None:
    report = profile_adapter_subspace_geometry(
        _model, [], _spec_factory, [{"rank": 2, "target_modules": _RESTRICTED}]
    )
    assert report["status"] == "not_authorized"
    assert report["result"] is None
    with pytest.raises(SubspaceGeometryError, match="matrix must not be empty"):
        profile_adapter_subspace_geometry(_model, _corpus(), _spec_factory, [])


def test_protected_objectives_cover_the_guarded_quantities() -> None:
    assert PROTECTED_OBJECTIVES == (
        "loss", "good_probability_mass", "bad_probability_mass", "mean_margin"
    )
