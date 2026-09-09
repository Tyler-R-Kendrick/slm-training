"""Actual search owners: compatible credit, replay, allocation and typed remedies."""

from copy import deepcopy

import pytest

from slm_training.harness_core.activity_contract import contract_digest
from slm_training.autoresearch.evidence_ledger import rank_arms_by_evidence
from slm_training.autoresearch.search.allocation import (
    AllocationPlan,
    FidelityObservation,
    checked_observations,
    fidelity_report,
    next_allocations,
)
from slm_training.autoresearch.search.evidence import (
    SearchEffect,
    SearchIdentity,
    compatible_effects,
    effect_from_loss_reports,
)
from slm_training.autoresearch.search.stagnation import SearchSignal, stagnation_actions


def identity():
    fields = {
        key: "a" * 64
        for key in (
            "suite_digest",
            "selection_digest",
            "data_digest",
            "ancestor_digest",
            "model_digest",
            "host_digest",
            "fidelity_digest",
            "analysis_digest",
        )
    }
    return SearchIdentity(
        endpoint="denoising_ce",
        endpoint_version="v1",
        units="nats_per_masked_token",
        direction="minimize",
        estimator="masked_ce/v1",
        independence="conditional_on_ancestor",
        **fields,
    )


def effect(**changes):
    values = dict(
        identity=identity(),
        slug="useful",
        treatment_id="b" * 64,
        replicate_id="seed-7",
        comparison_id="c" * 64,
        attempt_id="first",
        complete=True,
        control=2.0,
        candidate=1.0,
        paired_case_ids=("case",),
        paired_root_ids=("root",),
        required_case_ids=("case",),
    )
    return SearchEffect(**(values | changes))


def test_evidence_changes_selection_retry_does_not():
    candidates = ["unmeasured", "useful"]
    assert rank_arms_by_evidence(candidates, {}) == candidates
    row = effect()
    retry = effect(attempt_id="retry")
    selected = rank_arms_by_evidence(
        candidates, {}, search_identity=identity(), search_effects=[row]
    )
    assert selected[0] == "useful"
    assert (
        rank_arms_by_evidence(
            candidates, {}, search_identity=identity(), search_effects=[row, retry]
        )
        == selected
    )
    assert compatible_effects([row, retry], identity())[1] == ["duplicate_retry"]


@pytest.mark.parametrize(
    "change",
    [
        {"required_case_ids": ("case", "case")},
        {"paired_root_ids": ("root", "other")},
        {"control": float("nan")},
        {"candidate": float("inf")},
        {"control": -1e308, "candidate": 1e308},
    ],
)
def test_invalid_effects_refused(change):
    with pytest.raises(ValueError):
        effect(**change)


def test_missing_hard_case_and_relabelled_replicate_are_not_evidence():
    with pytest.raises(ValueError, match="locked pairs"):
        effect(required_case_ids=("case", "hard"))
    with pytest.raises(ValueError, match="relabelled"):
        compatible_effects([effect(), effect(replicate_id="renamed")], identity())
    changed = identity().model_copy(update={"endpoint_version": "v2"})
    assert compatible_effects([effect(identity=changed)], identity())[1] == [
        "incompatible_identity"
    ]


def plan(**changes):
    values = dict(
        algorithm="successive_halving",
        treatment_ids=("a" * 64, "b" * 64, "c" * 64),
        checkpoints=(1, 3),
        total_updates=9,
        random_seed=7,
        direction="minimize",
        endpoint_identity="d" * 64,
    )
    return AllocationPlan(**(values | changes))


def observations(design):
    return [
        FidelityObservation(
            plan_digest=contract_digest(design),
            treatment_id=key,
            updates=step,
            value=value,
            cursor_digest="f" * 64,
            attempt_id=f"{key[0]}-{step}",
            endpoint_identity=design.endpoint_identity,
        )
        for key, values in zip(
            design.treatment_ids, ((1.0, 0.9), (2.0, 0.8), (3.0, 0.1)), strict=True
        )
        for step, value in zip(design.checkpoints, values, strict=True)
    ]


def test_default_off_and_unknown_fidelity_never_prunes_slow_starter():
    assert next_allocations(plan()) == []
    design = plan(enabled=True)
    rows = observations(design)
    allocations = next_allocations(design, [row for row in rows if row.updates == 1])
    assert {a.treatment_id for a in allocations} == set(design.treatment_ids)
    report = fidelity_report(design, rows)
    assert report["pairwise_rank_agreement"] == 0
    assert report["selection_regret"] == pytest.approx(0.8)
    assert report["consumed_updates"] == 9
    assert report["pruning_authorized"] is False


def test_allocation_enforces_budget_and_continuation_cursor():
    design = plan(enabled=True, total_updates=3)
    rows = observations(design)
    assert next_allocations(design, [row for row in rows if row.updates == 1]) == []
    with pytest.raises(ValueError, match="budget"):
        checked_observations(design, rows)
    with pytest.raises(ValueError, match="checkpoint"):
        checked_observations(plan(enabled=True), [observations(plan(enabled=True))[1]])


@pytest.mark.parametrize("magnitude", [1e-300, 1e308])
@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_fidelity_ranking_uses_order_not_float_products(magnitude, direction):
    design = plan(enabled=True, direction=direction, treatment_ids=("a" * 64, "b" * 64))
    rows = [
        FidelityObservation(
            plan_digest=contract_digest(design),
            treatment_id=key,
            updates=update,
            value=value,
            cursor_digest="f" * 64,
            attempt_id=f"{key[0]}-{update}",
            endpoint_identity=design.endpoint_identity,
        )
        for key, value in zip(
            design.treatment_ids, (-magnitude, magnitude), strict=True
        )
        for update in design.checkpoints
    ]
    report = fidelity_report(design, rows)
    assert report["pairwise_rank_agreement"] == 1
    assert report["tied_config_pairs"] == 0
    assert report["selection_regret"] == 0
    # A genuine tie remains a tie even when the other checkpoint has huge gaps.
    tied = [
        row.model_copy(update={"value": 0.0}) if row.updates == 3 else row
        for row in rows
    ]
    assert fidelity_report(design, tied)["pairwise_rank_agreement"] is None


def test_stagnation_receipts_cannot_reset_measurement_fault():
    kwargs = dict(family_identity="a" * 64, evidence_identity="b" * 64)
    faults = [SearchSignal(cause="missing_measurement", **kwargs)]
    noise = [SearchSignal(cause=kind, **kwargs) for kind in ("retry", "receipt")]
    actions = stagnation_actions(faults + noise)
    assert len(actions) == 1 and actions[0].action == "repair_harness"
    result = SearchSignal(cause="complete_comparison", comparison_id="c" * 64, **kwargs)
    assert stagnation_actions(faults + noise + [result]) == []
    assert stagnation_actions(faults + [result] + faults + [result]) == actions


def test_null_without_authoritative_closure_does_not_retire():
    signal = SearchSignal(
        family_identity="a" * 64, evidence_identity="b" * 64, cause="powered_null"
    )
    assert stagnation_actions([signal]) == []


def loss_fixture():
    selected = {
        "selected_record_ids": ["case"],
        "selected_root_ids": ["root"],
        "input_sha256s": ["d" * 64],
        "selection_sha256": "a" * 64,
    }
    return {
        "selection": selected,
        "estimator_id": identity().estimator,
        "per_record": [
            {
                "id": "case",
                "root_id": "root",
                "input_sha256": "d" * 64,
                "units": identity().units,
                "estimator_id": identity().estimator,
                "selection_sha256": "a" * 64,
                "seed": 7,
                "evaluator_sha256": "e" * 64,
                "nll": 2.0,
                "nll_sum": 8.0,
                "masked_tokens": 4,
            }
        ],
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("nll", True),
        ("nll", float("nan")),
        ("masked_tokens", 0),
        ("masked_tokens", True),
        ("nll_sum", 9.0),
        ("seed", 8),
        ("evaluator_sha256", "f" * 64),
    ],
)
def test_loss_producer_boundary_refuses_corrupt_pairs(key, value):
    control = loss_fixture()
    candidate = deepcopy(control)
    lineage = effect().model_dump(
        include={"slug", "treatment_id", "replicate_id", "comparison_id", "attempt_id"}
    )
    assert (
        effect_from_loss_reports(identity(), lineage, control, candidate).benefit == 0
    )
    candidate["per_record"][0][key] = value
    with pytest.raises(ValueError):
        effect_from_loss_reports(identity(), lineage, control, candidate)


def test_real_loss_producer_reaches_compatible_credit(tmp_path):
    from slm_training.dsl.schema import write_jsonl
    from slm_training.evals.denoising_nll import DenoisingNLLConfig
    from slm_training.evals.loss_suites import evaluate_loss_suites
    from slm_training.harnesses.train_data.learner_corrections import (
        copy_fixture_records,
    )
    from tests.test_models.test_step_commit_histogram import _tiny_model

    _, records = copy_fixture_records()
    suite = tmp_path / "suites" / "smoke"
    suite.mkdir(parents=True)
    write_jsonl(suite / "records.jsonl", records)
    report = evaluate_loss_suites(
        _tiny_model(),
        tmp_path,
        base_suite="smoke",
        limit=2,
        nll_config=DenoisingNLLConfig(mask_seed=7, compute_legal_support=False),
    )
    resolved = identity().model_copy(
        update={
            "selection_digest": report["selection"]["selection_sha256"],
            "estimator": report["estimator_id"],
        }
    )
    lineage = effect().model_dump(
        include={"slug", "treatment_id", "replicate_id", "comparison_id", "attempt_id"}
    )
    measured = effect_from_loss_reports(resolved, lineage, report, report)
    assert measured.paired_case_ids == ("copy-2", "copy-3")
    assert measured.benefit == 0  # same-model null, not a treatment improvement
    accepted, rejected = compatible_effects([measured], resolved)
    assert accepted == [measured] and not rejected


@pytest.mark.parametrize("fault", ["missing", "duplicate", "extra", "null"])
def test_auxiliary_rows_cannot_hide_invalid_locked_broad_coverage(fault):
    control = loss_fixture()
    candidate = deepcopy(control)
    candidate["per_record"].append({"id": "ood-only", "nll": None})
    lineage = effect().model_dump(
        include={"slug", "treatment_id", "replicate_id", "comparison_id", "attempt_id"}
    )
    assert (
        effect_from_loss_reports(identity(), lineage, control, candidate).benefit == 0
    )
    if fault == "missing":
        candidate["per_record"].pop(0)
    elif fault == "duplicate":
        candidate["per_record"].append(deepcopy(candidate["per_record"][0]))
    elif fault == "extra":
        candidate["per_record"].append({**candidate["per_record"][0], "id": "extra"})
    else:
        candidate["per_record"][0]["nll"] = None
    with pytest.raises(ValueError):
        effect_from_loss_reports(identity(), lineage, control, candidate)
