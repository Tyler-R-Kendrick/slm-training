"""Real compiled preflight and loss owners at the driver/search integration seam."""

import json
from types import SimpleNamespace

import pytest

from scripts import autotrain_search as search
from scripts import run_autotrain_continuous as driver
from scripts.autotrain_nll import run_arm_eval_nll
from slm_training.autoresearch.climb_policy import load_climb_policy
from slm_training.autoresearch.preflight.compiled_treatment import begin_attempt
from slm_training.autoresearch.schemas import ExperimentSpec
from slm_training.autoresearch.search.evidence import SearchEffect, record_search_effect
from tests.test_autoresearch.test_search_preflight_ingestion import (
    _lock,
    production_inputs as production_inputs,
    arms as arms,
    train_dir as train_dir,
)


def test_paired_sd_default_is_campaign_evidence_not_source_calibration(tmp_path):
    from scripts.autotrain_records import record_screening_paired_sd
    from slm_training.autoresearch.storage import CampaignStore

    protected = driver.screening_expectations_path()
    original = protected.read_bytes()
    kwargs = dict(metric_leaf="eval_nll", sd=0.05, n=6, campaign_id="camp",
                  control_id="control", candidate_id="candidate")
    campaign_dir = tmp_path / "camp"
    assert record_screening_paired_sd(None, campaign_dir=campaign_dir, **kwargs)
    assert record_screening_paired_sd(None, campaign_dir=campaign_dir, **kwargs)
    events = CampaignStore("camp", tmp_path).verify_event_chain()
    assert len(events) == 1 and events[0]["event_type"] == "paired_sd_observed"
    with pytest.raises(PermissionError, match="cannot modify source"):
        record_screening_paired_sd(protected, campaign_dir=campaign_dir, **kwargs)
    assert protected.read_bytes() == original


def _specs(inputs):
    return [
        ExperimentSpec.model_validate_json(path.read_text())
        for path in inputs[1].values()
    ]


def test_current_compiled_identity_selects_before_any_campaign_lock(
    production_inputs, monkeypatch
):
    store = production_inputs[0]
    control, candidate = _specs(production_inputs)
    other = candidate.model_copy(
        update={
            "experiment_id": "other",
            "knobs": candidate.knobs.model_copy(update={"lr": 0.002}),
        }
    )
    owner = SimpleNamespace(
        _manifest=driver._manifest,
        _slug_from_candidate_id=lambda eid: eid,
        _evidence_ranked_slug=driver._evidence_ranked_slug,
    )
    policy = load_climb_policy()
    context = {
        "integration": "e0eca9f9910244ecc20eb480d363f1852417b599",
        "policy": policy,
    }
    _, identity, _, _ = search._prospective(store, control, candidate, owner, context)
    # A synthetic historical loss canary tests routing, not neural improvement.
    record_search_effect(
        store,
        SearchEffect(
            identity=identity,
            slug="other",
            treatment_id="a" * 64,
            replicate_id="planned-canary",
            comparison_id="b" * 64,
            attempt_id="canary-attempt",
            complete=True,
            control=10.0,
            candidate=1.0,
            paired_case_ids=("case",),
            paired_root_ids=("root",),
            required_case_ids=("case",),
        ),
    )
    monkeypatch.setattr(
        search, "loop_campaigns", lambda *_args, **_kwargs: [store.load_campaign()]
    )
    matrix = {
        "campaign_id": store.campaign_id,
        "recommended_experiment_id": candidate.experiment_id,
        "hypotheses": [
            {"experiment": s.model_dump(mode="json")}
            for s in (control, candidate, other)
        ],
    }
    chosen = search.choose_matrix(
        matrix, owner, root=store.root.parent, loop_id="fixture", **context
    )
    assert chosen["recommended_experiment_id"] == "other"
    assert matrix["recommended_experiment_id"] == "candidate"
    assert chosen["hypotheses"] == matrix["hypotheses"]
    events = store.verify_event_chain()
    assert not any(
        e["event_type"] in {"experiment_campaign_locked", "experiment_attempt_started"}
        for e in events
    )
    assert sum(e["event_type"] == "search_selection_observed" for e in events) == 1


def test_saved_real_loss_reaches_credit_once_and_corruption_is_refused(
    production_inputs,
):
    store, _, _, eval_root, model = production_inputs
    pair, _ = _lock(production_inputs)
    for eid in pair["arm_ids"]:
        attempt = begin_attempt(store, pair, eid)
        store.append_event(
            "experiment_attempt_returned",
            experiment_id=eid,
            detail={**attempt, "exit_code": 0},
        )
        run_arm_eval_nll(
            store.root / "runs" / eid, {"test_dir": eval_root, "model": model}
        )
    journal = SimpleNamespace(
        store=store, value={"locked_designs": {"candidate": pair}}
    )
    search.record_cycle_credit(journal)
    search.record_cycle_credit(journal)
    events = store.verify_event_chain()
    assert sum(e["event_type"] == "search_effect_recorded" for e in events) == 1
    assert not any(e["event_type"] == "search_ingestion_refused" for e in events)
    path = store.root / "runs/candidate/eval_nll_records.json"
    saved = json.loads(path.read_text())
    saved["row_evidence"][0]["nll"] = 123.0
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="unverified_diagnostic"):
        search._loss_report(store, "candidate")
    search.record_cycle_credit(journal)
    events = store.verify_event_chain()
    assert sum(e["event_type"] == "search_effect_recorded" for e in events) == 1
    assert sum(e["event_type"] == "search_ingestion_refused" for e in events) == 1


def test_driver_forwards_strong_context_to_existing_selector(monkeypatch):
    from slm_training.autoresearch import evidence_ledger

    seen = {}

    def pick(candidates, ledger, **kwargs):
        seen.update(kwargs)
        return candidates[0]

    monkeypatch.setattr(evidence_ledger, "pick_evidence_ranked_slug", pick)
    identity, effects = object(), [object()]
    assert (
        driver._evidence_ranked_slug(
            ["fixture"],
            stats={},
            boosts={},
            search_identity=identity,
            search_effects=effects,
        )
        == "fixture"
    )
    assert seen["search_identity"] is identity and seen["search_effects"] is effects


@pytest.mark.parametrize("exit_code", [10, 124, 2, True, None])
def test_latest_incomplete_attempt_cannot_credit_old_valid_rows(
    production_inputs, exit_code
):
    store, _, _, eval_root, model = production_inputs
    pair, _ = _lock(production_inputs)
    for eid in pair["arm_ids"]:
        attempt = begin_attempt(store, pair, eid)
        store.append_event(
            "experiment_attempt_returned",
            experiment_id=eid,
            detail={**attempt, "exit_code": 0},
        )
        run_arm_eval_nll(
            store.root / "runs" / eid, {"test_dir": eval_root, "model": model}
        )
    attempt = begin_attempt(store, pair, "candidate")
    store.append_event(
        "experiment_attempt_returned",
        experiment_id="candidate",
        detail={**attempt, "exit_code": exit_code},
    )
    search.record_cycle_credit(
        SimpleNamespace(store=store, value={"locked_designs": {"candidate": pair}})
    )
    events = store.verify_event_chain()
    assert not any(e["event_type"] == "search_effect_recorded" for e in events)
    assert any(e["event_type"] == "search_ingestion_refused" for e in events)
