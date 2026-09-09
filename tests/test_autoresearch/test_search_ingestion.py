"""Real loss producer and CampaignStore; locked inputs are public contract fixtures.

The same untrained model supplies both arms: this proves ingestion of a null,
not a trained treatment improvement or independent scientific confirmation.
"""

from copy import deepcopy
from dataclasses import asdict
from functools import partial
import hashlib
import json
import platform

import pytest
import torch

from slm_training.autoresearch.preflight.compiled_treatment import (
    begin_attempt,
    persist_pair,
)
from slm_training.autoresearch.search.evidence import (
    contract_digest,
    compatible_effects,
)
from slm_training.autoresearch.search.ingestion import ingest_loss_reports as ingest
from slm_training.autoresearch.storage import CampaignStore
from slm_training.dsl.schema import write_jsonl
from slm_training.evals import denoising_nll
from slm_training.evals.denoising_nll import DenoisingNLLConfig
from slm_training.evals.loss_suites import evaluate_loss_suites
from slm_training.evals.measurement_identity import content_digest, selected_identity
from slm_training.harnesses.train_data.learner_corrections import copy_fixture_records
from slm_training.harness_core.lineage.records import content_sha
from tests.test_autoresearch.test_experiment_campaign import _manifest
from tests.test_models.test_step_commit_histogram import _tiny_model

ingest_loss_reports = partial(ingest, artifact_digest=content_sha)


@pytest.fixture
def inputs(tmp_path):
    model = _tiny_model()
    _, records = copy_fixture_records()
    suite = tmp_path / "suites" / "smoke"
    suite.mkdir(parents=True)
    write_jsonl(suite / "records.jsonl", records)
    checkpoint = tmp_path / "ancestor.pt"
    torch.save(model.state_dict(), checkpoint)
    ancestor = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cfg = DenoisingNLLConfig(mask_seed=7, compute_legal_support=False)
    # The producer currently embeds this identity expression inside execution.
    # Parent must expose it for preflight; no outcome is inspected to lock it.
    from pathlib import Path

    scorer = hashlib.sha256(Path(denoising_nll.__file__).read_bytes()).hexdigest()
    estimator = "conditional_masked_token_ce/" + content_digest(
        {
            **cfg.key(),
            "evaluator_sha256": scorer,
            "position_filter": "all_eligible",
        }
    )
    config_digest = contract_digest(asdict(model.config))
    manifest = _manifest(
        claim_class="wiring",
        arms=[
            {"arm_id": role, "role": role, "config_sha256": config_digest}
            for role in ("control", "candidate")
        ],
        endpoints=[
            {
                "endpoint_id": "meaning",
                "metric": "smoke.eval_nll",
                "direction": "decrease",
                "role": "primary",
                "minimum_effect": 0.01,
            }
        ],
        locked_eval_manifest_sha256=hashlib.sha256(
            (suite / "records.jsonl").read_bytes()
        ).hexdigest(),
    )
    store = CampaignStore(manifest.campaign_id, tmp_path / "campaigns")
    store.lock_experiment_campaign(manifest)
    source = {"denoising_nll.py": scorer}
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "machine": platform.machine(),
        "device": "cpu",
        "threads": torch.get_num_threads(),
    }
    pair = {
        "schema": "compiled_treatment_design/v1",
        "hypothesis_id": contract_digest(manifest.hypothesis),
        "design_digest": contract_digest(
            {"config": config_digest, "ancestor": ancestor}
        ),
        "replicate_id": contract_digest({"ancestor": ancestor, "seed": 7}),
        "randomness": {"seed": 7, "loss_mask_seed": cfg.mask_seed},
        "independence": "conditional_on_shared_ancestor",
        "arm_ids": ["control", "candidate"],
        "arm_config_sha256s": [config_digest, config_digest],
        "treatment_ids": [config_digest, config_digest],
        "search_slug": "public-same-model-null",
        "source": [source, source],
        "runtime": [runtime, runtime],
        "design": {
            "intervention": {"kind": "mechanism"},
            "bindings": {
                "architecture": contract_digest(
                    {k: list(v.shape) for k, v in model.state_dict().items()}
                ),
                "tokenizer_layout": contract_digest(model.tokenizer.token_to_id),
                "training_snapshot": contract_digest(
                    {"fixture": "untrained", "model_config": config_digest}
                ),
                "starting_checkpoint": ancestor,
                "starting_checkpoint_role": "warm_start",
                "resource_contract": {"basis": "updates", "planned_totals": [0, 0]},
                "endpoint": {
                    "loss_measurement": {
                        "endpoint_id": "meaning",
                        "version": denoising_nll.LOSS_SUITE_VERSION,
                        "units": "nats_per_masked_token",
                        "estimator_id": estimator,
                        "evaluator_sha256": scorer,
                        "selection": selected_identity(records),
                    }
                },
            },
        },
    }
    return store, manifest, pair, model, cfg, tmp_path


def _execute(inputs):
    store, manifest, pair, model, cfg, root = inputs
    path = persist_pair(store, pair)
    attempts = [begin_attempt(store, pair, arm) for arm in pair["arm_ids"]]
    report = evaluate_loss_suites(
        model, root, base_suite="smoke", limit=2, nll_config=cfg
    )
    lineage = {
        "design_sha256": path.stem,
        "control_attempt_id": attempts[0]["attempt_id"],
        "candidate_attempt_id": attempts[1]["attempt_id"],
    }
    return store, manifest, lineage, report


def test_real_loss_report_is_recorded_once_as_diagnostic_null(inputs):
    store, manifest, lineage, report = _execute(inputs)
    effect = ingest_loss_reports(
        store, manifest, lineage=lineage, control=report, candidate=report
    )
    assert effect.complete and effect.benefit == 0
    assert effect.paired_case_ids == ("copy-2", "copy-3")
    assert (
        ingest_loss_reports(
            store, manifest, lineage=lineage, control=report, candidate=report
        )
        == effect
    )
    events = [
        e
        for e in store.verify_event_chain()
        if e["event_type"] == "search_effect_recorded"
    ]
    assert len(events) == 1
    assert events[0]["detail"]["claim_class"] == "diagnostic_only"
    saved = json.loads(
        (
            store.root
            / "artifacts/search_effects"
            / f"{events[0]['artifact_sha256']}.json"
        ).read_text()
    )
    assert compatible_effects([saved], effect.identity)[0] == [effect]


@pytest.mark.parametrize(
    "missing", ["loss_measurement", "units", "estimator_id", "selection"]
)
def test_legacy_lock_does_not_acquire_guessed_identity(inputs, missing):
    endpoint = inputs[2]["design"]["bindings"]["endpoint"]
    if missing == "loss_measurement":
        endpoint.pop(missing)
    else:
        endpoint["loss_measurement"].pop(missing)
    store, manifest, lineage, report = _execute(inputs)
    with pytest.raises(ValueError, match=f"missing_locked_input:{missing}"):
        ingest_loss_reports(
            store, manifest, lineage=lineage, control=report, candidate=report
        )
    assert not (store.root / "artifacts/search_effects").exists()


@pytest.mark.parametrize("field", ["seed", "evaluator_sha256", "input_sha256"])
def test_same_wrong_identity_in_both_reports_is_not_accepted(inputs, field):
    store, manifest, lineage, report = _execute(inputs)
    invalid = deepcopy(report)
    for row in invalid["per_record"]:
        row[field] = 8 if field == "seed" else "f" * 64
    with pytest.raises(ValueError):
        ingest_loss_reports(
            store, manifest, lineage=lineage, control=invalid, candidate=invalid
        )
    assert not (store.root / "artifacts/search_effects").exists()


def test_wrong_campaign_and_attempt_are_refused(inputs):
    store, manifest, lineage, report = _execute(inputs)
    wrong = manifest.model_copy(
        update={"hypothesis": "Changed after the outcomes were observed."}
    )
    with pytest.raises(ValueError, match="campaign_lock_mismatch"):
        ingest_loss_reports(
            store, wrong, lineage=lineage, control=report, candidate=report
        )
    lineage["candidate_attempt_id"] = lineage["control_attempt_id"]
    with pytest.raises(ValueError, match="attempt_binding_mismatch"):
        ingest_loss_reports(
            store, manifest, lineage=lineage, control=report, candidate=report
        )
    assert not (store.root / "artifacts/search_effects").exists()


def test_changed_locked_artifact_cannot_record_credit(inputs):
    store, manifest, lineage, report = _execute(inputs)
    path = (
        store.root / "artifacts/treatment_designs" / f"{lineage['design_sha256']}.json"
    )
    changed = json.loads(path.read_text())
    changed["search_slug"] = "renamed"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="artifact_changed"):
        ingest_loss_reports(
            store, manifest, lineage=lineage, control=report, candidate=report
        )


def test_conflicting_retry_is_refused_before_recording_effect(inputs):
    store, manifest, lineage, report = _execute(inputs)
    original = ingest_loss_reports(
        store, manifest, lineage=lineage, control=report, candidate=report
    )
    retry = begin_attempt(store, inputs[2], "candidate")
    lineage["candidate_attempt_id"] = retry["attempt_id"]
    repeated = ingest_loss_reports(
        store, manifest, lineage=lineage, control=report, candidate=report
    )
    assert original.comparison_id == repeated.comparison_id
    assert original.attempt_id != repeated.attempt_id
    assert compatible_effects([original, repeated], original.identity)[1] == [
        "duplicate_retry"
    ]
    bad = deepcopy(report)
    for row in bad["per_record"]:
        row["nll"] += 1
        row["nll_sum"] += row["masked_tokens"]
    with pytest.raises(ValueError, match="conflicting retry"):
        ingest_loss_reports(
            store, manifest, lineage=lineage, control=report, candidate=bad
        )
    assert (
        len(
            [
                e
                for e in store.verify_event_chain()
                if e["event_type"] == "search_effect_recorded"
            ]
        )
        == 2
    )


def test_search_stays_outside_the_orchestration_dependency_cycle():
    from pathlib import Path
    from slm_training.quality import martin

    _, edges = martin.build(root=Path(__file__).resolve().parents[2])
    assert all(
        "slm_training.autoresearch.search" not in cycle
        for cycle in martin.cycles(edges)
    )
