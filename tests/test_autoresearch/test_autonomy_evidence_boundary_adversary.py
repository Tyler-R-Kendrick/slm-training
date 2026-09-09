"""Independent evidence-boundary falsifiers; synthetic inputs are not model wins.

These tests exercise canonical owners. Rejection assertions must not become
xfails or assertions accepting the malformed evidence they are meant to exclude.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from slm_training.evals.measurement_identity import loss_record_rows
from slm_training.harnesses.model_build.eval_measurement import suite_result_cacheable


def _complete_metrics():
    return dict(
        n=1,
        document_n=1,
        completed_document_n=1,
        incomplete_document_n=0,
        decode_timeout_count=0,
        measurement_complete=True,
        selected_record_ids=["case"],
        parse_rate=1.0,
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_complete_counts_do_not_make_invalid_quality_cacheable(value):
    metrics = _complete_metrics()
    assert suite_result_cacheable(metrics), "positive control must be cacheable"
    metrics["parse_rate"] = value
    assert not suite_result_cacheable(metrics)


@pytest.mark.parametrize("second_mean", [1.0, 2.0])
def test_loss_flattening_cannot_hide_duplicate_selected_observations(second_mean):
    row = dict(id="case", mean_nll=1.0, masked_tokens=1, nll_sum=1.0)
    single = loss_record_rows({"broad": {"per_record": [row]}})
    assert len(single) == 1 and single[0]["nll"] == 1.0
    repeated = {**row, "mean_nll": second_mean, "nll_sum": second_mean}
    with pytest.raises(ValueError):
        loss_record_rows({"broad": {"per_record": [row, repeated]}})


def test_loss_producer_auxiliary_ood_rows_do_not_cancel_complete_broad_comparison():
    from slm_training.autoresearch.search.evidence import effect_from_loss_reports
    from slm_training.evals.measurement_identity import row_identity, selected_identity
    from slm_training.harnesses.train_data.learner_corrections import (
        copy_fixture_records,
    )
    from tests.test_autoresearch.test_search_contracts import effect, identity

    _, records = copy_fixture_records()
    selection = selected_identity(records[:1])
    resolved = identity().model_copy(
        update={"selection_digest": selection["selection_sha256"]}
    )
    categories = {}
    for category, record in zip(("broad", "schema_ood"), records, strict=True):
        row = row_identity(
            record,
            selection_sha256=selection["selection_sha256"],
            seed=7,
            estimator_id=resolved.estimator,
            evaluator_sha256="f" * 64,
        )
        row.update(
            id=record.id,
            mean_nll=1.0,
            nll_sum=2.0,
            masked_tokens=2,
            units=resolved.units,
        )
        categories[category] = {"per_record": [row]}
    report = dict(
        selection=selection,
        estimator_id=resolved.estimator,
        categories=categories,
        per_record=loss_record_rows(categories),
    )
    assert len(report["per_record"]) == 2
    lineage = effect().model_dump(
        include={"slug", "treatment_id", "replicate_id", "comparison_id", "attempt_id"}
    )
    comparison = effect_from_loss_reports(resolved, lineage, report, report)
    assert comparison.complete
    assert comparison.paired_case_ids == tuple(selection["selected_record_ids"])
    assert comparison.benefit == 0  # Same-report diagnostic, not improvement.


def test_preflight_cache_rejects_scorer_successor_without_waiting_for_version_bump(
    tmp_path, monkeypatch
):
    from slm_training.harness_core.eval_cache import (
        EvalCache,
        EvalCacheConfig,
        EvalCacheMode,
    )
    from slm_training.harnesses.model_build import ModelBuildConfig, eval_runner

    config = ModelBuildConfig(train_dir=tmp_path, test_dir=tmp_path)
    cache = EvalCache(
        EvalCacheConfig(mode=EvalCacheMode.READ_WRITE, root=tmp_path / "cache")
    )
    monkeypatch.setattr(eval_runner, "_eval_data_sha", lambda _: "a" * 64)
    monkeypatch.setattr(eval_runner, "evaluator_identity", lambda _: "b" * 64)

    def preflight():
        return eval_runner._suite_cache_preflight(
            config,
            checkpoint=tmp_path / "unloaded.pt",
            cache=cache,
            suite_limit=1,
            suite_offset=0,
            generation_overrides={},
            checkpoint_sha256="c" * 64,
            checkpoint_bundle_sha256="d" * 64,
        )

    original = preflight()
    assert original.key is not None
    cache.put(original.key, _complete_metrics(), dependencies=original.dependencies)
    assert preflight().cached_metrics is not None
    # Only the shared live scorer identity changes; weights/data/config stay fixed.
    monkeypatch.setattr(eval_runner, "evaluator_identity", lambda _: "e" * 64)
    assert preflight().cached_metrics is None


def test_bundle_byte_counts_are_integers_not_boolean_hash_consistency(tmp_path):
    from slm_training.harness_core.checkpoint_bundle import (
        stage_checkpoint_bundle,
        validate_bundle,
    )
    from tests.test_harnesses.model_build.test_checkpoint_bundle import _checkpoint

    root = tmp_path / "bundles-owner"
    checkpoint = _checkpoint(tmp_path / "source", marker=b"x")
    digest = stage_checkpoint_bundle(root, checkpoint, {})
    directory, manifest = validate_bundle(root, digest)
    assert manifest["files"]["last.pt"]["bytes"] == 1
    manifest["files"]["last.pt"]["bytes"] = True
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    successor_digest = hashlib.sha256(raw).hexdigest()
    successor = directory.parent / successor_digest
    successor.mkdir()
    for component in manifest["files"]:
        (successor / component).write_bytes((directory / component).read_bytes())
    (successor / "manifest.json").write_bytes(raw)
    with pytest.raises(ValueError):
        validate_bundle(root, successor_digest)
    # An invalid proposed successor must not invalidate the original bundle.
    assert validate_bundle(root, digest)[0] == directory


def test_revoked_runtime_lease_cannot_republish_identical_valid_bundle(tmp_path):
    from slm_training.autoresearch.runtime.activity_runtime import (
        ActivityRuntime,
        StaleLease,
    )
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harness_core.activity_contract import ActivitySpec, ResourceGrant
    from slm_training.harness_core.checkpoint_bundle import (
        current_bundle_digest,
        publish_activity_bundle,
        stage_checkpoint_bundle,
    )
    from tests.test_harnesses.model_build.test_checkpoint_bundle import _checkpoint

    root = tmp_path / "published"
    digest = stage_checkpoint_bundle(root, _checkpoint(tmp_path / "source"), {})
    with ActivityRuntime(CampaignStore("adversarial-publication", tmp_path)) as runtime:
        runtime.register(
            ActivitySpec(
                activity_id="publisher",
                family="fixture",
                kind="control",
                source_digest="a" * 40,
                environment_digest="b" * 64,
                input_digest="c" * 64,
                output_namespace="attempts",
                grant=ResourceGrant(
                    interrupt_seconds=10, kill_grace_seconds=0.1, total_seconds=12
                ),
            )
        )
        lease = runtime.claim_next(capabilities={"local_process"})
        assert lease is not None
        publish_activity_bundle(runtime, lease, root, digest, None)
        pointer = (root / "current.json").read_bytes()
        runtime.cancel("publisher", reason="explicit fixture cancellation")
        with pytest.raises(StaleLease):
            publish_activity_bundle(runtime, lease, root, digest, digest)
        assert (root / "current.json").read_bytes() == pointer
        assert current_bundle_digest(root) == digest


def test_fidelity_finite_inputs_cannot_emit_nonfinite_regret():
    from slm_training.autoresearch.search.allocation import (
        AllocationPlan,
        FidelityObservation,
        fidelity_report,
    )
    from slm_training.autoresearch.search.evidence import contract_digest

    plan = AllocationPlan(
        algorithm="rotation",
        enabled=True,
        treatment_ids=("a" * 64, "b" * 64),
        checkpoints=(1, 2),
        total_updates=4,
        random_seed=7,
        direction="maximize",
        endpoint_identity="c" * 64,
    )
    rows = [
        FidelityObservation(
            plan_digest=contract_digest(plan),
            treatment_id=treatment,
            updates=updates,
            value=value,
            cursor_digest="d" * 64,
            attempt_id=f"{treatment[0]}-{updates}",
            endpoint_identity=plan.endpoint_identity,
        )
        for treatment, values in zip(
            plan.treatment_ids, ((1.0, -1e308), (0.0, 1e308)), strict=True
        )
        for updates, value in zip(plan.checkpoints, values, strict=True)
    ]
    # A typed refusal is valid; a report containing Infinity is not evidence.
    try:
        report = fidelity_report(plan, rows)
    except ValueError:
        return
    json.dumps(report, allow_nan=False)
