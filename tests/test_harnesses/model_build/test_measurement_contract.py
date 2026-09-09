"""Regressions on the actual evaluation producers, not transcribed helpers."""

import json
from dataclasses import replace

import pytest

from scripts.autotrain_metrics import finite_metric, rate_to_pm
from scripts.autotrain_measurement import measurement_is_complete
from slm_training.dsl.schema import write_jsonl
from slm_training.harnesses.model_build import eval_runner
from tests.test_harnesses.model_build.test_eval_runner_resume import (
    SlowStubDecoder,
    _config,
    _identity,
    _record,
    clock,
    suite,
)

__all__ = ["clock", "suite"]


def test_cache_requires_positive_document_partition_and_exact_selected_units():
    from slm_training.harnesses.model_build.eval_measurement import suite_result_cacheable

    valid = {"n": 2, "document_n": 1, "fragment_n": 1, "completed_document_n": 1,
             "incomplete_document_n": 0, "decode_timeout_count": 0,
             "measurement_complete": True, "selected_record_ids": ["doc", "fragment"]}
    assert suite_result_cacheable(valid)
    invalid_changes = (
        {"document_n": 0, "completed_document_n": 0, "fragment_n": 2},
        {"fragment_n": 0}, {"fragment_n": True}, {"pending_document_n": 1},
        {"incomplete_document_n": 1}, {"completed_document_n": 2},
        {"selected_record_ids": ["doc"]}, {"selected_record_ids": ["doc", "doc"]},
        {"selected_record_ids": ["doc", 1]}, {"measurement_complete": None},
    )
    for changes in invalid_changes:
        assert not suite_result_cacheable({**valid, **changes}), changes
    legacy = {key: value for key, value in valid.items() if key != "selected_record_ids"}
    assert not suite_result_cacheable(legacy)


def test_nll_only_is_not_complete_decoded_measurement():
    assert not measurement_is_complete({
        "control_metrics": {"eval_nll": 1.2},
        "candidate_metrics": {"eval_nll": 1.1}, "reasons": [],
    })


def test_invalid_nll_row_invalidates_payload_instead_of_shrinking(tmp_path):
    from scripts.autotrain_metrics import read_eval_nll_records

    (tmp_path / "eval_nll_records.json").write_text(json.dumps({
        "schema": "eval_nll_records/v1", "definition_hash": "old",
        "records": {"easy": 1.0, "hard": float("nan")},
    }))
    assert read_eval_nll_records(tmp_path) == ({}, None)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_nonfinite_metrics_are_unavailable(value):
    assert finite_metric(value) is None
    assert rate_to_pm(value) is None


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_invalid_rate_is_not_clamped_to_a_valid_rate(value):
    assert rate_to_pm(value) is None


@pytest.mark.parametrize("payload", [{}, {"n": 0}, {"measurement_complete": False}])
def test_sentinel_scoreboard_cannot_be_cached(payload):
    assert not eval_runner._suite_result_cacheable(payload)


@pytest.mark.parametrize("limit", [0, -1])
def test_empty_selection_refuses_before_decode(clock, suite, tmp_path, limit):
    decoder = SlowStubDecoder(clock)
    with pytest.raises(ValueError, match="selection|eval_limit"):
        eval_runner.evaluate(
            _config(suite, tmp_path, "empty", eval_limit=limit),
            model=decoder, publish_agentv=False,
        )
    assert decoder.calls == 0


def test_duplicate_selected_ids_refuse_before_decode(clock, suite, tmp_path):
    write_jsonl(suite["test"] / "suites/smoke/records.jsonl", [_record(), _record()])
    decoder = SlowStubDecoder(clock)
    with pytest.raises(ValueError, match="duplicate"):
        eval_runner.evaluate(_config(suite, tmp_path, "duplicate"), model=decoder)
    assert decoder.calls == 0


def test_partial_single_suite_does_not_publish_agentv(clock, suite, tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("partial measurement reached authoritative publication")

    monkeypatch.setattr("slm_training.evals.agentv.publish_model_evaluation", refuse)
    result = eval_runner.evaluate(
        _config(suite, tmp_path, "partial", eval_limit=6),
        model=SlowStubDecoder(clock), max_records_this_run=2, **_identity(suite),
    )
    assert result["completed_document_n"] == 2
    assert result["measurement_complete"] is False


@pytest.mark.parametrize("change", ["seed", "tokenizer"])
def test_resume_rejects_changed_randomness_or_bundle(clock, suite, tmp_path, change):
    config = _config(suite, tmp_path, "resume", eval_limit=6)
    eval_runner.evaluate(config, model=SlowStubDecoder(clock), publish_agentv=False,
                         max_records_this_run=2, **_identity(suite))
    original = eval_runner.partial_scoreboard_path(config.run_dir, "smoke").read_bytes()
    if change == "seed":
        config = replace(config, seed=config.seed + 1)
    else:
        suite["checkpoint"].with_suffix(".tokenizer.json").write_text("{}")
    result = eval_runner.evaluate(
        config, model=SlowStubDecoder(clock), publish_agentv=False,
        max_records_this_run=2, **_identity(suite),
    )
    assert result["resume"]["replayed_record_n"] == 0
    assert result["resume"]["resume_rejected"]
    archived = config.run_dir / result["resume"]["rejected_evidence"]
    assert archived.read_bytes() == original


def test_sdk_failure_preserves_measured_counts(clock, suite, tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("sdk unavailable")

    monkeypatch.setattr("slm_training.evals.agentv.publish_model_evaluation", unavailable)
    config = _config(suite, tmp_path, "sdk", eval_limit=6)
    with pytest.raises(RuntimeError, match="sdk unavailable"):
        eval_runner.evaluate(config, model=SlowStubDecoder(clock))
    payload = json.loads((config.run_dir / "eval_smoke.json").read_text())
    assert payload["completed_document_n"] == 6
    assert payload["publication_complete"] is False


def test_scoring_failure_preserves_decoded_input(clock, suite, tmp_path, monkeypatch):
    from slm_training.evals import meaningful_program

    original = meaningful_program.binding_aware_meaningful_v2
    def unavailable(*args, **kwargs):
        raise RuntimeError("bridge unavailable")
    config = _config(suite, tmp_path, "score-repair", eval_limit=1)
    monkeypatch.setattr(meaningful_program, "binding_aware_meaningful_v2", unavailable)
    with pytest.raises(RuntimeError, match="bridge unavailable"):
        eval_runner.evaluate(config, model=SlowStubDecoder(clock), publish_agentv=False,
                             max_records_this_run=1, **_identity(suite))
    partial = eval_runner.load_partial_scoreboard(config.run_dir, "smoke")
    assert partial["records"]["smoke-00"]["prediction"]
    assert not partial["measurement_complete"]
    monkeypatch.setattr(meaningful_program, "binding_aware_meaningful_v2", original)
    decoder = SlowStubDecoder(clock)
    result = eval_runner.evaluate(config, model=decoder, publish_agentv=False,
                                  max_records_this_run=1, **_identity(suite))
    assert result["measurement_complete"]
    assert decoder.calls == 0



def test_scoring_failure_preserves_entire_already_decoded_batch(clock, suite, tmp_path, monkeypatch):
    from slm_training.evals import meaningful_program
    from tests.test_harnesses.model_build.test_eval_runner_resume import BatchedStubDecoder

    original = meaningful_program.binding_aware_meaningful_v2
    def unavailable(*args, **kwargs):
        raise RuntimeError("first score interrupted after whole batch decoded")
    config = _config(suite, tmp_path, "batch-score-repair", eval_limit=2,
                     generate_batch_size=2)
    decoder = BatchedStubDecoder(clock)
    monkeypatch.setattr(meaningful_program, "binding_aware_meaningful_v2", unavailable)
    with pytest.raises(RuntimeError, match="whole batch decoded"):
        eval_runner.evaluate(config, model=decoder, publish_agentv=False,
                             max_records_this_run=2, **_identity(suite))
    assert decoder.batches == [["smoke-00", "smoke-01"]]
    partial = eval_runner.load_partial_scoreboard(config.run_dir, "smoke")
    assert set(partial["records"]) == {"smoke-00", "smoke-01"}
    assert not partial["measurement_complete"]
    monkeypatch.setattr(meaningful_program, "binding_aware_meaningful_v2", original)
    resumed = BatchedStubDecoder(clock)
    result = eval_runner.evaluate(config, model=resumed, publish_agentv=False,
                                  max_records_this_run=2, **_identity(suite))
    assert resumed.batches == []
    assert result["measurement_complete"]
    assert result["resume"]["replayed_record_n"] == 2


def test_suite_cache_rejects_incomplete_measurements():
    assert eval_runner._suite_result_cacheable(
        {"n": 1, "document_n": 1, "completed_document_n": 1,
         "decode_timeout_count": 0, "incomplete_document_n": 0,
         "measurement_complete": True, "selected_record_ids": ["document-1"]}
    )
    assert not eval_runner._suite_result_cacheable(
        {"decode_timeout_count": 1, "incomplete_document_n": 1}
    )


def test_active_suite_writer_refuses_second_writer(clock, suite, tmp_path):
    import fcntl
    config = _config(suite, tmp_path, "writer", eval_limit=1)
    config.run_dir.mkdir(parents=True)
    decoder = SlowStubDecoder(clock)
    with (config.run_dir / ".eval_smoke.lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="writer already active"):
            eval_runner.evaluate(config, model=decoder, publish_agentv=False)
    assert decoder.calls == 0


def test_record_randomness_does_not_restart_after_resume(clock, suite, tmp_path):
    import random
    class RandomDecoder(SlowStubDecoder):
        def generate_with_stats(self, prompt, **kwargs):
            self.samples.append(random.random())
            return super().generate_with_stats(prompt, **kwargs)
    direct = RandomDecoder(clock)
    direct.samples = []
    config = _config(suite, tmp_path, "random-direct", eval_limit=4)
    eval_runner.evaluate(config, model=direct, publish_agentv=False)
    resumed = RandomDecoder(clock)
    resumed.samples = []
    config = replace(config, run_id="random-chunks")
    for _ in range(2):
        eval_runner.evaluate(config, model=resumed, publish_agentv=False,
                             max_records_this_run=2, **_identity(suite))
    assert direct.samples == resumed.samples


def _complete_cache_counts(**changes):
    return dict(n=1, document_n=1, completed_document_n=1, incomplete_document_n=0,
                decode_timeout_count=0, measurement_complete=True,
                selected_record_ids=["document"], parse_rate=1.0) | changes


@pytest.mark.parametrize("key,value", [
    ("document_n", None), ("completed_latency_n", True), ("fallback_count", 1.5),
    ("latency_ms_p95", -1.0), ("latency_ms_p50", "10"), ("parse_rate", None),
    ("binder_reference_f1", 1.1), ("contract_recall", True),
    ("placeholder_fidelity", "1.0"), ("parse_rate_ci95", [0.1, 1.1]),
    ("parse_rate_ci95", [0.9, 0.1]), ("decode_stats", {"mean": float("inf")}),
    ("metric_defined_n", {"placeholder_validity": 1}),
])
def test_cache_rejects_invalid_numeric_leaves_and_denominators(key, value):
    assert not eval_runner._suite_result_cacheable(_complete_cache_counts(**{key: value}))


def test_cache_preserves_undefined_optional_metrics_and_nonbinomial_telemetry():
    metrics = _complete_cache_counts(
        placeholder_validity=None, metric_defined_n={"placeholder_validity": 0},
        constrained_fallback_rate=2.0,
    )
    assert eval_runner._suite_result_cacheable(metrics)


@pytest.mark.parametrize("category", ["broad", "binding", "structural", "schema_ood"])
def test_loss_flattening_rejects_duplicates_within_each_category(category):
    from slm_training.evals.loss_suites import per_record_nll_rows

    row = dict(id="case", mean_nll=1.0, masked_tokens=2, nll_sum=2.0)
    with pytest.raises(ValueError, match="duplicate loss record identity"):
        per_record_nll_rows({category: {"per_record": [row, row]}})


def test_loss_flattening_retains_legitimate_cross_category_overlap_and_nulls():
    from slm_training.evals.loss_suites import per_record_nll_rows

    broad = dict(id="case", mean_nll=1.0, masked_tokens=2, nll_sum=2.0)
    binding = dict(id="case", mean_nll=None, masked_tokens=0, nll_sum=0.0)
    rows = per_record_nll_rows({"broad": {"per_record": [broad]},
                               "binding": {"per_record": [binding]}})
    assert len(rows) == 1 and rows[0]["nll"] == 1.0
    assert rows[0]["binding_nll"] is None and rows[0]["masked_tokens"] == 2


@pytest.mark.parametrize("changes", [
    {"mean_nll": True}, {"mean_nll": "1"}, {"mean_nll": float("nan")},
    {"nll_sum": float("inf")}, {"nll_sum": -1}, {"nll_sum": True},
    {"masked_tokens": True}, {"masked_tokens": 1.5}, {"masked_tokens": -1},
    {"masked_tokens": 0}, {"mean_nll": None},
])
def test_loss_flattening_rejects_invalid_sufficient_statistics(changes):
    from slm_training.evals.loss_suites import per_record_nll_rows

    row = dict(id="case", mean_nll=1.0, nll_sum=2.0, masked_tokens=2)
    with pytest.raises(ValueError):
        per_record_nll_rows({"broad": {"per_record": [{**row, **changes}]}})


def test_secondary_suite_cache_rejects_changed_evaluator(clock, suite, tmp_path, monkeypatch):
    from slm_training.harness_core.eval_cache import EvalCache, EvalCacheConfig, EvalCacheMode

    cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ_WRITE, root=tmp_path / "cache"))
    config = _config(suite, tmp_path, "secondary-cache", eval_limit=1)
    write_jsonl(suite["test"] / "records.jsonl", [_record(id="smoke-00")])
    # Exercise the secondary index itself; preflight invalidation has a separate
    # unchanged adversary regression. Dataset identities still use actual bytes.
    monkeypatch.setattr(eval_runner, "_suite_cache_preflight",
                        lambda *args, **kwargs: eval_runner._SuiteCachePreflight())
    monkeypatch.setattr(eval_runner, "evaluator_identity", lambda _: "a" * 64)
    first = SlowStubDecoder(clock)
    result = eval_runner.evaluate(config, model=first, cache=cache,
                                  publish_agentv=False, **_identity(suite))
    assert first.calls == 1 and eval_runner._suite_result_cacheable(result)
    replay = SlowStubDecoder(clock)
    eval_runner.evaluate(config, model=replay, cache=cache,
                         publish_agentv=False, **_identity(suite))
    assert replay.calls == 0
    monkeypatch.setattr(eval_runner, "evaluator_identity", lambda _: "b" * 64)
    successor = SlowStubDecoder(clock)
    result = eval_runner.evaluate(config, model=successor, cache=cache,
                                  publish_agentv=False, **_identity(suite))
    assert successor.calls == 1 and result["evaluator_sha256"] == "b" * 64


@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize("sdk_failure", [False, True])
@pytest.mark.parametrize("previously_published", [False, True])
def test_complete_cache_replay_must_publish_in_current_run(clock, suite, tmp_path, monkeypatch, preflight, sdk_failure, previously_published):
    from slm_training.harness_core.eval_cache import EvalCache, EvalCacheConfig, EvalCacheMode

    config = _config(suite, tmp_path, "cache-origin", eval_limit=1)
    write_jsonl(suite["test"] / "records.jsonl", [_record(id="smoke-00")])
    cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ_WRITE, root=tmp_path / "cache"))
    monkeypatch.setattr("slm_training.evals.agentv.publish_model_evaluation",
                        lambda *args, **kwargs: {"output": "fixture-prior-publication.json"})
    original = eval_runner.evaluate(config, model=SlowStubDecoder(clock), cache=cache,
                                    publish_agentv=previously_published, **_identity(suite))
    assert original["measurement_complete"] and original["publication_complete"] is previously_published
    replay_config = replace(config, run_id="cache-destination")
    calls = []
    def publish(run_dir, board, **kwargs):
        calls.append((run_dir, board))
        if sdk_failure:
            raise RuntimeError("injected SDK unavailability")
        return {"output": str(run_dir / "fixture-agentv-result.json")}
    monkeypatch.setattr("slm_training.evals.agentv.publish_model_evaluation", publish)
    def forbidden_model_load(*args, **kwargs):
        pytest.fail("cache hit constructed a model")
    monkeypatch.setattr(eval_runner, "build_model", forbidden_model_load)
    decoder = SlowStubDecoder(clock)
    arguments = ({"checkpoint": suite["checkpoint"]} if preflight
                 else {"model": decoder, **_identity(suite)})
    if sdk_failure:
        with pytest.raises(RuntimeError, match="SDK unavailability"):
            eval_runner.evaluate(replay_config, cache=cache, **arguments)
    else:
        result = eval_runner.evaluate(replay_config, cache=cache, **arguments)
        assert result["publication_complete"] is True
        assert result["agentv"]["output"].startswith(str(replay_config.run_dir))
    assert len(calls) == 1 and calls[0][0] == replay_config.run_dir
    assert decoder.calls == 0
    durable = json.loads((replay_config.run_dir / "eval_smoke.json").read_text())
    assert durable["cache_replay"] and durable["completed_document_n"] == 1
    assert durable["publication_complete"] is (not sdk_failure)
    assert json.loads((replay_config.run_dir / "eval.json").read_text()) == durable


def test_unpublished_replay_cannot_inherit_prior_sdk_receipt(suite, tmp_path):
    metrics = _complete_cache_counts(publication_complete=True, agentv={"output": "prior-result"})
    config = _config(suite, tmp_path, "unpublished-replay")
    result = eval_runner._replay_cached_suite(config, metrics, record_n=1)
    assert result["publication_complete"] is False and "agentv" not in result
    assert metrics["publication_complete"] is True and metrics["agentv"]["output"] == "prior-result"
