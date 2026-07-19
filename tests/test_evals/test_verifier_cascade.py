"""Regression tests for the EFS2-04 verifier cascade wiring."""

from __future__ import annotations

import json
from typing import Any

from slm_training.data.verify.stack import Gate
from slm_training.evals.verifier_cascade import (
    Verdict,
    VerifierCache,
    VerifierCascade,
    VerifierResultV1,
    VerifierStage,
    VerifierStageSpec,
    default_openui_cascade,
    make_gate_stage,
)


def _stage(
    stage_id: str,
    evaluator,
    *,
    sound_fail: bool = False,
    cost_hint: float = 1.0,
    skip: tuple[str, ...] = (),
    version: str = "1",
) -> VerifierStage:
    spec = VerifierStageSpec(
        stage_id=stage_id,
        version=version,
        name=stage_id,
        sound_fail=sound_fail,
        cost_hint=cost_hint,
        skip_stages_on_fail=skip,
        cache_policy="exact",
    )
    return VerifierStage(spec, evaluator)


def _pass(stage_id: str) -> VerifierResultV1:
    return VerifierResultV1(stage_id=stage_id, status=Verdict.PASS)


def _fail(stage_id: str) -> VerifierResultV1:
    return VerifierResultV1(stage_id=stage_id, status=Verdict.FAIL)


def _unknown(stage_id: str) -> VerifierResultV1:
    return VerifierResultV1(stage_id=stage_id, status=Verdict.UNKNOWN)


def test_sound_fail_prunes_and_skips_expensive() -> None:
    calls: list[str] = []

    def cheap(_source: str, _ctx: dict[str, Any] | None) -> VerifierResultV1:
        calls.append("cheap")
        return _fail("cheap")

    def expensive(_source: str, _ctx: dict[str, Any] | None) -> VerifierResultV1:
        calls.append("expensive")
        return _pass("expensive")

    cascade = VerifierCascade(
        [
            _stage("cheap", cheap, sound_fail=True, skip=("expensive",)),
            _stage("expensive", expensive, cost_hint=100.0),
        ]
    )
    result = cascade.evaluate("id", "source")

    assert result.pruned is True
    assert result.prune_stage_id == "cheap"
    assert result.final_status is Verdict.FAIL
    assert calls == ["cheap"]
    assert len(result.results) == 2
    assert result.results[1].status is Verdict.NOT_APPLICABLE
    assert result.results[1].skipped is True
    assert result.total_cost == 1.0


def test_unknown_continues_and_expensive_runs() -> None:
    calls: list[str] = []

    def cheap(_source: str, _ctx: dict[str, Any] | None) -> VerifierResultV1:
        calls.append("cheap")
        return _unknown("cheap")

    def expensive(_source: str, _ctx: dict[str, Any] | None) -> VerifierResultV1:
        calls.append("expensive")
        return _pass("expensive")

    cascade = VerifierCascade(
        [
            _stage("cheap", cheap, sound_fail=True),
            _stage("expensive", expensive, cost_hint=10.0),
        ]
    )
    result = cascade.evaluate("id", "source")

    assert result.pruned is False
    assert result.final_status is Verdict.UNKNOWN
    assert calls == ["cheap", "expensive"]
    assert result.total_cost == 11.0


def test_error_continues_and_is_not_cached() -> None:
    calls: list[int] = []

    def flaky(_source: str, _ctx: dict[str, Any] | None) -> VerifierResultV1:
        calls.append(len(calls))
        if len(calls) == 1:
            return VerifierResultV1(stage_id="flaky", status=Verdict.ERROR, sound=False)
        return _pass("flaky")

    cascade = VerifierCascade([_stage("flaky", flaky, sound_fail=True)])
    first = cascade.evaluate("id", "source")
    second = cascade.evaluate("id", "source")

    assert first.results[0].status is Verdict.ERROR
    assert second.results[0].status is Verdict.PASS
    assert len(calls) == 2


def test_cache_hit_avoids_expensive_re_evaluation() -> None:
    calls: list[str] = []

    def cheap(_source: str, _ctx: dict[str, Any] | None) -> VerifierResultV1:
        calls.append("cheap")
        return _pass("cheap")

    def expensive(_source: str, _ctx: dict[str, Any] | None) -> VerifierResultV1:
        calls.append("expensive")
        return _pass("expensive")

    cache = VerifierCache()
    cascade = VerifierCascade(
        [
            _stage("cheap", cheap, cost_hint=1.0),
            _stage("expensive", expensive, cost_hint=100.0),
        ],
        cache=cache,
    )
    cascade.evaluate("id", "source")
    result2 = cascade.evaluate("id", "source")

    assert calls == ["cheap", "expensive"]
    assert result2.results[1].cached is True
    assert result2.results[1].cost == 0.0
    assert result2.cache_hits >= 1


def test_cache_key_changes_with_stage_version() -> None:
    cache = VerifierCache()

    def pass_fn(_source: str, _ctx: dict[str, Any] | None) -> VerifierResultV1:
        return _pass("s")

    stage_v1 = _stage("s", pass_fn, version="1")
    stage_v2 = _stage("s", pass_fn, version="2")

    key1 = cache.key("x", stage_v1.spec, None, "pack", None)
    key2 = cache.key("x", stage_v2.spec, None, "pack", None)
    assert key1 != key2


def test_unsound_fail_does_not_prune() -> None:
    calls: list[str] = []

    def unsound(_source: str, _ctx: dict[str, Any] | None) -> VerifierResultV1:
        calls.append("unsound")
        return _fail("unsound")

    def next_stage(_source: str, _ctx: dict[str, Any] | None) -> VerifierResultV1:
        calls.append("next")
        return _pass("next")

    cascade = VerifierCascade(
        [
            _stage("unsound", unsound, sound_fail=False),
            _stage("next", next_stage),
        ]
    )
    result = cascade.evaluate("id", "source")

    assert result.pruned is False
    assert result.final_status is Verdict.FAIL
    assert calls == ["unsound", "next"]


def test_default_openui_cascade_prunes_empty_source() -> None:
    cascade = default_openui_cascade()
    result = cascade.evaluate("empty", "")

    assert result.pruned is True
    assert result.prune_stage_id == Gate.LEXICAL.value
    lexical = result.results[0]
    assert lexical.status is Verdict.FAIL
    assert lexical.sound is True
    # Later stages should be skipped because lexical failed.
    assert any(r.status is Verdict.NOT_APPLICABLE for r in result.results)


def test_flat_eval_runs_all_stages_even_after_fail() -> None:
    cascade = default_openui_cascade()
    result = cascade.evaluate_flat("empty", "")

    assert result.pruned is True
    # Flat run should exercise every gate (some may be PASS/FAIL/SKIP).
    assert len(result.results) == 6
    assert all(r.stage_id is not None for r in result.results)


def test_cascade_result_serializes_to_json() -> None:
    def cheap(_source: str, _ctx: dict[str, Any] | None) -> VerifierResultV1:
        return _pass("cheap")

    cascade = VerifierCascade([_stage("cheap", cheap)])
    result = cascade.evaluate("id", "source")
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    loaded = json.loads(serialized)

    assert loaded["candidate_id"] == "id"
    assert loaded["final_status"] == "PASS"
    assert loaded["pruned"] is False
    assert loaded["results"][0]["status"] == "PASS"


def test_make_gate_stage_wraps_existing_gate() -> None:
    from slm_training.data.verify.stack import GateResult, GateStatus

    stage = make_gate_stage(
        Gate.LEXICAL,
        lambda source: GateResult(Gate.LEXICAL, GateStatus.PASS, "ok"),
    )
    result = stage.evaluate("ok", None)
    assert result.status is Verdict.PASS
    assert result.stage_id == Gate.LEXICAL.value
