"""Regression tests for SDE3-01 content-addressed evaluation cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.evals.eval_cache import (
    CACHE_SCHEMA_VERSION,
    EvalCache,
    EvalCacheConfig,
    EvalCacheKey,
    EvalCacheMode,
    metric_result_key,
    request_generation_key,
    suite_result_key,
)


def test_key_fingerprint_is_deterministic() -> None:
    key = EvalCacheKey(
        layer="generation",
        checkpoint_sha256="abc",
        request_sha256="def",
        policy={"temperature": 0.7},
        component_versions={"evals.scoring": "v2"},
    )
    assert key.fingerprint() == key.fingerprint()


def test_key_fingerprint_changes_with_dependency() -> None:
    base = EvalCacheKey(
        layer="generation",
        checkpoint_sha256="abc",
        request_sha256="def",
        policy={"temperature": 0.7},
    )
    changed_policy = EvalCacheKey(
        layer="generation",
        checkpoint_sha256="abc",
        request_sha256="def",
        policy={"temperature": 0.8},
    )
    changed_request = EvalCacheKey(
        layer="generation",
        checkpoint_sha256="abc",
        request_sha256="dee",
        policy={"temperature": 0.7},
    )
    assert base.fingerprint() != changed_policy.fingerprint()
    assert base.fingerprint() != changed_request.fingerprint()


def test_cache_round_trip(tmp_path: Path) -> None:
    cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ_WRITE, root=tmp_path))
    key = EvalCacheKey(layer="test", request_sha256="r1")
    payload = {"prediction": "hello", "score": 0.9}
    cache.put(key, payload)
    assert cache.get(key) == payload


def test_mode_off_returns_none(tmp_path: Path) -> None:
    seed = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ_WRITE, root=tmp_path))
    key = EvalCacheKey(layer="test", request_sha256="r1")
    seed.put(key, {"x": 1})

    cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.OFF, root=tmp_path))
    assert cache.get(key) is None


def test_mode_read_returns_existing_but_does_not_write(tmp_path: Path) -> None:
    write_cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ_WRITE, root=tmp_path))
    key = EvalCacheKey(layer="test", request_sha256="r1")
    write_cache.put(key, {"x": 1})

    read_cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ, root=tmp_path))
    assert read_cache.get(key) == {"x": 1}
    with pytest.raises(RuntimeError):
        read_cache.put(key, {"x": 2})


def test_mode_refresh_overwrites(tmp_path: Path) -> None:
    cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ_WRITE, root=tmp_path))
    key = EvalCacheKey(layer="test", request_sha256="r1")
    cache.put(key, {"x": 1})

    refresh_cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.REFRESH, root=tmp_path))
    refresh_cache.put(key, {"x": 2})
    assert cache.get(key) == {"x": 2}


def test_corrupted_entry_is_rejected(tmp_path: Path) -> None:
    cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ_WRITE, root=tmp_path))
    key = EvalCacheKey(layer="test", request_sha256="r1")
    cache.put(key, {"x": 1})

    path = tmp_path / key.fingerprint()[:2] / f"{key.fingerprint()[2:]}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["payload_checksum"] = "deadbeef"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert cache.get(key) is None


def test_schema_mismatch_is_rejected(tmp_path: Path) -> None:
    cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ_WRITE, root=tmp_path))
    key = EvalCacheKey(layer="test", request_sha256="r1")
    cache.put(key, {"x": 1})

    path = tmp_path / key.fingerprint()[:2] / f"{key.fingerprint()[2:]}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = CACHE_SCHEMA_VERSION + 1
    path.write_text(json.dumps(data), encoding="utf-8")

    assert cache.get(key) is None


def test_suite_result_key_includes_all_dependencies() -> None:
    key = suite_result_key(
        suite="smoke",
        checkpoint_sha256="ckpt",
        eval_data_manifest_sha="data",
        eval_suite_manifest_sha="suite",
        eval_limit=10,
        evaluation_policy={"grammar_constrained": True},
        component_versions={"evals.scoring": "v2"},
    )
    assert key.layer == "suite_result"
    assert key.suite == "smoke"
    assert key.checkpoint_sha256 == "ckpt"


def test_suite_result_key_distinguishes_eval_offset() -> None:
    common = {
        "suite": "rico_held",
        "checkpoint_sha256": "ckpt",
        "eval_data_manifest_sha": "data",
        "eval_suite_manifest_sha": "suite",
        "eval_limit": 9,
        "evaluation_policy": {},
        "component_versions": {},
    }

    prefix = suite_result_key(**common, extra={"eval_offset": 0})
    shifted = suite_result_key(**common, extra={"eval_offset": 40})

    assert prefix.fingerprint() != shifted.fingerprint()


def test_request_generation_key_and_metric_key_differ() -> None:
    gen_key = request_generation_key(
        checkpoint_sha256="ckpt",
        request_sha256="req",
        evaluation_policy={},
        component_versions={},
    )
    metric_key = metric_result_key(
        prediction_sha256="req",
        source_record_sha256="rec",
        evaluation_policy={},
        component_versions={},
    )
    assert gen_key.fingerprint() != metric_key.fingerprint()


def test_nonfinite_floats_are_dropped(tmp_path: Path) -> None:
    cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ_WRITE, root=tmp_path))
    key = EvalCacheKey(layer="test", request_sha256="r1")
    cache.put(key, {"ok": 1.0, "bad": float("inf"), "nested": {"bad": float("nan")}})
    payload = cache.get(key)
    assert payload is not None
    assert payload["ok"] == 1.0
    assert payload["bad"] is None
    assert payload["nested"]["bad"] is None
