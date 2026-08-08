"""Tests for the PCT-002 hermetic replay bundle."""

from __future__ import annotations

import dataclasses

import pytest

from slm_training.models.decode_stats import DecodeIdentityV1
from slm_training.runtime.telemetry import (
    ReplayBundleV1,
    append_replay_bundle,
    build_replay_bundle,
    iter_replay_bundles,
    mirror_bundle_best_effort,
    replay_bundle_integrity_ok,
)


def _identity(**overrides: object) -> DecodeIdentityV1:
    base = dict(
        contract_version=2,
        pack_id="openui",
        tokenizer_id="dsl-native/v5",
        artifact_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        evaluator_version="eval/v1",
        evaluator_hash="c" * 64,
        code_commit="deadbeef",
    )
    base.update(overrides)
    return DecodeIdentityV1(**base)


def _solver_state_event(fp: str) -> dict:
    return {
        "kind": "solver_state",
        "state_fingerprint": fp,
        "decision_level": 0,
        "domain": {},
    }


class _FakeStore:
    """Minimal ReplayBundleStore duck-type for append/iter tests."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def append(self, row: dict) -> str:
        trajectory_id = f"{len(self.rows):04d}"
        self.rows.append(row)
        return trajectory_id

    def iter_kind(self, kind: str):
        for row in self.rows:
            if row.get("kind", "decode") == kind:
                yield row


def test_replay_bundle_is_lossless_and_golden() -> None:
    bundle = build_replay_bundle(
        request_id="req-1",
        identity=_identity(),
        seed=7,
        decoder_choices=({"kind": "decoder_choice", "action": "select"},),
        legal_domain_size=42,
        legal_domain_status="complete",
        legal_domain_digest="d" * 64,
        artifact_state={"cache": "warm"},
        final_evidence={"outcome": "accept"},
    )
    assert bundle.schema_version == "replay_bundle/v1"
    assert bundle.request_id == "req-1"
    assert bundle.seed == 7
    assert bundle.identity == _identity().to_dict()
    assert bundle.replayable
    assert bundle.non_replayable_reasons == ()
    assert bundle.bundle_hash
    assert replay_bundle_integrity_ok(bundle)
    assert ReplayBundleV1.from_dict(bundle.to_dict()) == bundle


def test_missing_request_id_fails_closed() -> None:
    bundle = build_replay_bundle(request_id="", identity=_identity())
    assert not bundle.replayable
    assert "missing request_id" in bundle.non_replayable_reasons


def test_missing_checkpoint_identity_fails_closed() -> None:
    bundle = build_replay_bundle(request_id="req-2", identity=None)
    assert not bundle.replayable
    assert "missing checkpoint identity" in bundle.non_replayable_reasons


def test_solver_replay_violation_marks_non_replayable() -> None:
    # A certified_deduction citing a before_fingerprint that was never
    # established by a solver_state event is a real VSS1-04 violation.
    bad_events = (
        {
            "kind": "certified_deduction",
            "before_fingerprint": "never-seen",
            "after_fingerprint": "also-never-seen",
            "removed": [],
            "certificate_id": None,
        },
    )
    bundle = build_replay_bundle(
        request_id="req-3", identity=_identity(), decoder_choices=bad_events
    )
    assert not bundle.replayable
    assert any("solver replay violation" in r for r in bundle.non_replayable_reasons)


def test_clean_solver_events_remain_replayable() -> None:
    events = (_solver_state_event("fp0"),)
    bundle = build_replay_bundle(
        request_id="req-4", identity=_identity(), decoder_choices=events
    )
    assert bundle.replayable
    assert bundle.non_replayable_reasons == ()


def test_observability_disabled_or_enabled_yields_identical_bundle() -> None:
    """build_replay_bundle never touches telemetry state, so binding or not
    binding a CycleTelemetry/RunTrace can never change the resulting bundle."""
    kwargs = dict(
        request_id="req-5",
        identity=_identity(),
        seed=3,
        decoder_choices=({"kind": "decoder_choice", "action": "select"},),
    )
    without_telemetry = build_replay_bundle(**kwargs)

    from slm_training.runtime.telemetry import CycleTelemetry, bind_telemetry

    with bind_telemetry(CycleTelemetry()):
        with_telemetry = build_replay_bundle(**kwargs)

    assert without_telemetry.bundle_hash == with_telemetry.bundle_hash
    assert without_telemetry == with_telemetry


def test_export_errors_never_propagate() -> None:
    bundle = build_replay_bundle(request_id="req-6", identity=_identity())

    def _raising_sink(payload: dict) -> None:
        raise RuntimeError("sink is down")

    assert mirror_bundle_best_effort(bundle, _raising_sink) is False
    # No sink at all is also a no-op, not an error.
    assert mirror_bundle_best_effort(bundle, None) is False

    calls: list[dict] = []
    assert mirror_bundle_best_effort(bundle, calls.append) is True
    assert calls == [bundle.to_dict()]


def test_redaction_scrubs_secret_shaped_strings_and_bounds_length() -> None:
    bundle = build_replay_bundle(
        request_id="req-7",
        identity=_identity(),
        artifact_state={"note": "token=sk-abcdefghijklmnopqrstuvwx"},
        final_evidence={"long": "x" * 1000},
    )
    assert "sk-abcdefghijklmnopqrstuvwx" not in bundle.artifact_state["note"]
    assert "[REDACTED]" in bundle.artifact_state["note"]
    assert len(bundle.final_evidence["long"]) < 1000


def test_replay_bundle_is_frozen() -> None:
    bundle = build_replay_bundle(request_id="req-8", identity=_identity())
    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.request_id = "tampered"  # type: ignore[misc]


def test_replay_bundle_tamper_and_schema_rejection() -> None:
    bundle = build_replay_bundle(request_id="req-9", identity=_identity())

    tampered = bundle.to_dict()
    tampered["replayable"] = not tampered["replayable"]
    with pytest.raises(ValueError, match="hash mismatch"):
        ReplayBundleV1.from_dict(tampered)

    wrong_schema = bundle.to_dict()
    wrong_schema["schema_version"] = "replay_bundle/v0"
    with pytest.raises(ValueError, match="unsupported replay bundle schema"):
        ReplayBundleV1.from_dict(wrong_schema)


def test_append_and_iter_replay_bundles_round_trip_through_a_store() -> None:
    store = _FakeStore()
    first = build_replay_bundle(request_id="req-10", identity=_identity(), seed=1)
    second = build_replay_bundle(request_id="req-11", identity=_identity(), seed=2)

    append_replay_bundle(store, first)
    append_replay_bundle(store, second)

    replayed = list(iter_replay_bundles(store))
    assert replayed == [first, second]
