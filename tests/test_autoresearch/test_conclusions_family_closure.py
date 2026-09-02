"""Tests for hypothesis-family conclusion criteria (WP-4).

Covers policy v2 superset + fallback, family-key canonicalization and the
observed lever taxonomy restriction, append-only idempotent closure, reopen
semantics, and the ``concluded_family`` preflight plugin.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from slm_training.autoresearch import conclusions
from slm_training.autoresearch.conclusions import (
    CLOSED_APPROACHES_SCHEMA,
    DEFAULT_CLOSED_APPROACHES_PATH,
    POLICY_V1_PATH,
    POLICY_V2_PATH,
    POLICY_V3_PATH,
    FamilyLedger,
    canonicalize_lever_key,
    closed_approach_id,
    conclude,
    conclusion_policy,
    family_key_for_levers,
    is_family_closed,
    known_lever_taxonomy,
    load_closed_approaches,
    load_policy,
)
from slm_training.autoresearch.preflight import concluded_family

TAXONOMY = frozenset({"bounds", "canvas", "component-plan", "binder-topology"})


def _failure(
    record_id: str,
    levers: tuple[str, ...] = ("bounds", "canvas"),
    *,
    outcome: str = "confirm_failed",
    n_seeds: int = 8,
    decidable: bool = True,
    **extra: object,
) -> dict:
    return {
        "record_id": record_id,
        "lever_keys": list(levers),
        "outcome": outcome,
        "n_seeds": n_seeds,
        "decidable": decidable,
        **extra,
    }


def _three_failures(**extra: object) -> list[dict]:
    return [_failure(f"ev-{i}", **extra) for i in range(3)]


# ---------------------------------------------------------------------------
# Policy v2 + fallback
# ---------------------------------------------------------------------------


def test_policy_v2_and_v3_carry_the_conclusion_block() -> None:
    """v2/v3 recalibrated screening fields (#1716, P2) on top of v1; the
    conclusion block and the schema family are what this module depends on."""
    v1 = json.loads(POLICY_V1_PATH.read_text(encoding="utf-8"))
    v2 = json.loads(POLICY_V2_PATH.read_text(encoding="utf-8"))
    v3 = json.loads(POLICY_V3_PATH.read_text(encoding="utf-8"))
    assert v1["version"] != v2["version"] != v3["version"]
    assert v3["schema"] == v2["schema"] == v1["schema"] == "autotrain_climb_policy/v1"
    assert "conclusion_policy" not in v1
    for payload in (v2, v3):
        block = payload["conclusion_policy"]
        assert block["family_close_after_adequately_powered_failures"] == 3
        assert block["adequate_power_requires"] == {"min_seeds": 8, "decidable": True}
        assert block["closed_families_reopen_on"] == [
            "new_lever_key",
            "harness_version_change",
        ]


def test_load_policy_prefers_v3_then_v2(tmp_path: Path) -> None:
    policy = load_policy()
    assert policy.path.name == "policy.v3.json"
    v3 = json.loads(POLICY_V3_PATH.read_text(encoding="utf-8"))
    assert policy.version == v3["version"]
    block = conclusion_policy(policy)
    assert block["family_close_after_adequately_powered_failures"] == 3
    # Without v3 on disk the loader takes v2 (no fallback notice).
    for src in (POLICY_V1_PATH, POLICY_V2_PATH):
        (tmp_path / src.name).write_text(src.read_text(encoding="utf-8"), "utf-8")
    assert load_policy(tmp_path).path.name == "policy.v2.json"


def test_load_policy_falls_back_to_v1_with_single_notice(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "policy.v1.json").write_text(
        POLICY_V1_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    conclusions._fallback_notified.discard(str(tmp_path.resolve()))
    with caplog.at_level(logging.INFO, logger=conclusions.logger.name):
        first = load_policy(tmp_path)
        second = load_policy(tmp_path)
    assert first.path.name == "policy.v1.json"
    assert second.path == first.path
    notices = [r for r in caplog.records if "falling back" in r.getMessage()]
    assert len(notices) == 1
    # v1 lacks the block; defaults apply and mirror v2.
    block = conclusion_policy(first)
    assert block["adequate_power_requires"] == {"min_seeds": 8, "decidable": True}


# ---------------------------------------------------------------------------
# Family keys + taxonomy
# ---------------------------------------------------------------------------


def test_family_key_canonicalization() -> None:
    assert canonicalize_lever_key("  Component_Plan ") == "component-plan"
    key = family_key_for_levers(["canvas", "Bounds", "bounds", "component_plan"])
    assert key == "bounds+canvas+component-plan"
    # Order and duplicates never change the family.
    assert key == family_key_for_levers(["component-plan", "bounds", "canvas"])


def test_known_lever_taxonomy_matches_committed_ledger() -> None:
    taxonomy = known_lever_taxonomy()
    assert {"bounds", "canvas", "component-plan", "binder-topology"} <= taxonomy
    assert all(k == canonicalize_lever_key(k) for k in taxonomy)


def test_family_ledger_excludes_extra_taxonomy_levers() -> None:
    records = [
        _failure("ev-known", ("bounds",)),
        _failure("ev-unknown", ("bounds", "warp-drive")),
        {"record_id": "", "lever_keys": ["bounds"], "outcome": "confirm_failed"},
    ]
    ledger = FamilyLedger.from_records(records, taxonomy=TAXONOMY)
    assert set(ledger.families) == {"bounds"}
    assert len(ledger.excluded) == 2


def test_adequately_powered_failures_filtering() -> None:
    records = [
        _failure("ev-ok"),
        _failure("ev-underpowered", n_seeds=7),
        _failure("ev-undecidable", decidable=False),
        _failure("ev-not-failure", outcome="confirmed"),
        _failure("ev-ship-rejected", outcome="ship_rejected"),
    ]
    ledger = FamilyLedger.from_records(records, taxonomy=TAXONOMY)
    failures = ledger.adequately_powered_failures("bounds+canvas")
    assert sorted(r.record_id for r in failures) == ["ev-ok", "ev-ship-rejected"]


# ---------------------------------------------------------------------------
# conclude(): threshold, append-only, idempotent
# ---------------------------------------------------------------------------


def test_conclude_below_threshold_appends_nothing(tmp_path: Path) -> None:
    path = tmp_path / "closed.json"
    appended = conclude(
        _three_failures()[:2],
        path=path,
        taxonomy=TAXONOMY,
        harness_versions={},
        run_date="2026-08-09",
    )
    assert appended == []
    assert not path.exists()


def test_conclude_closes_family_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "closed.json"
    records = _three_failures(hypothesis_text="serves I10 ladder position")
    appended = conclude(
        records,
        path=path,
        taxonomy=TAXONOMY,
        harness_versions={"evals.scoring": "v22"},
        run_date="2026-08-09",
    )
    assert len(appended) == 1
    record = appended[0]
    assert record.family_key == "bounds+canvas"
    assert record.evidence_record_ids == ["ev-0", "ev-1", "ev-2"]
    assert record.goal_invariant_served == "I10"
    assert record.reopen_conditions == ["new_lever_key", "harness_version_change"]
    assert record.closed_at_run_date == "2026-08-09"
    assert record.id == closed_approach_id("bounds+canvas", ["ev-0", "ev-1", "ev-2"])

    # Idempotent: same records add nothing.
    again = conclude(
        records,
        path=path,
        taxonomy=TAXONOMY,
        harness_versions={"evals.scoring": "v22"},
        run_date="2026-08-10",
    )
    assert again == []
    _, stored = load_closed_approaches(path)
    assert len(stored) == 1


def test_conclude_is_append_only(tmp_path: Path) -> None:
    path = tmp_path / "closed.json"
    sentinel = {
        "id": "deadbeef",
        "family_key": "component-plan",
        "closed_at_run_date": "2026-01-01",
        "evidence_record_ids": ["old-1"],
        "goal_invariant_served": "unknown",
        "reopen_conditions": ["new_lever_key"],
        "harness_versions": {},
    }
    path.write_text(
        json.dumps({"schema": CLOSED_APPROACHES_SCHEMA, "records": [sentinel]}),
        encoding="utf-8",
    )
    conclude(
        _three_failures(),
        path=path,
        taxonomy=TAXONOMY,
        harness_versions={},
        run_date="2026-08-09",
    )
    _, stored = load_closed_approaches(path)
    assert stored[0] == sentinel, "existing entries must never be rewritten"
    assert len(stored) == 2
    assert stored[1]["family_key"] == "bounds+canvas"


def test_conclude_skips_family_with_still_binding_closure(tmp_path: Path) -> None:
    path = tmp_path / "closed.json"
    records = _three_failures()
    conclude(
        records,
        path=path,
        taxonomy=TAXONOMY,
        harness_versions={},
        run_date="2026-08-09",
    )
    # Different evidence ids, same family, closure still binding → skip.
    more = [_failure(f"ev-new-{i}") for i in range(3)]
    again = conclude(
        more, path=path, taxonomy=TAXONOMY, harness_versions={}, run_date="2026-08-10"
    )
    assert again == []
    _, stored = load_closed_approaches(path)
    assert len(stored) == 1


def test_conclude_can_reclose_after_harness_version_change(tmp_path: Path) -> None:
    path = tmp_path / "closed.json"
    conclude(
        _three_failures(),
        path=path,
        taxonomy=TAXONOMY,
        harness_versions={"evals.scoring": "v22"},
        run_date="2026-08-09",
    )
    reclosed = conclude(
        [_failure(f"ev-new-{i}") for i in range(3)],
        path=path,
        taxonomy=TAXONOMY,
        harness_versions={"evals.scoring": "v23"},
        run_date="2026-08-10",
    )
    assert len(reclosed) == 1
    _, stored = load_closed_approaches(path)
    assert len(stored) == 2


def test_goal_invariant_unknown_when_ambiguous_or_absent(tmp_path: Path) -> None:
    path = tmp_path / "closed.json"
    records = [
        _failure("ev-0", hypothesis_text="serves I10"),
        _failure("ev-1", hypothesis_text="serves I6"),
        _failure("ev-2"),
    ]
    appended = conclude(
        records,
        path=path,
        taxonomy=TAXONOMY,
        harness_versions={},
        run_date="2026-08-09",
    )
    assert appended[0].goal_invariant_served == "unknown"


def test_closed_approach_id_is_content_addressed() -> None:
    a = closed_approach_id("bounds+canvas", ["ev-1", "ev-0"])
    b = closed_approach_id("bounds+canvas", ["ev-0", "ev-1"])
    c = closed_approach_id("bounds", ["ev-0", "ev-1"])
    assert a == b
    assert a != c
    assert len(a) == 64


# ---------------------------------------------------------------------------
# is_family_closed + reopen semantics
# ---------------------------------------------------------------------------


@pytest.fixture()
def closed_ledger(tmp_path: Path) -> Path:
    path = tmp_path / "closed.json"
    conclude(
        _three_failures(),
        path=path,
        taxonomy=TAXONOMY,
        harness_versions={"evals.scoring": "v22", "gates.ship": "v6"},
        run_date="2026-08-09",
    )
    return path


def test_is_family_closed_true_when_binding(closed_ledger: Path) -> None:
    assert is_family_closed(
        "bounds+canvas",
        current_lever_keys=["bounds", "canvas"],
        harness_versions={"evals.scoring": "v22", "gates.ship": "v6"},
        path=closed_ledger,
    )
    # Family-key argument is canonicalized before lookup.
    assert is_family_closed(
        "Canvas+BOUNDS",
        current_lever_keys=["canvas", "bounds"],
        harness_versions={"evals.scoring": "v22"},
        path=closed_ledger,
    )


def test_new_lever_key_reopens(closed_ledger: Path) -> None:
    assert not is_family_closed(
        "bounds+canvas",
        current_lever_keys=["bounds", "canvas", "binder-topology"],
        harness_versions={"evals.scoring": "v22"},
        path=closed_ledger,
    )


def test_harness_version_change_reopens(closed_ledger: Path) -> None:
    assert not is_family_closed(
        "bounds+canvas",
        current_lever_keys=["bounds", "canvas"],
        harness_versions={"evals.scoring": "v99"},
        path=closed_ledger,
    )
    # Components absent from the closing record cannot reopen it.
    assert is_family_closed(
        "bounds+canvas",
        current_lever_keys=["bounds", "canvas"],
        harness_versions={"harness.model_build.eval": "v80"},
        path=closed_ledger,
    )


def test_unclosed_family_is_open(closed_ledger: Path) -> None:
    assert not is_family_closed(
        "component-plan",
        current_lever_keys=["component-plan"],
        harness_versions={},
        path=closed_ledger,
    )


# ---------------------------------------------------------------------------
# Committed artifacts
# ---------------------------------------------------------------------------


def test_committed_closed_approaches_ledger_is_empty_but_valid() -> None:
    payload, records = load_closed_approaches(DEFAULT_CLOSED_APPROACHES_PATH)
    assert payload["schema"] == CLOSED_APPROACHES_SCHEMA
    assert records == []


# ---------------------------------------------------------------------------
# Preflight plugin
# ---------------------------------------------------------------------------


def test_plugin_contract_surface() -> None:
    assert concluded_family.CHECK.check_id == "concluded_family"
    verdict = concluded_family.CHECK.run(
        {"hypothesis_text": "x", "lever_keys": [], "n_seeds": 2}
    )
    assert verdict.check_id == "concluded_family"
    assert verdict.verdict == "pass"


def test_plugin_blocks_closed_family(
    closed_ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conclusions, "DEFAULT_CLOSED_APPROACHES_PATH", closed_ledger)
    monkeypatch.setattr(
        conclusions,
        "current_harness_versions",
        lambda components=None: {"evals.scoring": "v22", "gates.ship": "v6"},
    )
    verdict = concluded_family.CHECK.run(
        {
            "hypothesis_text": "retry the same thing",
            "lever_keys": ["bounds", "canvas"],
            "config_fingerprint": "abc",
            "n_seeds": 8,
            "steps": 40,
            "minimum_effect": 0.01,
            "endpoint_metric": "held_out.structural_similarity",
        }
    )
    assert verdict.verdict == "block"
    _, stored = load_closed_approaches(closed_ledger)
    closure_id = stored[0]["id"]
    assert any(closure_id in reason for reason in verdict.reasons)
    assert any("reopen conditions" in reason for reason in verdict.reasons)
    assert verdict.data["closed_approach_id"] == closure_id
    assert verdict.data["reopen_conditions"] == [
        "new_lever_key",
        "harness_version_change",
    ]


def test_plugin_passes_when_reopen_condition_met(
    closed_ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conclusions, "DEFAULT_CLOSED_APPROACHES_PATH", closed_ledger)
    monkeypatch.setattr(
        conclusions,
        "current_harness_versions",
        lambda components=None: {"evals.scoring": "v22"},
    )
    verdict = concluded_family.CHECK.run(
        {"lever_keys": ["bounds", "canvas", "binder-topology"], "n_seeds": 8}
    )
    assert verdict.verdict == "pass"


def test_plugin_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    verdict = concluded_family.CHECK.run(None)  # type: ignore[arg-type]
    assert verdict.verdict == "pass"

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("ledger unavailable")

    monkeypatch.setattr(conclusions, "binding_closure", _boom)
    verdict = concluded_family.CHECK.run({"lever_keys": ["bounds"]})
    # Package fail-soft law: a check bug is "warn" (never a silent "pass"),
    # so a crash reads differently in the persisted payload than a
    # verified-open family.
    assert verdict.verdict == "warn"
    assert any("fails open" in reason for reason in verdict.reasons)
