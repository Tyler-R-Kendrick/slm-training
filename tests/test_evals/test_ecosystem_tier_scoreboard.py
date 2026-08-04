"""Tests for the tier-split generation × formal scoreboard."""

from __future__ import annotations

import pytest

from slm_training.dsl.ecosystem_tier import (
    CORE_FORMAL_MODULES,
    ECOSYSTEM_FORMAL_MODULES,
)
from slm_training.evals.ecosystem_tier_scoreboard import (
    SCOREBOARD_SCHEMA,
    aggregate_generation_rows,
    build_ecosystem_tier_scoreboard,
    scoreboard_markdown,
)
from slm_training.harness_core.bounded_process import BoundedProcessResult, ProcessOutcome
from slm_training.evals import ecosystem_tier_scoreboard as scoreboard


def test_aggregate_generation_rows_means() -> None:
    rows = [
        {"parse_rate": 1.0, "component_type_recall": 0.5},
        {"parse_rate": 0.5, "component_type_recall": 0.25},
    ]
    agg = aggregate_generation_rows(rows)
    assert agg["parse_rate"] == 0.75
    assert agg["component_type_recall"] == 0.375
    assert agg["parse_rate__n"] == 2


def test_scoreboard_splits_generation_and_formal() -> None:
    statuses = {
        **{m: "proved" for m in CORE_FORMAL_MODULES},
        **{m: "unknown" for m in ECOSYSTEM_FORMAL_MODULES},
    }
    board = build_ecosystem_tier_scoreboard(
        generation_metrics={
            "parse_rate": 0.95,
            "component_type_recall": 0.4,
            "structural_similarity": 0.55,
        },
        formal_statuses=statuses,
        recipe={"fixture_demo": True},
    )
    assert board["schema"] == SCOREBOARD_SCHEMA
    assert board["production_core"]["generation_metrics"]["parse_rate"] == 0.95
    assert "component_type_recall" in board["ecosystem_library"]["generation_metrics"]
    assert board["production_core"]["formal_preflight"]["rollup"] == "proved"
    assert board["ecosystem_library"]["formal_preflight"]["rollup"] == "unknown"
    # Core formal success is independent of ecosystem formal rollup.
    assert board["separation"]["core_formal_ok"] is True
    assert board["separation"]["separation_holds"] is True
    assert board["honesty"]["ship_claim"] is False
    assert "version_stamp" in board

    md = scoreboard_markdown(board)
    assert "Production core" in md
    assert "Ecosystem library" in md
    assert "Formal preflight" in md


def test_scoreboard_from_rows_only() -> None:
    board = build_ecosystem_tier_scoreboard(
        generation_rows=[
            {"parse_rate": 1.0, "component_type_recall": 0.0},
            {"parse_rate": 1.0, "component_type_recall": 1.0},
        ],
        formal_statuses={m: "proved" for m in CORE_FORMAL_MODULES},
    )
    assert board["production_core"]["generation_metrics"]["parse_rate"] == 1.0
    assert (
        board["ecosystem_library"]["generation_metrics"]["component_type_recall"]
        == 0.5
    )


def test_probe_lean_timeout_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scoreboard,
        "run_formal_process",
        lambda *args, **kwargs: BoundedProcessResult(
            command=tuple(args[0]),
            outcome=ProcessOutcome.TIMED_OUT,
            returncode=124,
            stdout="",
            stderr="",
            duration_seconds=0.0,
            timed_out=True,
        ),
    )

    statuses = scoreboard.probe_lean_formal_statuses(timeout_s=1.0)

    assert set(statuses.values()) == {"error:TimeoutExpired"}
