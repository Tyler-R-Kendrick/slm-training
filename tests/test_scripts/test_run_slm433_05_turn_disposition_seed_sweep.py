"""Tests for VAR3-05 (SLM-433): seed-replication check for VAR3-04.

Stubs ``run_training`` rather than calling the real corpus pipeline: VAR3-04's
own test file already covers that a real corpus is built correctly and
carries genuine per-row content. This module's only new logic is the
seed-sweep aggregation (``disposition_counts``/``verdict``), which is what
these tests exercise directly and cheaply.
"""

from __future__ import annotations

import scripts.run_slm433_05_turn_disposition_seed_sweep as sweep


def _stub_report(seed: int, disposition: str) -> dict:
    return {
        "disposition": disposition,
        "arms": {
            "held_out": {
                "disposition_off": {"composite_penalized_error_rate": 0.5},
                "disposition_trained": {
                    "composite_penalized_error_rate": 0.4 if disposition == "held_out_improvement" else 0.5,
                    "abstention_rate": 0.5,
                },
            }
        },
        "corpus": {"train_roots_skipped": [], "dev_roots_skipped": []},
    }


def test_replicates_every_seed(monkeypatch) -> None:
    monkeypatch.setattr(
        sweep,
        "run_training",
        lambda **kwargs: _stub_report(kwargs["seed"], "held_out_improvement"),
    )
    report = sweep.run_sweep(seeds=(1, 2, 3))
    assert [row["seed"] for row in report["rows"]] == [1, 2, 3]
    assert report["disposition_counts"] == {"held_out_improvement": 3}
    assert report["verdict"] == "replicates_every_seed"
    assert report["claim_class"] == "wiring"


def test_does_not_replicate(monkeypatch) -> None:
    monkeypatch.setattr(
        sweep, "run_training", lambda **kwargs: _stub_report(kwargs["seed"], "no_difference")
    )
    report = sweep.run_sweep(seeds=(1, 2))
    assert report["verdict"] == "does_not_replicate"
    assert report["disposition_counts"] == {"no_difference": 2}


def test_seed_sensitive_mixed_dispositions(monkeypatch) -> None:
    dispositions = iter(["held_out_improvement", "no_difference", "held_out_improvement"])
    monkeypatch.setattr(
        sweep, "run_training", lambda **kwargs: _stub_report(kwargs["seed"], next(dispositions))
    )
    report = sweep.run_sweep(seeds=(1, 2, 3))
    assert report["verdict"] == "seed_sensitive"
    assert report["disposition_counts"] == {"held_out_improvement": 2, "no_difference": 1}
    assert sum(report["disposition_counts"].values()) == 3
