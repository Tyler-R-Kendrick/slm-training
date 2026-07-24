from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments.slm236_recurrent_latent_disposition import (
    RecurrentLatentDispositionV1,
    build_report,
    validate_report,
)


def test_rsc4_is_fail_closed_and_round_trips() -> None:
    root = Path(__file__).resolve().parents[3]
    report = build_report(root)
    assert report.disposition == "blocked"
    assert validate_report(report, root) == []
    assert RecurrentLatentDispositionV1.from_dict(report.to_dict()) == report
