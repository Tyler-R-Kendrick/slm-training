"""Conclusion writer: the WP-4 production caller for family closures."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.autoresearch.conclusions import (
    CLOSED_APPROACHES_SCHEMA,
    load_closed_approaches,
)
from slm_training.autoresearch.heal.conclusion_writer import (
    conclusion_record_from_evidence,
    write_family_closures,
)


def _evidence_row(**overrides) -> dict:
    row = {
        "schema_version": "evidence_record/v1",
        "experiment_id": "exp-1",
        "campaign_id": "camp-1",
        "hypothesis_text": "lever helps",
        "lever_keys": ["ltr-tail", "literal-close"],
        "config_fingerprint": "a" * 64,
        "endpoint_metric": "smoke.structural_similarity",
        "effect_size": 0.0,
        "n_seeds": 8,
        "p_value": 0.4,
        "outcome": "confirm_failed",
        "blocker": None,
        "version_stamp": {},
        "source_path": "docs/design/x.json",
        "run_date": "2026-08-15",
    }
    row.update(overrides)
    return row


class TestMapping:
    def test_maps_powered_failure(self) -> None:
        record = conclusion_record_from_evidence(_evidence_row())
        assert record is not None
        assert record.is_failure
        assert record.n_seeds == 8
        assert record.decidable is True

    def test_missing_p_value_is_not_decidable(self) -> None:
        # A measurement that could not have rejected anything can never help
        # close a family (power law applied at conclusion time).
        record = conclusion_record_from_evidence(_evidence_row(p_value=None))
        assert record is not None
        assert record.decidable is False

    def test_no_levers_is_dropped(self) -> None:
        assert conclusion_record_from_evidence(
            _evidence_row(lever_keys=[])
        ) is None


class TestWritePath:
    def _taxonomy(self) -> frozenset[str]:
        return frozenset({"ltr-tail", "literal-close"})

    def test_closes_family_after_threshold_and_is_idempotent(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from slm_training.autoresearch import conclusions as conc

        monkeypatch.setattr(
            conc, "known_lever_taxonomy", lambda *a, **k: self._taxonomy()
        )
        target = tmp_path / "closed_approaches.v1.json"
        rows = [
            _evidence_row(experiment_id=f"exp-{i}", config_fingerprint=str(i) * 64)
            for i in range(3)
        ]
        appended = write_family_closures(records=rows, path=target)
        assert [r.family_key for r in appended] == ["literal-close+ltr-tail"]
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["schema"] == CLOSED_APPROACHES_SCHEMA
        assert len(payload["records"]) == 1

        again = write_family_closures(records=rows, path=target)
        assert again == []
        _, records = load_closed_approaches(target)
        assert len(records) == 1

    def test_underpowered_failures_never_close(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from slm_training.autoresearch import conclusions as conc

        monkeypatch.setattr(
            conc, "known_lever_taxonomy", lambda *a, **k: self._taxonomy()
        )
        target = tmp_path / "closed_approaches.v1.json"
        rows = [
            _evidence_row(
                experiment_id=f"exp-{i}",
                n_seeds=1,
                p_value=None,
            )
            for i in range(5)
        ]
        assert write_family_closures(records=rows, path=target) == []
        assert not target.exists()

    def test_empty_records_are_noop(self, tmp_path: Path) -> None:
        target = tmp_path / "closed_approaches.v1.json"
        assert write_family_closures(records=[], path=target) == []
        assert not target.exists()
