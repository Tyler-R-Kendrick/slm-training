"""Tests for VAR3-04 (SLM-433): real argument-bound turn-disposition corpus."""

from __future__ import annotations

from slm_training.versioning import build_version_stamp

from scripts.run_slm433_04_turn_disposition_real_corpus import (
    DEV_SOURCE,
    TRAIN_SOURCE,
    _admit_raw_traces,
    _rows_from_raw,
)


def test_real_admitted_roots_carry_more_than_one_reference_signature(tmp_path) -> None:
    """The whole point of VAR3-04: unlike VAR3-03's zero-argument fixture,
    a real corpus built from the production operator library carries
    genuine per-row distinguishing reference-row content."""
    stamp = build_version_stamp("config.levers")
    raw, skipped = _admit_raw_traces(
        source=TRAIN_SOURCE, target_count=2, work_dir=tmp_path / "train", stamp=stamp
    )
    assert len(raw) == 2
    assert isinstance(skipped, list)

    rows, rejected = _rows_from_raw(raw, action_frequency={}, split="train")
    assert rows
    assert isinstance(rejected, list)

    signatures = {
        (reference.ref_kind.value, reference.value_type)
        for row in rows
        for reference in _view(row).reference_rows
    }
    assert len(signatures) > 1, (
        "a real corpus must carry more than one (ref_kind, value_type) "
        "signature across its reference rows -- a single signature would "
        "reproduce VAR3-03's degenerate no-argument-slot case"
    )


def test_dev_source_is_disjoint_from_train_source() -> None:
    """Train and dev roots must come from wholly distinct source files so
    trace-level leakage is structurally impossible, never merely checked
    after the fact."""
    assert TRAIN_SOURCE != DEV_SOURCE
    assert TRAIN_SOURCE.exists()
    assert DEV_SOURCE.exists()


def _view(row):
    from slm_training.models.operator_policy_view import operator_policy_input_from_dict

    return operator_policy_input_from_dict(row.policy_input)
