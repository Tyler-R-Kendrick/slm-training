"""Regression coverage for the bounded SLM-403 policy-matrix runner."""

from __future__ import annotations

import json

import pytest

from scripts.run_dsh3_28_typed_operator_policy import (
    CAP2_MANIFEST,
    _select_source_records,
    main,
)


def test_record_slice_uses_first_source_order_duplicate(tmp_path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"id": "target", "prompt": "first", "openui": "Button()"},
                {"id": "other", "prompt": "other", "openui": "Stack([])"},
                {"id": "target", "prompt": "second", "openui": "Text()"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    selected = _select_source_records(source, "target")

    assert [record.prompt for record in selected] == ["first"]
    with pytest.raises(ValueError, match="does not contain record"):
        _select_source_records(source, "missing")


def test_duplicate_head_family_is_rejected_before_matrix_execution(tmp_path) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--output-dir",
                str(tmp_path / "evidence"),
                "--corpus-work-dir",
                str(tmp_path / "corpus"),
                "--head-families",
                "local_flat,local_flat",
            ]
        )


def test_cap2_mode_is_bound_to_the_current_immutable_fixture() -> None:
    manifest = json.loads(CAP2_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["suite_version"] == "cap2_operator_v2"
    assert manifest["generation"]["max_combinations_per_operator"] == 32
