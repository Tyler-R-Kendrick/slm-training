"""Awwwards fixture → ExampleRecord tests."""

from __future__ import annotations

import pytest

from slm_training.data.awwwards import AwwwardsConfig, build_awwwards_records, site_to_record
from slm_training.dsl import bridge_available, validate


@pytest.mark.skipif(not bridge_available(), reason="openui bridge missing")
def test_awwwards_fixture_records_validate() -> None:
    records = build_awwwards_records(AwwwardsConfig(max_sites=5))
    assert len(records) >= 3
    for record in records:
        validate(record.openui)
        assert record.source == "awwwards"
        assert record.design_md


def test_site_to_record_form_tag() -> None:
    record = site_to_record(
        {
            "id": "t1",
            "title": "Formy",
            "tags": ["form", "signup"],
            "description": "signup",
        },
        attach_design_md=False,
    )
    assert "Input(" in record.openui
    assert record.design_md is None
