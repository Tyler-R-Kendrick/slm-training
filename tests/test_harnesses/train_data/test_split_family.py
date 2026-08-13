from slm_training.harnesses.train_data.catalog import (
    TRUSTED_ANCHOR_FAMILIES,
    source_anchor_report,
)
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1
from slm_training.dsl.schema import ExampleRecord
import pytest


def test_derivatives_cannot_cross_root_families() -> None:
    policy = RootFamilySplitPolicyV1()
    policy.require_same_family(
        root_family_id="root-a", member_family_ids=("root-a", "root-a")
    )
    with pytest.raises(ValueError, match="inside root family"):
        policy.require_same_family(
            root_family_id="root-a", member_family_ids=("root-a", "root-b")
        )


def test_synthetic_only_is_not_promotion_authoritative() -> None:
    records = [
        ExampleRecord(
            id="s1",
            prompt="need a compact screen",
            openui="root = Stack([x])",
            placeholders=[],
            split="train",
            source="programspec_generated",
            meta={"source_family": "programspec_generated"},
        )
    ]
    report = source_anchor_report(records)
    assert report["deficit"] is True
    assert report["promotion_authoritative"] is False
    assert "rico_real" in TRUSTED_ANCHOR_FAMILIES
