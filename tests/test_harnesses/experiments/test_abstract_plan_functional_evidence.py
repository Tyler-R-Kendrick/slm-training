from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.abstract_plan_functional_evidence import (
    ARMS,
    PATHS,
    AbstractPlanProtocolV1,
    FunctionalVerdict,
    GATES,
    PlanEvidenceRow,
    build_campaign,
    locked_shard_ids,
    load_locked_protocol,
    summarize_functional_evidence,
)


def _rows() -> list[PlanEvidenceRow]:
    rows = []
    for record in ("a", "b"):
        for arm in ARMS:
            for path in PATHS:
                value = 1.0 if arm == "learned_abstract_plan" else 0.0
                rows.append(
                    PlanEvidenceRow(
                        record_id=record,
                        group_id="group",
                        arm=arm,
                        path=path,
                        plan_tokens=None if arm == "no_plan" else (1, 2),
                        total_tokens=2,
                        latency_ms=1.0,
                        metrics={
                            "binding_aware_meaningful_v2": value,
                            "binder_reference_f1": value,
                            "canonical_ast_match": value,
                        },
                        gates={gate: True for gate in GATES},
                        result_ref="fixture://row",
                        plan_length=0 if arm == "no_plan" else 2,
                        binder_count=1,
                        reference_diameter=1,
                        factor="fixture",
                        pair_manifest_sha256="a" * 64
                        if arm == "shuffle_between_examples"
                        else None,
                    )
                )
    return rows


def test_unavailable_learned_arm_fails_closed_without_proxy_claim() -> None:
    report = summarize_functional_evidence(AbstractPlanProtocolV1("a" * 64, ("a", "b")), _rows(), learned_available=False)
    assert report["verdict"] == FunctionalVerdict.UNAVAILABLE.value
    assert report["ship_eligible"] is False


def test_requires_every_control_path_and_gate_vector() -> None:
    with pytest.raises(ValueError, match="every arm/path"):
        summarize_functional_evidence(AbstractPlanProtocolV1("a" * 64, ("a", "b")), _rows()[:-1], learned_available=True)


def test_campaign_locks_all_controls_and_never_promotes_raw_path() -> None:
    campaign = build_campaign(AbstractPlanProtocolV1("a" * 64, ("a", "b")))
    assert {arm.arm_id for arm in campaign.arms} == set(ARMS)
    raw = next(row for row in _rows() if row.path == "raw")
    with pytest.raises(ValueError, match="diagnostic-only"):
        PlanEvidenceRow(
            **{**raw.__dict__, "promotion_eligible": True}
        ).validate()


def test_locked_protocol_has_certified_heldout_records() -> None:
    protocol = load_locked_protocol()
    assert protocol.manifest_sha256 == "b4ad49cf1b73ad50528709daaad53dbf4846036c9dea787f1c2017c16e0a2d48"
    assert len(protocol.record_ids) == 226


def test_locked_shards_are_deterministic_and_exactly_partition_the_protocol() -> None:
    protocol = AbstractPlanProtocolV1("a" * 64, tuple(str(index) for index in range(11)))
    shards = [
        locked_shard_ids(protocol, shard_index=index, shard_count=3)
        for index in range(3)
    ]
    assert shards == [
        locked_shard_ids(protocol, shard_index=index, shard_count=3)
        for index in range(3)
    ]
    assert set().union(*(set(shard) for shard in shards)) == set(protocol.record_ids)
    assert sum(len(shard) for shard in shards) == len(protocol.record_ids)


def test_f1_can_establish_a_control_effect_when_meaning_is_flat() -> None:
    rows = _rows()
    changed = []
    for row in rows:
        metrics = dict(row.metrics)
        metrics["binding_aware_meaningful_v2"] = 0.0
        metrics["binder_reference_f1"] = 1.0 if row.arm == "learned_abstract_plan" else 0.0
        changed.append(PlanEvidenceRow(**{**row.__dict__, "metrics": metrics}))
    report = summarize_functional_evidence(
        AbstractPlanProtocolV1("a" * 64, ("a", "b")), changed, learned_available=True
    )
    assert report["verdict"] == FunctionalVerdict.FUNCTIONAL.value
