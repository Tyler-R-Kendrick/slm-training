"""SGS-009 (SLM-456): mechanism disposition and supersession report tests.

Covers the issue's fixture-matrix validation ask directly: known positive
(measured, adopted), null (fixture, retained), and blocked mechanisms, plus
stale-evidence supersession and missing-evidence downgrade enforcement.
"""

from __future__ import annotations

import dataclasses

import pytest

from tests.casefiles import case_values

from slm_training.harnesses.experiments.mechanism_disposition_report import (
    RECORD_SCHEMA,
    REPORT_SCHEMA,
    SUPERSESSION_SCHEMA,
    Disposition,
    MechanismDispositionReportV1,
    SupersessionEntryV1,
    append_supersession,
    build_disposition_report,
    build_mechanism_disposition_record,
    build_supersession_entry,
    disposition_record_integrity_ok,
    disposition_report_integrity_ok,
)


def _measured_record(**overrides: object) -> object:
    fields = dict(
        mechanism_id="m_measured",
        owner_module="src/slm_training/example.py",
        owner_version="v1",
        default_state="on",
        authority_class="advisory-learned",
        evidence_cutoff="deadbeef",
        evidence_class="measured",
        disposition=Disposition.ADOPT_OPTIONAL,
        rationale="measured gain on held-out set",
        activation_evidence="MechanismActivationV1: invoked=12, changed_top_choice=5",
    )
    fields.update(overrides)
    return build_mechanism_disposition_record(**fields)


def _fixture_record(**overrides: object) -> object:
    fields = dict(
        mechanism_id="m_fixture",
        owner_module="src/slm_training/example.py",
        owner_version="v1",
        default_state="off",
        authority_class="evaluation-only",
        evidence_cutoff="cafef00d",
        evidence_class="fixture",
        disposition=Disposition.RETAIN_DIAGNOSTIC,
        rationale="wiring-only fixture, no production evidence",
    )
    fields.update(overrides)
    return build_mechanism_disposition_record(**fields)


def _blocked_record(**overrides: object) -> object:
    fields = dict(
        mechanism_id="m_blocked",
        owner_module="src/slm_training/example.py",
        owner_version="v0",
        default_state="blocked",
        authority_class="evaluation-only",
        evidence_cutoff="",
        evidence_class="blocked",
        disposition=Disposition.BLOCKED,
        rationale="upstream dependency not implemented",
        external_dependencies=("SLM-999",),
    )
    fields.update(overrides)
    return build_mechanism_disposition_record(**fields)


# ---------------------------------------------------------------------------
# AC: wired-but-unmeasured mechanisms cannot be represented as adopted
# ---------------------------------------------------------------------------


def test_adopted_disposition_requires_measured_evidence() -> None:
    with pytest.raises(ValueError, match="requires evidence_class"):
        build_mechanism_disposition_record(
            mechanism_id="m",
            owner_module="x.py",
            owner_version="v1",
            default_state="off",
            authority_class="advisory-learned",
            evidence_cutoff="abc",
            evidence_class="fixture",
            disposition=Disposition.ADOPT_PRIMARY,
            rationale="premature",
        )


def test_adopted_disposition_requires_activation_evidence() -> None:
    with pytest.raises(ValueError, match="requires non-empty activation_evidence"):
        build_mechanism_disposition_record(
            mechanism_id="m",
            owner_module="x.py",
            owner_version="v1",
            default_state="off",
            authority_class="advisory-learned",
            evidence_cutoff="abc",
            evidence_class="measured",
            disposition=Disposition.ADOPT_OPTIONAL,
            rationale="measured but no activation evidence cited",
        )


def test_measured_evidence_with_activation_allows_adoption() -> None:
    record = _measured_record()
    assert record.disposition == Disposition.ADOPT_OPTIONAL


# ---------------------------------------------------------------------------
# AC: fixture-only results cannot promote defaults/checkpoints
# ---------------------------------------------------------------------------


def test_fixture_evidence_cannot_set_default_state_on() -> None:
    with pytest.raises(ValueError, match="cannot set default_state='on'"):
        build_mechanism_disposition_record(
            mechanism_id="m",
            owner_module="x.py",
            owner_version="v1",
            default_state="on",
            authority_class="evaluation-only",
            evidence_cutoff="abc",
            evidence_class="fixture",
            disposition=Disposition.RETAIN_DIAGNOSTIC,
            rationale="fixture only",
        )


def test_missing_evidence_forces_a_downgraded_disposition() -> None:
    """Fixture-class evidence is only ever usable with a non-adopted
    disposition -- the downgrade path (e.g. inconclusive) stays available."""
    record = build_mechanism_disposition_record(
        mechanism_id="m",
        owner_module="x.py",
        owner_version="v1",
        default_state="off",
        authority_class="evaluation-only",
        evidence_cutoff="abc",
        evidence_class="fixture",
        disposition=Disposition.INCONCLUSIVE,
        rationale="insufficient evidence to classify",
    )
    assert record.disposition == Disposition.INCONCLUSIVE


@pytest.mark.parametrize(
    "overrides,match",
    case_values(__file__, "test_rejects_unknown_enum_like_values"),
)
def test_rejects_unknown_enum_like_values(overrides: dict, match: str) -> None:
    fields = dict(
        mechanism_id="m",
        owner_module="x.py",
        owner_version="v1",
        default_state="off",
        authority_class="evaluation-only",
        evidence_cutoff="abc",
        evidence_class="fixture",
        disposition=Disposition.RETAIN_DIAGNOSTIC,
        rationale="r",
    )
    fields.update(overrides)
    with pytest.raises(ValueError, match=match):
        build_mechanism_disposition_record(**fields)


def test_rejects_mechanism_id_required() -> None:
    with pytest.raises(ValueError, match="mechanism_id is required"):
        build_mechanism_disposition_record(
            mechanism_id="",
            owner_module="x.py",
            owner_version="v1",
            default_state="off",
            authority_class="evaluation-only",
            evidence_cutoff="abc",
            evidence_class="fixture",
            disposition=Disposition.RETAIN_DIAGNOSTIC,
            rationale="r",
        )


# ---------------------------------------------------------------------------
# Record: frozen, deterministic, tamper-evident
# ---------------------------------------------------------------------------


def test_record_is_frozen_deterministic_and_replayable() -> None:
    first = _measured_record()
    second = _measured_record()
    assert first == second
    assert first.record_hash == second.record_hash
    assert disposition_record_integrity_ok(first)

    with pytest.raises(dataclasses.FrozenInstanceError):
        first.default_state = "off"  # type: ignore[misc]

    from slm_training.harnesses.experiments.mechanism_disposition_report import (
        MechanismDispositionRecordV1,
    )

    assert MechanismDispositionRecordV1.from_dict(first.to_dict()) == first


def test_record_tamper_and_schema_rejection() -> None:
    from slm_training.harnesses.experiments.mechanism_disposition_report import (
        MechanismDispositionRecordV1,
    )

    record = _measured_record()
    tampered = record.to_dict()
    tampered["disposition"] = "reject"
    with pytest.raises(ValueError, match="hash mismatch"):
        MechanismDispositionRecordV1.from_dict(tampered)

    wrong_schema = record.to_dict()
    wrong_schema["schema_version"] = f"{RECORD_SCHEMA}-bad"
    with pytest.raises(ValueError, match="unsupported mechanism disposition record schema"):
        MechanismDispositionRecordV1.from_dict(wrong_schema)


# ---------------------------------------------------------------------------
# Supersession: append-only, same-mechanism-linked, machine-readable
# ---------------------------------------------------------------------------


def test_supersession_requires_the_same_mechanism_id() -> None:
    a = _fixture_record(mechanism_id="m_a")
    b = _fixture_record(mechanism_id="m_b")
    with pytest.raises(ValueError, match="same mechanism_id"):
        build_supersession_entry(superseded=a, superseding=b, reason="unrelated")


def test_supersession_requires_a_reason() -> None:
    a = _fixture_record(evidence_cutoff="old")
    b = _fixture_record(evidence_cutoff="new")
    with pytest.raises(ValueError, match="reason is required"):
        build_supersession_entry(superseded=a, superseding=b, reason="")


def test_append_supersession_is_pure_and_append_only() -> None:
    a = _fixture_record(evidence_cutoff="old")
    b = _fixture_record(evidence_cutoff="new")
    entry = build_supersession_entry(
        superseded=a, superseding=b, reason="newer evidence available"
    )
    original: tuple[SupersessionEntryV1, ...] = ()
    updated = append_supersession(original, entry)

    assert original == ()  # never mutated
    assert updated == (entry,)


def test_supersession_entry_tamper_and_schema_rejection() -> None:
    a = _fixture_record(evidence_cutoff="old")
    b = _fixture_record(evidence_cutoff="new")
    entry = build_supersession_entry(superseded=a, superseding=b, reason="r")

    tampered = entry.to_dict()
    tampered["reason"] = "different reason"
    with pytest.raises(ValueError, match="hash mismatch"):
        SupersessionEntryV1.from_dict(tampered)

    wrong_schema = entry.to_dict()
    wrong_schema["schema_version"] = f"{SUPERSESSION_SCHEMA}-bad"
    with pytest.raises(ValueError, match="unsupported supersession entry schema"):
        SupersessionEntryV1.from_dict(wrong_schema)


def test_stale_document_is_linked_not_deleted() -> None:
    old = _fixture_record(mechanism_id="m_stale", evidence_cutoff="old_sha")
    new = _fixture_record(mechanism_id="m_stale", evidence_cutoff="new_sha")
    entry = build_supersession_entry(
        superseded=old, superseding=new, reason="re-measured with more data"
    )
    report = build_disposition_report((old, new), (entry,))

    # The stale record is still present, never rewritten or removed.
    assert old in report.records
    assert new in report.records
    assert entry in report.supersessions
    markdown = report.to_markdown()
    assert "old_sha" in markdown
    assert "new_sha" in markdown
    assert "re-measured with more data" in markdown


# ---------------------------------------------------------------------------
# Report: negative/null results always visible; deterministic; replayable
# ---------------------------------------------------------------------------


def test_report_never_drops_negative_or_null_results() -> None:
    measured = _measured_record()
    fixture = _fixture_record()
    blocked = _blocked_record()
    rejected = build_mechanism_disposition_record(
        mechanism_id="m_rejected",
        owner_module="x.py",
        owner_version="v1",
        default_state="off",
        authority_class="evaluation-only",
        evidence_cutoff="abc",
        evidence_class="rejected",
        disposition=Disposition.REJECT,
        rationale="regressed the held-out benchmark",
    )

    report = build_disposition_report((measured, fixture, blocked, rejected))
    assert report.records == (measured, fixture, blocked, rejected)

    markdown = report.to_markdown()
    for record in (measured, fixture, blocked, rejected):
        assert record.mechanism_id in markdown
        assert record.disposition.value in markdown


def test_known_positive_null_blocked_fixture_matrix() -> None:
    """The issue's own validation ask: a small matrix of known positive
    (measured/adopted), null (fixture/retained), and blocked mechanisms."""
    positive = _measured_record()
    null = _fixture_record()
    blocked = _blocked_record()

    report = build_disposition_report((positive, null, blocked))
    by_id = {r.mechanism_id: r for r in report.records}

    assert by_id["m_measured"].disposition == Disposition.ADOPT_OPTIONAL
    assert by_id["m_fixture"].disposition == Disposition.RETAIN_DIAGNOSTIC
    assert by_id["m_blocked"].disposition == Disposition.BLOCKED
    assert disposition_report_integrity_ok(report)


def test_report_is_frozen_deterministic_and_replayable() -> None:
    records = (_measured_record(), _fixture_record())
    first = build_disposition_report(records)
    second = build_disposition_report(records)
    assert first == second
    assert first.report_hash == second.report_hash
    assert disposition_report_integrity_ok(first)

    with pytest.raises(dataclasses.FrozenInstanceError):
        first.records = ()  # type: ignore[misc]

    assert MechanismDispositionReportV1.from_dict(first.to_dict()) == first


def test_report_tamper_and_schema_rejection() -> None:
    report = build_disposition_report((_measured_record(),))

    tampered = report.to_dict()
    tampered["records"] = []
    with pytest.raises(ValueError, match="hash mismatch"):
        MechanismDispositionReportV1.from_dict(tampered)

    wrong_schema = report.to_dict()
    wrong_schema["schema_version"] = f"{REPORT_SCHEMA}-bad"
    with pytest.raises(ValueError, match="unsupported mechanism disposition report schema"):
        MechanismDispositionReportV1.from_dict(wrong_schema)


def test_disposition_enum_values_match_the_slm160_precedent() -> None:
    assert {d.value for d in Disposition} == {
        "adopt_primary",
        "adopt_optional",
        "retain_diagnostic",
        "revise_and_retest",
        "reject",
        "blocked",
        "inconclusive",
    }
