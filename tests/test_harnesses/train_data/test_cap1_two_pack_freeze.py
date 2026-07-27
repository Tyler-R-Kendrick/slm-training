"""DSH2-07 (SLM-367): two-pack generic CAP1 freeze + schema-sensitivity suite."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from slm_training.dsl.mini_pack import DSL_ID as MINI_DSL_ID
from slm_training.harnesses.train_data import cap1_two_pack_freeze as m


@pytest.fixture(scope="module")
def suite() -> m.Cap1TwoPackSuiteV1:
    return m.build_cap1_two_pack_suite()


def decide(row: m.SuiteRowV1) -> str:
    """Synthetic schema-grounded decider: abstain without a schema; pick the
    first candidate whose components and closed values the schema licenses."""
    schema = json.loads(row.schema_text) if row.schema_text.strip() else {}
    domains = schema.get("closed_value_domains")
    components = set(schema.get("component_names", ()))
    if not domains and not components:
        return m.ABSTAIN
    for candidate in row.candidates:
        usage = m.closed_value_usage(row.pack_id, candidate)
        used_components = m.component_usage(row.pack_id, candidate)
        if used_components <= components and all(
            value in tuple(domains.get(field, ())) for field, value in usage.items()
        ):
            return candidate
    return m.ABSTAIN


# --------------------------------------------------------------------------
# Structure: strata, disjoint families, ambiguity recording
# --------------------------------------------------------------------------


def test_all_strata_present_per_pack(suite) -> None:
    for pack_id in m.PACK_IDS:
        present = {row.stratum for row in suite.rows if row.pack_id == pack_id}
        assert set(m.STRATA) <= present, (pack_id, set(m.STRATA) - present)


def test_disjoint_root_families(suite) -> None:
    families = {
        pack_id: {row.root_family_id for row in suite.rows if row.pack_id == pack_id}
        for pack_id in m.PACK_IDS
    }
    assert families[m.OPENUI_PACK_ID].isdisjoint(families[MINI_DSL_ID])
    # Two base cases per pack → at least two distinct families per pack.
    for pack_id in m.PACK_IDS:
        assert len(families[pack_id]) >= 2


def test_mini_rows_exercise_generic_pack(suite) -> None:
    mini_rows = suite.rows_for(MINI_DSL_ID)
    assert mini_rows
    assert all(row.schema_text for row in mini_rows)
    projection = json.loads(mini_rows[0].schema_text)
    assert projection["dsl"] == MINI_DSL_ID
    assert projection["closed_value_domains"]["mode"] == ["fast", "safe"]


def test_ambiguity_accepted_set_or_excluded_recording(suite) -> None:
    ambiguous = [row for row in suite.rows if row.stratum == m.STRATUM_AMBIGUITY]
    accepted = [row for row in ambiguous if row.ambiguity_disposition == m.DISPOSITION_ACCEPTED_SET]
    excluded = [row for row in ambiguous if row.ambiguity_disposition == m.DISPOSITION_EXCLUDED]
    assert accepted, "expected accepted-set ambiguous rows"
    assert all(len(row.targets) >= 2 for row in accepted)
    assert all(
        "SLM-344" in str(row.provenance.get("ambiguity_evidence", "")) for row in accepted
    )
    assert excluded, "expected at least one excluded ambiguous row"
    assert all(row.exclusion_reason for row in excluded)
    # Excluded rows are never determinacy-scored.
    adjudication = m.adjudicate_suite(suite, decide)
    assert adjudication.determinacy["calibrated"] is True
    assert adjudication.determinacy["excluded_rows"] == len(excluded)
    assert adjudication.determinacy["determinate_rows"] == sum(
        1 for row in suite.rows if row.ambiguity_disposition == m.DISPOSITION_DETERMINATE
    )


def test_counterfactual_rows_are_one_fact(suite) -> None:
    for row in suite.rows_for(m.OPENUI_PACK_ID, m.STRATUM_COUNTERFACTUAL):
        assert row.provenance["case"]["counterfactual"]["single_dimension_change"] is True
    for row in suite.rows_for(MINI_DSL_ID, m.STRATUM_COUNTERFACTUAL):
        assert row.provenance["case"]["counterfactual"]["single_fact_change"] is True


def test_marker_permutation_rotates_surfaces(suite) -> None:
    for row in suite.rows:
        if row.stratum != m.STRATUM_MARKER_PERMUTATION:
            continue
        permutation = row.provenance["marker_permutation"]
        assert permutation, row.group_id
        original = next(
            r for r in suite.rows if r.group_id == row.group_id and r.stratum == "original"
        )
        assert row.targets[0] != original.targets[0]


# --------------------------------------------------------------------------
# Schema sensitivity + adjudication
# --------------------------------------------------------------------------


def test_relevant_perturbations_change_decisions(suite) -> None:
    report = m.score_schema_sensitivity(suite, decide)
    assert report.relevant_sensitivity == 1.0
    assert report.invariant_rate == 1.0
    assert report.passed is True
    for group in report.per_group.values():
        for stratum in m.RELEVANT_STRATA:
            assert group["strata"][stratum]["changed"] is True
        for stratum in m.INVARIANT_STRATA:
            assert group["strata"][stratum]["changed"] is False


def test_irrelevant_perturbations_approximately_invariant_with_tolerance(suite) -> None:
    flip_row_id = next(
        row.row_id for row in suite.rows if row.stratum == m.STRATUM_SCHEMA_IRRELEVANT
    )

    def jittery_decide(row: m.SuiteRowV1) -> str:
        # One irrelevant-perturbation flip out of many invariant rows.
        if row.row_id == flip_row_id:
            return m.ABSTAIN
        return decide(row)

    strict = m.score_schema_sensitivity(suite, jittery_decide)
    assert strict.invariant_rate < 1.0
    assert strict.passed is False
    tolerant = m.score_schema_sensitivity(suite, jittery_decide, tolerance=0.2)
    assert tolerant.passed is True


def test_constant_decider_fails_sensitivity(suite) -> None:
    report = m.score_schema_sensitivity(suite, lambda row: row.targets[0] if row.targets else m.ABSTAIN)
    # Constant decider never reacts to relevant perturbations.
    assert report.relevant_sensitivity < 1.0
    assert report.passed is False


def test_adjudication_scores(suite) -> None:
    adjudication = m.adjudicate_suite(suite, decide)
    assert adjudication.facts_accuracy == 1.0
    assert adjudication.prompt_invariance_rate == 1.0
    assert adjudication.canonical_accuracy > 0.5  # abstains on contradicted/empty schema
    assert adjudication.cap0_retention["accuracy"] == 1.0


# --------------------------------------------------------------------------
# Gates: CAP0 retention mandatory, determinacy calibration
# --------------------------------------------------------------------------


def test_gates_pass_on_full_suite(suite) -> None:
    gates = m.evaluate_gates(suite, m.adjudicate_suite(suite, decide))
    assert gates.passed is True, gates.failures


def test_suite_without_cap0_retention_fails_gates() -> None:
    suite = m.build_cap1_two_pack_suite(include_cap0=False)
    adjudication = m.adjudicate_suite(suite, decide)
    gates = m.evaluate_gates(suite, adjudication)
    assert gates.passed is False
    assert "cap0_retention_present" in gates.failures
    assert "cap0_retention_perfect" in gates.failures


def test_broken_retention_fails_gates(suite) -> None:
    gates = m.evaluate_gates(suite, m.adjudicate_suite(suite, lambda row: m.ABSTAIN))
    assert gates.passed is False
    assert "cap0_retention_perfect" in gates.failures


# --------------------------------------------------------------------------
# Immutability + freeze
# --------------------------------------------------------------------------


def test_integrity_ok(suite) -> None:
    report = m.verify_suite_integrity(suite)
    assert report.ok is True, report.violations
    assert report.checked_rows == len(suite.rows)
    assert report.manifest_sha256 == suite.suite_sha256


def test_integrity_fail_closed_on_tamper(suite) -> None:
    tampered_row = replace(suite.rows[0], prompt="tampered prompt")
    tampered = replace(suite, rows=(tampered_row,) + suite.rows[1:])
    report = m.verify_suite_integrity(tampered)
    assert report.ok is False
    assert any("tampered" in v or "hash mismatch" in v for v in report.violations)


def test_integrity_fail_closed_on_manifest_swap(suite) -> None:
    other = m.build_cap1_two_pack_suite(include_cap0=False)
    swapped = replace(suite, manifest=other.manifest)
    report = m.verify_suite_integrity(swapped)
    assert report.ok is False


def test_freeze_create_once(suite, tmp_path) -> None:
    path = m.freeze_suite(suite, tmp_path)
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["suite_sha256"] == suite.suite_sha256
    assert payload["manifest"]["manifest_sha256"] == suite.suite_sha256
    # Identical rewrite is a no-op; different (still integrity-valid) content raises.
    assert m.freeze_suite(suite, tmp_path) == path
    other = m.build_cap1_two_pack_suite(seed=1)
    assert other.suite_sha256 != suite.suite_sha256
    with pytest.raises(FileExistsError):
        m.freeze_suite(other, tmp_path)


def test_freeze_refuses_tampered_suite(suite, tmp_path) -> None:
    tampered_row = replace(suite.rows[0], prompt="tampered prompt")
    tampered = replace(suite, rows=(tampered_row,) + suite.rows[1:])
    with pytest.raises(m.TwoPackFreezeError):
        m.freeze_suite(tampered, tmp_path)


def test_suite_cards_provenance(suite) -> None:
    cards = {card.pack_id: card for card in suite.cards}
    assert set(cards) == set(m.PACK_IDS)
    mini_card = cards[MINI_DSL_ID]
    assert mini_card.generation_provenance["completeness"]["complete"] is True
    for card in cards.values():
        assert card.rows_per_stratum
        assert card.generation_provenance["cases"]
        assert "decide_contract" in card.adjudication_provenance


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_build_deterministic(suite) -> None:
    rebuilt = m.build_cap1_two_pack_suite()
    assert rebuilt.suite_sha256 == suite.suite_sha256
    assert rebuilt.to_dict() == suite.to_dict()
