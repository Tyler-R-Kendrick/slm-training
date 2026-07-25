"""SLM-357 (DSH1-05): expanded-equivalent answers + canonical preference."""

from __future__ import annotations

import pytest

from slm_training.dsl.canonicalize import canonical_fingerprint, canonicalize
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.harnesses.train_data.equivalence_transforms import (
    CANONICALIZES_TO,
    EQUIVALENT_TO,
    KIND_MARGINS,
    PACK_APPROVED_TRANSFORMS,
    UNPROVEN_TRANSFORMS,
    ExpansionV1,
    NegativeFixtureV1,
    compare_canonicality,
    materialize_equivalence_class,
    materialize_sft_records,
    prove_equivalence,
)

FIXTURE = (
    'root = Stack([header, cta], "column")\n'
    'header = TextContent(":title", "large-heavy")\n'
    'cta = Button(":cta", "primary")\n'
)
CANONICAL = canonicalize(FIXTURE)
REQUIRED_ROLES = tuple(sorted(extract_placeholders(CANONICAL)))


@pytest.fixture(scope="module")
def equivalence_class():
    return materialize_equivalence_class(FIXTURE)


def test_every_expansion_canonicalizes_to_same_class(equivalence_class):
    assert equivalence_class.expansions, "expected at least one proven expansion"
    kinds = {item.kind for item in equivalence_class.expansions}
    assert kinds == set(KIND_MARGINS), f"missing proven kinds: {set(KIND_MARGINS) - kinds}"
    for item in equivalence_class.expansions:
        assert item.form != equivalence_class.canonical_form
        assert canonicalize(item.form) == equivalence_class.canonical_form
        assert item.canonical_fingerprint == equivalence_class.canonical_fingerprint


def test_pack_approved_transforms_prove_on_fixture():
    for transform in PACK_APPROVED_TRANSFORMS:
        result = prove_equivalence(transform, CANONICAL)
        assert isinstance(result, ExpansionV1), transform.id


def test_unproven_transform_stop_rule(equivalence_class):
    """Order-sensitive rewrite without authority: negative fixture only."""
    negatives = {item.transform_id: item for item in equivalence_class.negative_fixtures}
    for transform in UNPROVEN_TRANSFORMS:
        assert transform.id in negatives
        negative = negatives[transform.id]
        assert negative.counterexample_fingerprint != negative.source_fingerprint
        assert "unproven" in negative.reason or "authority" in negative.reason
    admitted_ids = {item.transform_id for item in equivalence_class.expansions}
    assert not admitted_ids & {t.id for t in UNPROVEN_TRANSFORMS}


def test_order_sensitive_properties_never_reordered_without_authority():
    swapped = CANONICAL.replace("[v0, v1]", "[v1, v0]")
    assert swapped != CANONICAL
    assert canonical_fingerprint(swapped) != canonical_fingerprint(CANONICAL)
    result = prove_equivalence(UNPROVEN_TRANSFORMS[0], CANONICAL)
    assert isinstance(result, NegativeFixtureV1)


def test_incomplete_shorter_never_outranks_complete():
    expanded = materialize_equivalence_class(FIXTURE).expansions[0].form
    shorter_incomplete = (
        'root = Stack([v0], "column")\nv0 = TextContent(":title")'
    )
    assert len(shorter_incomplete) < len(expanded)
    verdict = compare_canonicality(
        expanded, shorter_incomplete, required_roles=REQUIRED_ROLES
    )
    assert verdict.preferred == expanded
    assert verdict.other == shorter_incomplete
    assert "missing required roles" in verdict.completeness_guard
    # Guard also fires when the incomplete side is passed first.
    flipped = compare_canonicality(
        shorter_incomplete, expanded, required_roles=REQUIRED_ROLES
    )
    assert flipped.preferred == expanded


def test_canonical_outranks_expanded_only_within_class():
    expanded = materialize_equivalence_class(FIXTURE).expansions[0].form
    verdict = compare_canonicality(
        CANONICAL, expanded, required_roles=REQUIRED_ROLES
    )
    assert verdict.same_equivalence_class
    assert verdict.relation == "CANONICALLY_PREFERRED_TO"
    assert verdict.preferred == CANONICAL
    assert verdict.preferred_cost <= verdict.other_cost
    # Different classes with complete roles are unordered.
    other = compare_canonicality(
        CANONICAL,
        'root = Stack([v0], "column")\nv0 = TextContent(":title")',
        required_roles=(":title",),
    )
    assert not other.same_equivalence_class
    assert other.relation is None


def test_materialization_canonical_primary_and_pairs(equivalence_class):
    record, pairs, relations = materialize_sft_records(equivalence_class)
    assert record.openui == equivalence_class.canonical_form
    accepted = [target.text for target in record.accepted_outputs]
    assert accepted == [item.form for item in equivalence_class.expansions]
    assert len(pairs) == len(equivalence_class.expansions)
    for pair, item in zip(pairs, equivalence_class.expansions):
        # Preference pairs carry both forms.
        assert pair.chosen == equivalence_class.canonical_form
        assert pair.rejected == item.form
        assert pair.chosen_score == 1.0
        assert pair.rejected_score == pytest.approx(1.0 - item.margin)
        assert pair.meta["transform_version"] == item.transform_version
        assert pair.meta["margin"] == item.margin
        assert pair.meta["provenance"]
    relation_kinds = [relation.kind for relation in relations]
    assert relation_kinds.count(EQUIVALENT_TO) == len(equivalence_class.expansions)
    assert relation_kinds.count(CANONICALIZES_TO) == len(equivalence_class.expansions)


def test_expanded_preferred_when_task_asks_for_expansion(equivalence_class):
    record, pairs, _ = materialize_sft_records(
        equivalence_class, prefer_expanded=True
    )
    first = equivalence_class.expansions[0]
    assert record.openui == first.form
    assert [target.text for target in record.accepted_outputs] == [
        equivalence_class.canonical_form
    ]
    assert record.meta["prefer_expanded"] is True
    for pair in pairs:
        assert pair.chosen != equivalence_class.canonical_form
        assert pair.rejected == equivalence_class.canonical_form
        assert pair.chosen_score == 1.0
    with pytest.raises(ValueError, match="prefer_expanded"):
        empty = type(equivalence_class)(
            canonical_form=equivalence_class.canonical_form,
            canonical_fingerprint=equivalence_class.canonical_fingerprint,
            expansions=(),
            negative_fixtures=(),
        )
        materialize_sft_records(empty, prefer_expanded=True)


def test_margins_and_versions_deterministic_and_provenance_stamped():
    first = materialize_equivalence_class(FIXTURE)
    second = materialize_equivalence_class(FIXTURE)
    assert first.to_dict() == second.to_dict()
    record_a, pairs_a, _ = materialize_sft_records(first)
    record_b, pairs_b, _ = materialize_sft_records(second)
    assert record_a.to_dict() == record_b.to_dict()
    assert [pair.to_dict() for pair in pairs_a] == [
        pair.to_dict() for pair in pairs_b
    ]
    for item in first.expansions:
        assert item.margin == KIND_MARGINS[item.kind]
        assert item.transform_version == "v1"
        assert item.provenance["transform"]["id"] == item.transform_id
        assert item.provenance["proof"]
