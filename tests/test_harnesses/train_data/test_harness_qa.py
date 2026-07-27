"""DSH1-08 (SLM-360) acceptance: capped identity, verified suffix completion."""

from __future__ import annotations

import pytest

from slm_training.dsl.harness_dsl import (
    HarnessOperation,
    parse_harness_task,
)
from slm_training.dsl.parser import ParseError, validate
from slm_training.harnesses.train_data.harness_qa import (
    AntiIdentityNegative,
    accepted_completion_outputs,
    atomic_form_coverage_violations,
    build_accepted_completion_set,
    build_complete_suffix_records,
    cap_identity_records,
    complete_suffix_record,
    identity_row_counts,
    matched_structure_permuted_markers_negative,
    same_symbols_different_structure_negative,
    statement_prefix_boundaries,
)
from slm_training.harnesses.train_data.scope_corpus import (
    ScopeCorpusConfig,
    build_scope_corpus,
)

SOURCE = (
    'root = Stack([hero, cta], "column")\n'
    'hero = TextContent(":hero.title")\n'
    'cta = Button(":cta.label", true)'
)


@pytest.fixture(scope="module")
def canonical() -> str:
    return validate(SOURCE).serialized


@pytest.fixture(scope="module")
def corpus():
    return build_scope_corpus(
        root_id="root1",
        openui=SOURCE,
        split_group_id="root1",
        program_family_id="fam:root1",
        lineage_id="root1",
        config=ScopeCorpusConfig(),
    )


# --------------------------------------------------------------------------- #
# COMPLETE_SUFFIX
# --------------------------------------------------------------------------- #


def test_statement_prefix_boundaries_are_every_interior_line(canonical) -> None:
    assert statement_prefix_boundaries(canonical) == (1, 2)


def test_every_boundary_prefix_and_suffix_independently_validate(canonical) -> None:
    for cut in statement_prefix_boundaries(canonical):
        completion_set = build_accepted_completion_set(canonical, cut=cut)
        validate(completion_set.prefix)
        for member in completion_set.accepted:
            validate(member)
            validate(f"{completion_set.prefix}\n{member}")


def test_prefix_plus_canonical_reproduces_the_full_document(canonical) -> None:
    for cut in statement_prefix_boundaries(canonical):
        completion_set = build_accepted_completion_set(canonical, cut=cut)
        assert f"{completion_set.prefix}\n{completion_set.canonical}" == canonical


def test_illegal_cut_raises(canonical) -> None:
    with pytest.raises(ValueError):
        build_accepted_completion_set(canonical, cut=0)
    with pytest.raises(ValueError):
        build_accepted_completion_set(canonical, cut=len(canonical.splitlines()))


def test_unrelated_peer_suffix_is_dropped_never_accepted_unverified(canonical) -> None:
    completion_set = build_accepted_completion_set(
        canonical, cut=1, peer_suffixes=("not { valid at all",)
    )
    assert "not { valid at all" not in completion_set.accepted


def test_legally_continuing_peer_suffix_is_accepted(canonical) -> None:
    # A distinct but independently-valid continuation of the same prefix:
    # same binder names ``root`` needs (hero, cta), different content.
    peer = 'hero = TextContent(":alt.hero")\ncta = Button(":alt.cta", false)'
    completion_set = build_accepted_completion_set(
        canonical, cut=1, peer_suffixes=(peer,)
    )
    assert peer in completion_set.accepted


def test_peer_suffix_that_parses_but_silently_drops_content_is_rejected(
    canonical,
) -> None:
    # This peer *parses* without error -- an unresolved reference is pruned,
    # not rejected -- but everything the peer added is silently discarded
    # once joined with the prefix (``root`` never references zzz/info, so
    # they're dead bindings). A weaker "doesn't raise" check would wrongly
    # accept it; only the byte-exact round-trip requirement catches this.
    peer = 'zzz = Card(":zzz.title")\ninfo = Card(":info.title")'
    completion_set = build_accepted_completion_set(
        canonical, cut=1, peer_suffixes=(peer,)
    )
    assert peer not in completion_set.accepted


def test_accepted_completion_outputs_excludes_only_the_canonical_member(
    canonical,
) -> None:
    peer = 'hero = TextContent(":alt.hero")\ncta = Button(":alt.cta", false)'
    completion_set = build_accepted_completion_set(
        canonical, cut=1, peer_suffixes=(peer,)
    )
    assert len(completion_set.accepted) >= 2
    outputs = accepted_completion_outputs(completion_set)
    assert completion_set.canonical not in {target.text for target in outputs}
    assert {target.text for target in outputs} == set(completion_set.accepted) - {
        completion_set.canonical
    }


def test_complete_suffix_task_is_harness_dsl_and_round_trips(canonical) -> None:
    completion_set = build_accepted_completion_set(canonical, cut=1)
    record = complete_suffix_record(completion_set, root_id="root1")
    task = parse_harness_task(record.prompt)
    assert task.operation is HarnessOperation.COMPLETE_SUFFIX
    assert task.payload == completion_set.prefix
    assert record.openui == completion_set.canonical


def test_complete_suffix_target_replays_to_a_verified_terminal(canonical) -> None:
    for record in build_complete_suffix_records(canonical, root_id="root1"):
        task = parse_harness_task(record.prompt)
        # Every accepted output -- canonical and alternates -- verifies against
        # the exact same prefix the model was shown, never a stronger claim.
        for target in record.output_targets:
            validate(f"{task.payload}\n{target.text}")


def test_complete_suffix_accepted_outputs_never_include_the_canonical_twice(
    canonical,
) -> None:
    for record in build_complete_suffix_records(canonical, root_id="root1"):
        alt_texts = [target.text for target in record.accepted_outputs]
        assert record.openui not in alt_texts


# --------------------------------------------------------------------------- #
# Identity cap + diagnostic marking
# --------------------------------------------------------------------------- #


def test_identity_rows_obey_the_declared_cap(corpus) -> None:
    records, _ = corpus
    capped = cap_identity_records(records, cap=1)
    counts = identity_row_counts(capped)
    assert counts  # at least one identity family present
    assert all(count <= 1 for count in counts.values())


def test_capped_identity_rows_are_marked_diagnostic(corpus) -> None:
    records, _ = corpus
    capped = cap_identity_records(records, cap=1)
    for record in capped:
        harness_meta = record.meta.get("harness_dsl")
        if harness_meta and harness_meta.get("operation") == HarnessOperation.IDENTITY.value:
            assert record.meta.get("diagnostic") is True


def test_non_identity_rows_are_never_marked_diagnostic(corpus) -> None:
    records, _ = corpus
    capped = cap_identity_records(records, cap=1)
    for record in capped:
        harness_meta = record.meta.get("harness_dsl")
        if not harness_meta or harness_meta.get("operation") != HarnessOperation.IDENTITY.value:
            assert "diagnostic" not in record.meta


def test_cap_drops_rows_beyond_the_declared_cap_never_silently_keeps_them(
    corpus,
) -> None:
    records, _ = corpus
    before = identity_row_counts(records)
    assert any(count > 1 for count in before.values())
    capped = cap_identity_records(records, cap=1)
    assert len(capped) < len(records)


def test_negative_cap_rejected() -> None:
    with pytest.raises(ValueError):
        cap_identity_records((), cap=-1)


# --------------------------------------------------------------------------- #
# Atomic-form reuse coverage
# --------------------------------------------------------------------------- #


def test_every_lexical_atomic_form_also_appears_nonidentity(corpus, canonical) -> None:
    records, _ = corpus
    # scope_corpus alone can leave gaps (e.g. no scope_canonical rows for a
    # fixture with nothing to decanonicalize); this producer's own
    # COMPLETE_SUFFIX rows are exactly what DSH1-08 adds to close them, so
    # the combined corpus -- not scope_corpus in isolation -- is what must
    # satisfy the acceptance criterion.
    combined = list(records) + build_complete_suffix_records(canonical, root_id="root1")
    assert atomic_form_coverage_violations(combined) == ()


def test_coverage_violation_detected_when_a_form_is_identity_only() -> None:
    records, _ = build_scope_corpus(
        root_id="root1",
        openui=SOURCE,
        split_group_id="root1",
        program_family_id="fam:root1",
        lineage_id="root1",
    )
    only_identity = [
        record
        for record in records
        if (record.meta.get("harness_dsl") or {}).get("operation")
        == HarnessOperation.IDENTITY.value
    ]
    assert only_identity
    violations = atomic_form_coverage_violations(only_identity)
    assert violations  # with no non-identity rows at all, every form violates


# --------------------------------------------------------------------------- #
# Anti-identity hard negatives
# --------------------------------------------------------------------------- #


def _identity_rows(records):
    return [
        record
        for record in records
        if (record.meta.get("harness_dsl") or {}).get("operation")
        == HarnessOperation.IDENTITY.value
    ]


def test_same_symbols_different_structure_negative_reuses_the_corruption_oracle(
    corpus,
) -> None:
    records, _ = corpus
    found = False
    for record in _identity_rows(records):
        negative = same_symbols_different_structure_negative(record)
        if negative is None:
            continue
        found = True
        assert isinstance(negative, AntiIdentityNegative)
        assert negative.chosen == record.openui
        assert negative.rejected != negative.chosen
        with pytest.raises(ParseError):
            validate(negative.rejected)
    assert found


def test_matched_structure_permuted_markers_negative_swaps_markers_not_content(
    corpus,
) -> None:
    records, _ = corpus
    found = False
    for record in _identity_rows(records):
        negative = matched_structure_permuted_markers_negative(record)
        if negative is None:
            continue
        found = True
        assert negative.chosen == record.openui
        assert negative.rejected != negative.chosen
        # Only marker tokens were substituted -- statement count and
        # structural punctuation (parens/brackets) are untouched.
        assert negative.rejected.count("\n") == negative.chosen.count("\n")
        assert negative.rejected.count("(") == negative.chosen.count("(")
        assert negative.rejected.count("[") == negative.chosen.count("[")
    assert found


def test_anti_identity_negative_rejects_identical_chosen_and_rejected() -> None:
    with pytest.raises(ValueError):
        AntiIdentityNegative(prompt="p", chosen="x", rejected="x", variant="v")


def test_negative_helpers_reject_non_identity_rows(corpus) -> None:
    records, _ = corpus
    non_identity = next(
        record
        for record in records
        if (record.meta.get("harness_dsl") or {}).get("operation")
        != HarnessOperation.IDENTITY.value
    )
    with pytest.raises(ValueError):
        same_symbols_different_structure_negative(non_identity)
    with pytest.raises(ValueError):
        matched_structure_permuted_markers_negative(non_identity)
