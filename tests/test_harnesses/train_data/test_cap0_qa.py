"""DSH1-08 (SLM-360): CAP0 QA views — identity cap, accepted sets, hard negatives."""

from __future__ import annotations

import dataclasses

import pytest

from slm_training.dsl.grammar_capabilities import GrammarCapabilityAdapterV1
from slm_training.dsl.harness_dsl import (
    HarnessOperation,
    parse_harness_task,
)
from slm_training.dsl.pack import get_pack
from slm_training.harnesses.train_data.cap0_qa import (
    SCORING_ACCEPTED_SET,
    Cap0QAAuditError,
    audit_no_single_gold_where_multi,
    generate_cap0_qa,
    identity_baseline_matches,
    identity_rows_within_cap,
    require_atomic_coverage,
    require_verified_terminals,
)
from slm_training.harnesses.train_data.equivalence_transforms import (
    CANONICALLY_PREFERRED_TO,
)
from slm_training.harnesses.train_data.partial_states import PartialStateClassV1

ANSWERS = (
    'item = TextContent(":w.text")\nroot = Stack([item], "column")',
    'item = TextContent(":w.text")\n$s = item\nroot = Stack([item], "column")',
    'hero = TextContent(":smoke.hero.title")\nroot = Stack([hero], "column")',
    'root = Stack([hero, cta], "column")\n'
    'hero = TextContent(":smoke.hero.title")\n'
    'cta = Button(":cta.label", "primary")',
    "root = Separator()",
)


@pytest.fixture(scope="module")
def adapter() -> GrammarCapabilityAdapterV1:
    return GrammarCapabilityAdapterV1(get_pack("openui"))


@pytest.fixture(scope="module")
def dataset(adapter: GrammarCapabilityAdapterV1):
    return generate_cap0_qa(ANSWERS, adapter=adapter, identity_cap=0.1, seed=7, dsl="openui")


def test_rows_are_canonical_harness_dsl(dataset) -> None:
    assert dataset.rows
    expected_ops = {
        "identity": HarnessOperation.IDENTITY,
        "canonicalize": HarnessOperation.CANONICALIZE,
        "complete_suffix": HarnessOperation.COMPLETE_SUFFIX,
    }
    assert {row.kind for row in dataset.rows} == set(expected_ops)
    for row in dataset.rows:
        parsed = parse_harness_task(row.prompt)
        assert parsed == row.task
        assert parsed.operation is expected_ops[row.kind]
        assert row.scoring == SCORING_ACCEPTED_SET


def test_identity_cap_enforced_and_diagnostic(dataset) -> None:
    identity = [row for row in dataset.rows if row.kind == "identity"]
    assert identity, "expected some diagnostic identity rows"
    assert all(row.diagnostic for row in identity)
    assert identity_rows_within_cap(dataset)
    assert dataset.stats["identity_fraction"] <= dataset.identity_cap
    # Nonidentity rows are never diagnostic.
    assert all(
        not row.diagnostic for row in dataset.rows if row.kind != "identity"
    )


def test_identity_cap_rejects_over_identity_mix(dataset) -> None:
    # A dataset whose rows are mostly identity must fail the cap guard.
    identity_rows = tuple(row for row in dataset.rows if row.kind == "identity")
    nonidentity = [row for row in dataset.rows if row.kind != "identity"]
    inflated = dataclasses.replace(
        dataset, rows=(*identity_rows, *identity_rows, *nonidentity[:1])
    )
    assert not identity_rows_within_cap(inflated)


def test_every_atomic_production_in_nonidentity_context(dataset) -> None:
    assert dataset.uncovered_productions == ()
    require_atomic_coverage(dataset)  # does not raise
    gaps = dataclasses.replace(dataset, uncovered_productions=("start->phantom",))
    with pytest.raises(Cap0QAAuditError):
        require_atomic_coverage(gaps)


def test_hard_negatives_both_kinds(dataset) -> None:
    kinds = {item.negative_kind for item in dataset.hard_negatives}
    assert kinds == {
        "same_symbols_different_structure",
        "matched_structure_permuted_markers",
    }
    for item in dataset.hard_negatives:
        assert item.verified
        assert item.negative_form != item.source_answer
        # Verified near-miss: parses but lands in a different equivalence class.
        assert item.negative_fingerprint != item.source_fingerprint
    same_symbols = next(
        item
        for item in dataset.hard_negatives
        if item.negative_kind == "same_symbols_different_structure"
    )
    import re

    token_re = re.compile(r"[A-Za-z_$:@][A-Za-z0-9_.$:@]*|\S")
    assert sorted(token_re.findall(same_symbols.negative_form)) == sorted(
        token_re.findall(same_symbols.source_answer)
    )
    permuted = next(
        item
        for item in dataset.hard_negatives
        if item.negative_kind == "matched_structure_permuted_markers"
    )
    # Markers permuted, never dropped or invented.
    from slm_training.dsl.placeholders import extract_placeholders

    assert sorted(extract_placeholders(permuted.negative_form)) == sorted(
        extract_placeholders(permuted.source_answer)
    )


def test_accepted_set_persisted_with_separate_canonical_preference(dataset) -> None:
    ambiguous = [row for row in dataset.rows if len(row.accepted_completions) > 1]
    assert ambiguous, "expected at least one ambiguous prefix"
    for row in ambiguous:
        # ALL verified accepted completions persisted, none labeled as the gold.
        assert row.completion is None
        assert row.canonical_preference is not None
        assert row.canonical_preference in {
            item.source for item in row.accepted_completions
        }
        assert row.provenance["ambiguous"] is True
    audit_no_single_gold_where_multi(dataset)  # does not raise


def test_no_single_gold_where_multi_audit_rejects(dataset) -> None:
    ambiguous = next(
        row for row in dataset.rows if len(row.accepted_completions) > 1
    )
    mislabeled = dataclasses.replace(
        ambiguous, completion=ambiguous.accepted_completions[0].source
    )
    bad = dataclasses.replace(
        dataset,
        rows=tuple(mislabeled if row is ambiguous else row for row in dataset.rows),
    )
    with pytest.raises(Cap0QAAuditError):
        audit_no_single_gold_where_multi(bad)


def test_accepted_completions_replay_to_verified_terminals(dataset) -> None:
    require_verified_terminals(dataset)  # does not raise
    completions = [
        item for row in dataset.rows for item in row.accepted_completions
    ]
    assert completions
    for item in completions:
        assert item.terminal_verified
        assert item.terminal_state_class == PartialStateClassV1.COMPLETE.value
        assert item.source  # full terminal source persisted
    broken = dataclasses.replace(
        completions[0], terminal_verified=False
    )
    row = next(row for row in dataset.rows if row.accepted_completions)
    patched = dataclasses.replace(
        row,
        accepted_completions=(broken, *row.accepted_completions[1:]),
    )
    bad = dataclasses.replace(
        dataset,
        rows=tuple(patched if row is r else r for r in dataset.rows),
    )
    with pytest.raises(Cap0QAAuditError):
        require_verified_terminals(bad)


def test_canonicalize_rows_use_dsh105_preference(dataset) -> None:
    rows = [row for row in dataset.rows if row.kind == "canonicalize"]
    assert rows
    for row in rows:
        assert row.completion is not None
        assert row.completion != row.task.payload  # expanded -> canonical
        assert CANONICALLY_PREFERRED_TO in row.relations
        assert row.provenance["transform_id"]
        assert row.provenance["transform_version"]


def test_identity_baseline_stop_rule(dataset) -> None:
    verdict = identity_baseline_matches(dataset)
    assert verdict["matches"] is False
    assert verdict["do_not_train"] is False
    assert verdict["non_diagnostic_rows"] > 0
    # A suite solvable by copying trips the stop rule.
    copyable = [
        dataclasses.replace(row, completion=row.task.payload)
        for row in dataset.rows
        if not row.diagnostic and row.completion is not None
    ]
    identity_only_suite = dataclasses.replace(dataset, rows=tuple(copyable))
    verdict = identity_baseline_matches(identity_only_suite)
    assert verdict["matches"] is True
    assert verdict["do_not_train"] is True


def test_non_verified_answer_rejected(adapter: GrammarCapabilityAdapterV1) -> None:
    with pytest.raises(Cap0QAAuditError):
        generate_cap0_qa(
            ["TextConten", *ANSWERS], adapter=adapter, dsl="openui"
        )


def test_determinism(adapter: GrammarCapabilityAdapterV1) -> None:
    first = generate_cap0_qa(ANSWERS, adapter=adapter, identity_cap=0.1, seed=7, dsl="openui")
    second = generate_cap0_qa(ANSWERS, adapter=adapter, identity_cap=0.1, seed=7, dsl="openui")
    assert first == second
    assert first.to_dict() == second.to_dict()
