"""SLM-434 (LAR0-07): scoped port of the SLM-308 bounded distance oracle.

Ported from the never-merged SLM-308 fork (commit ``ae5448c5``) — only the
oracle module itself (``distance_to_target`` and its EXACT/BOUNDED/UNKNOWN
classification), not SLM-308's pairwise progress margin loss or SLM-310's
corruption-distribution knob, which the SLM-317/SLM-431 harness never
references. See ``test_models/test_tree_edit_diffusion.py`` for the
``value_label_mode``/``stop_slot_accounting`` config-knob tests.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import slm_training.harnesses.experiments.slm308_distance_oracle as oracle
from slm_training.harnesses.experiments.slm308_distance_oracle import (
    DistanceKind,
    clear_caches,
    distance_to_target,
    effective_distance,
)
from slm_training.models.tree_edit_diffusion import TreeEditSpace, parse_statements

SEED = 'root = Stack([], "column")'
GOLD = 'root = Stack([cta], "column")\ncta = Button(":cta.label")'


@pytest.fixture()
def space():
    return TreeEditSpace()


@pytest.fixture(autouse=True)
def _clean_oracle_caches():
    clear_caches()
    yield
    clear_caches()


def test_exact_distance_small_cases(space) -> None:
    seed = parse_statements(SEED)
    gold = parse_statements(GOLD)
    label = distance_to_target(
        seed, gold, space=space, inventory=[":cta.label"], node_budget=4
    )
    assert label.kind is DistanceKind.EXACT
    assert label.distance == 1
    assert label.value_target(8) == pytest.approx(1.0 - 1 / 8)
    ident = distance_to_target(gold, gold, space=space, inventory=[":x"])
    assert ident.kind is DistanceKind.EXACT and ident.distance == 0
    rev = distance_to_target(
        gold, seed, space=space, inventory=[":cta.label"], node_budget=4
    )
    assert rev.kind is DistanceKind.EXACT and rev.distance == 1


def test_invariant_proof_is_bounded_without_finite_bounds(space) -> None:
    seed = parse_statements(SEED)
    bad = parse_statements('root = Stack([cta], "column")\ncta = Button(":zzz")')
    label = distance_to_target(seed, bad, space=space, inventory=[":cta.label"])
    assert label.kind is DistanceKind.BOUNDED
    assert label.reason == "invariant:needs_slot_rebind"
    assert label.lo is None and label.hi is None
    assert label.value_target(8) is None
    assert effective_distance(label) is None


def test_budget_exhaustion_is_explicit_unknown(space) -> None:
    gold = parse_statements(GOLD)
    far = parse_statements(
        'root = Stack([a, b], "column")\na = TextContent(":x")\nb = Button(":y")'
    )
    inv = [":x", ":y", ":cta.label"]
    label = distance_to_target(
        far, gold, space=space, inventory=inv, max_depth=8, node_budget=1
    )
    assert label.kind is DistanceKind.UNKNOWN
    assert label.reason == "budget"
    assert label.value_target(8) is None
    # A proven witness path upgrades UNKNOWN to BOUNDED(lo, hi) -- the bounds
    # are proofs (fully explored layers + witness), never coerced exact.
    bounded = distance_to_target(
        far, gold, space=space, inventory=inv,
        max_depth=8, node_budget=1, upper_bound_witness=3,
    )
    assert bounded.kind is DistanceKind.BOUNDED
    assert bounded.reason == "budget"
    assert (bounded.lo, bounded.hi) == (2, 3)
    assert bounded.value_target(8) == pytest.approx(1.0 - 2.5 / 8)


def test_depth_bound_is_bounded_not_unknown(space) -> None:
    gold = parse_statements(GOLD)
    far = parse_statements(
        'root = Stack([a, b], "column")\na = TextContent(":x")\nb = Button(":y")'
    )
    label = distance_to_target(
        far, gold, space=space, inventory=[":x", ":y", ":cta.label"],
        max_depth=0, node_budget=1, upper_bound_witness=4,
    )
    assert label.kind is DistanceKind.BOUNDED
    assert label.reason == "depth_bound"
    assert (label.lo, label.hi) == (1, 4)


def test_cache_hit_and_version_change_misses(space, monkeypatch) -> None:
    seed = parse_statements(SEED)
    gold = parse_statements(GOLD)
    kwargs = dict(space=space, inventory=[":cta.label"], node_budget=4)
    first = distance_to_target(seed, gold, **kwargs)
    assert not first.cache_hit
    second = distance_to_target(seed, gold, **kwargs)
    assert second.cache_hit and second.kind is first.kind
    monkeypatch.setattr(oracle, "action_schema_version", lambda: "v999")
    third = distance_to_target(seed, gold, **kwargs)
    assert not third.cache_hit
    monkeypatch.setattr(oracle, "grammar_version", lambda: "sha256:deadbeef")
    fourth = distance_to_target(seed, gold, **kwargs)
    assert not fourth.cache_hit


def test_grammar_version_is_grammar_file_hash() -> None:
    assert oracle.grammar_version().startswith("sha256:")
    assert oracle.action_schema_version().startswith("v")


def test_effective_distance_none_when_no_finite_estimate() -> None:
    from slm_training.harnesses.experiments.slm308_distance_oracle import DistanceLabel

    unknown = DistanceLabel(kind=DistanceKind.UNKNOWN, reason="budget")
    assert effective_distance(unknown) is None
    bounded_no_bounds = DistanceLabel(
        kind=DistanceKind.BOUNDED, reason="frontier_exhausted", lo=None, hi=None
    )
    assert effective_distance(bounded_no_bounds) is None
    bounded = DistanceLabel(kind=DistanceKind.BOUNDED, reason="budget", lo=2, hi=4)
    assert effective_distance(bounded) == pytest.approx(3.0)
