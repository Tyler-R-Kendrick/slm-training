"""Checked executable LALR adapter: lockstep certification + bound soundness.

The adapter is *not* a parser authority. Lark stays the authority; these tests
certify that the static projection tracks it exactly over the canonical E1
corpus, that a tampered array is rejected, that the acceptance-length table is a
sound lower bound, and that nothing in production consumes any of it yet.
"""

from __future__ import annotations

import subprocess
from collections import deque
from pathlib import Path

import pytest

from slm_training.dsl.grammar.fastpath.completion_artifact import (
    StaticLalrAdapter,
    _parse_table_arrays,
)
from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.static_control_domain import (
    STATIC_LALR_CORPUS,
    certify_static_lalr_adapter,
    require_certified_static_lalr,
    static_lalr_adapter,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OWNING_MODULES = {
    "src/slm_training/dsl/grammar/fastpath/completion_artifact.py",
    "src/slm_training/dsl/grammar/fastpath/static_control_domain.py",
}


@pytest.fixture(scope="module")
def engine() -> OpenUIIncrementalEngine:
    return OpenUIIncrementalEngine()


@pytest.fixture(scope="module")
def adapter(engine: OpenUIIncrementalEngine) -> StaticLalrAdapter:
    return static_lalr_adapter(engine=engine)


def test_lockstep_certification_over_corpus(
    engine: OpenUIIncrementalEngine, adapter: StaticLalrAdapter
) -> None:
    certificate = certify_static_lalr_adapter(adapter, engine._parser)

    assert certificate.equal, certificate.mismatches
    assert certificate.programs == len(STATIC_LALR_CORPUS)
    assert certificate.steps > 0
    assert certificate.states_visited > 1
    # The fail-closed entry point agrees and is process cached.
    cached = require_certified_static_lalr(engine=engine)
    assert cached.equal
    assert require_certified_static_lalr(engine=engine) is cached


def test_rule_tensors_match_the_live_parser(engine: OpenUIIncrementalEngine) -> None:
    tensors, metadata = _parse_table_arrays(engine._parser)
    symbols = metadata["symbols"]
    rules = list(engine._parser.rules)

    assert len(tensors["rule_origins"]) == len(rules)
    assert len(tensors["rule_lengths"]) == len(rules)
    for rule, origin, length in zip(
        rules, tensors["rule_origins"], tensors["rule_lengths"], strict=True
    ):
        assert symbols[origin] == str(rule.origin.name)
        assert length == len(rule.expansion)
    # Reduce actions index the same canonical rule ids.
    for kind, target in zip(
        tensors["lalr_action_kinds"], tensors["lalr_action_targets"], strict=True
    ):
        if kind == 1:
            assert 0 <= target < len(rules)


def _certify_mutation(
    engine: OpenUIIncrementalEngine, array: str, values: list[int]
) -> None:
    """A tampered array must be refused, never silently certified."""

    tensors, metadata = _parse_table_arrays(engine._parser)
    assert values != tensors[array]
    tampered = StaticLalrAdapter.from_arrays({**tensors, array: values}, metadata)
    try:
        certificate = certify_static_lalr_adapter(tampered, engine._parser)
    except (IndexError, KeyError, RuntimeError, ValueError):
        return  # structurally rejected — also a refusal
    assert not certificate.equal
    assert certificate.mismatches


@pytest.mark.parametrize("array", ["rule_lengths", "rule_origins"])
def test_tampered_rule_tensors_fail_certification(
    engine: OpenUIIncrementalEngine, array: str
) -> None:
    tensors, _metadata = _parse_table_arrays(engine._parser)
    values = list(tensors[array])
    if array == "rule_lengths":
        values = [value + 1 for value in values]
    else:
        values = values[1:] + values[:1]
    _certify_mutation(engine, array, values)


@pytest.mark.parametrize("array", ["lalr_action_kinds", "lalr_action_targets"])
def test_tampered_start_edge_fails_certification(
    engine: OpenUIIncrementalEngine, array: str
) -> None:
    """Tamper with an edge the corpus provably exercises on its first token."""

    tensors, metadata = _parse_table_arrays(engine._parser)
    start = metadata["start_states"]["start"]
    row = tensors["lalr_state_colors"].index(start)
    offsets = tensors["lalr_state_offsets"]
    slot = next(
        index
        for index in range(offsets[row], offsets[row + 1])
        if metadata["symbols"][tensors["lalr_edge_symbols"][index]] == "NAME"
        and tensors["lalr_action_kinds"][index] == 0
    )
    values = list(tensors[array])
    if array == "lalr_action_kinds":
        values[slot] = 1  # a shift relabelled as a reduce
    else:
        values[slot] = (values[slot] + 1) % len(tensors["lalr_state_colors"])
    _certify_mutation(engine, array, values)


def _reachable_stacks(
    adapter: StaticLalrAdapter, *, limit: int, max_depth: int
) -> list[tuple[int, ...]]:
    """Breadth-first enumeration of stacks reachable by terminal shifts."""

    start = adapter.start_stack()
    seen = {start}
    queue = deque([start])
    while queue and len(seen) < limit:
        stack = queue.popleft()
        for symbol, _kind, _target in adapter.rows[stack[-1]]:
            name = adapter.symbols[symbol]
            if not name.isupper() or name == adapter.END_SYMBOL:
                continue
            nxt = adapter.step(stack, name)
            if nxt is None or nxt in seen or len(nxt) > max_depth:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return sorted(seen, key=len)


def _true_min_terminals(
    adapter: StaticLalrAdapter, stack: tuple[int, ...], *, cap: int
) -> int | None:
    """Brute-force shortest terminal count from ``stack`` to an accepting one."""

    seen = {stack}
    queue = deque([(stack, 0)])
    while queue:
        current, distance = queue.popleft()
        if adapter.step(current, adapter.END_SYMBOL) is not None:
            return distance
        if len(seen) > cap:
            return None
        for symbol, _kind, _target in adapter.rows[current[-1]]:
            name = adapter.symbols[symbol]
            if not name.isupper() or name == adapter.END_SYMBOL:
                continue
            nxt = adapter.step(current, name)
            if nxt is None or nxt in seen or len(nxt) > 40:
                continue
            seen.add(nxt)
            queue.append((nxt, distance + 1))
    return None


def test_state_min_terminals_is_a_sound_lower_bound(
    adapter: StaticLalrAdapter,
) -> None:
    """The serialized table never over-states the true stack-dependent minimum."""

    stacks = _reachable_stacks(adapter, limit=600, max_depth=10)
    assert len(stacks) > 100

    checked = 0
    tops: set[int] = set()
    for stack in stacks:
        truth = _true_min_terminals(adapter, stack, cap=20000)
        if truth is None:
            continue
        checked += 1
        tops.add(stack[-1])
        bound = adapter.min_terminals(stack[-1])
        assert bound >= 0, (stack[-1], bound)
        assert bound <= truth, (stack[-1], bound, truth)
    assert checked == len(stacks)
    assert len(tops) > 10
    # The shipped table is a real bound, not an all-zero placeholder.
    assert max(adapter.state_min_terminals) > 0


def _grep_python_sources(pattern: str) -> set[str]:
    found = subprocess.run(
        ["git", "grep", "-l", "-E", pattern, "--", "src/*.py", "scripts/*.py"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert found.returncode in (0, 1), found.stderr
    return {line for line in found.stdout.splitlines() if line}


def test_adapter_is_import_only_with_no_production_consumer() -> None:
    """Nothing under src/ or scripts/ wires the adapter into a decode path."""

    adapter_hits = _grep_python_sources(
        "StaticLalrAdapter|require_certified_static_lalr|static_lalr_adapter"
    )
    assert adapter_hits == _OWNING_MODULES, sorted(adapter_hits)

    # The acceptance-length bound is serialized and reported, never consumed.
    bound_hits = _grep_python_sources("state_min_terminals|min_terminals")
    assert bound_hits <= _OWNING_MODULES | {"scripts/build_completion_artifact.py"}, (
        sorted(bound_hits)
    )
