"""RESEARCH-08 Myhill–Nerode quotient unit/property tests (SLM-540)."""

from __future__ import annotations

from slm_training.dsl.grammar.fastpath.myhill_nerode import (
    DEFAULT_OFF,
    TerminalContinuationDFA,
    build_terminal_continuation_dfa,
    complete_domain_parity,
    language_equivalent,
    minimize_myhill_nerode,
    run_research08_pilot,
)
from slm_training.formal.complete_domain import CompleteDomainEvidenceV1


def _toy_nonminimal() -> TerminalContinuationDFA:
    states = (0, 1, 2)
    terminals = ("A", "B")
    delta = (
        (2, 0),
        (2, 1),
        (None, None),
    )
    return TerminalContinuationDFA(
        states=states,
        terminals=terminals,
        start=0,
        end=2,
        delta=delta,
        accepting=(False, False, True),
        accepts_sets=(frozenset({"A", "B"}), frozenset({"A", "B"}), frozenset()),
        min_terminals=(1, 1, 0),
        state_index={0: 0, 1: 1, 2: 2},
    )


def test_default_off_pilot_skips() -> None:
    class _Dummy:
        rows = {0: ()}
        symbols = ()
        start_state = 0
        end_state = 0

        def accepts(self, stack):  # noqa: ANN001
            return frozenset()

        def step(self, stack, term):  # noqa: ANN001
            return None

        def min_terminals(self, state):  # noqa: ANN001
            return 0

    out = run_research08_pilot(_Dummy(), enabled=False)  # type: ignore[arg-type]
    assert out["skipped"] is True
    assert out["default_off"] is True
    assert DEFAULT_OFF is True


def test_toy_minimization_merges_equivalent_states() -> None:
    dfa = _toy_nonminimal()
    q = minimize_myhill_nerode(dfa)
    assert q.n_blocks == 2
    equiv = language_equivalent(dfa, q)
    assert equiv["equivalent"] is True
    parity = complete_domain_parity(dfa, q)
    assert parity["parity_ok"] is True


def test_forged_coverage_complete_never_authorizes() -> None:
    forged = CompleteDomainEvidenceV1(
        state_id="x",
        candidates=(),
        legal_members=(),
        checked_sound=False,
        checked_complete=False,
        legacy_coverage_complete=True,
    )
    assert forged.is_proved_complete() is False
    assert forged.authorizes_pruning(expected_state_id="x") is False


def test_live_adapter_quotient_language_equivalent() -> None:
    from slm_training.dsl.grammar.fastpath.static_control_domain import (
        static_lalr_adapter,
    )

    adapter = static_lalr_adapter()
    dfa = build_terminal_continuation_dfa(adapter)
    q = minimize_myhill_nerode(dfa)
    assert language_equivalent(dfa, q)["equivalent"] is True
    assert q.n_blocks <= dfa.n_states
    assert complete_domain_parity(dfa, q)["parity_ok"] is True
