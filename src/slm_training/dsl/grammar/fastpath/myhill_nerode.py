"""RESEARCH-08 (SLM-540): Myhill–Nerode quotient of the static residual graph.

Builds an explicitly regular, finitely abstracted *terminal-continuation* DFA
from a certified ``StaticLalrAdapter`` (singleton-stack residual), then
deterministically quotients by Myhill–Nerode equivalence.

Default-off / research-only: never production authority. Request-local overlays
(scope, symbols, literals, macros, semantic state) are out of scope — this
module only sees the static adapter.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from slm_training.dsl.grammar.fastpath.completion_artifact import StaticLalrAdapter
from slm_training.formal.complete_domain import (
    CompleteDomainEvidenceV1,
    build_complete_domain,
)

SCHEMA = "research08_terminal_continuation_dfa/v1"
QUOTIENT_SCHEMA = "research08_myhill_nerode_quotient/v1"
DEFAULT_OFF = True


@dataclass(frozen=True)
class TerminalContinuationDFA:
    """Regular abstraction: states = LALR colors; alphabet = terminals + $END."""

    states: tuple[int, ...]
    terminals: tuple[str, ...]
    start: int
    end: int
    delta: tuple[tuple[int | None, ...], ...]  # state_idx -> terminal_idx -> state|None
    accepting: tuple[bool, ...]  # can consume $END from singleton stack
    accepts_sets: tuple[frozenset[str], ...]
    min_terminals: tuple[int, ...]
    state_index: Mapping[int, int]

    @property
    def n_states(self) -> int:
        return len(self.states)

    def to_serializable(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "states": list(self.states),
            "terminals": list(self.terminals),
            "start": self.start,
            "end": self.end,
            "delta": [
                {
                    self.terminals[j]: (
                        None if nxt is None else self.states[nxt]
                    )
                    for j, nxt in enumerate(row)
                }
                for row in self.delta
            ],
            "accepting": list(self.accepting),
            "accepts_sets": [sorted(s) for s in self.accepts_sets],
            "min_terminals": list(self.min_terminals),
        }

    def serialized_bytes(self) -> bytes:
        return json.dumps(
            self.to_serializable(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


@dataclass(frozen=True)
class QuotientDFA:
    """Myhill–Nerode quotient of a :class:`TerminalContinuationDFA`."""

    dfa: TerminalContinuationDFA
    source_states: tuple[int, ...]  # original colors, parallel to partition
    partition: tuple[int, ...]  # parallel to source_states -> block id
    n_blocks: int
    representative: tuple[int, ...]  # block id -> original state color

    def to_serializable(self) -> dict[str, Any]:
        return {
            "schema": QUOTIENT_SCHEMA,
            "n_blocks": self.n_blocks,
            "n_source_states": len(self.partition),
            "partition": {
                str(state): block
                for state, block in zip(
                    self.source_states, self.partition, strict=True
                )
            },
            "representative": list(self.representative),
            "dfa": self.dfa.to_serializable(),
        }

    def serialized_bytes(self) -> bytes:
        return json.dumps(
            self.to_serializable(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


def _terminal_alphabet(adapter: StaticLalrAdapter) -> tuple[str, ...]:
    names = {
        adapter.symbols[sym]
        for row in adapter.rows.values()
        for sym, _kind, _tgt in row
        if adapter.symbols[sym].isupper()
    }
    # $END is handled via accepting bit; keep it out of the step alphabet unless
    # present as a row symbol (Lark may list it).
    names.discard(StaticLalrAdapter.END_SYMBOL)
    return tuple(sorted(names))


def build_terminal_continuation_dfa(
    adapter: StaticLalrAdapter,
) -> TerminalContinuationDFA:
    """Finite regular abstraction over singleton-stack terminal residuals."""

    states = tuple(sorted(adapter.rows))
    index = {state: i for i, state in enumerate(states)}
    terminals = _terminal_alphabet(adapter)
    delta_rows: list[tuple[int | None, ...]] = []
    accepting: list[bool] = []
    accepts_sets: list[frozenset[str]] = []
    mins: list[int] = []
    for state in states:
        stack0 = (state,)
        accepts_sets.append(adapter.accepts(stack0))
        accepting.append(adapter.step(stack0, StaticLalrAdapter.END_SYMBOL) is not None)
        mins.append(int(adapter.min_terminals(state)))
        row: list[int | None] = []
        for term in terminals:
            nxt = adapter.step(stack0, term)
            if nxt is None:
                row.append(None)
            else:
                row.append(index[nxt[-1]])
        delta_rows.append(tuple(row))
    return TerminalContinuationDFA(
        states=states,
        terminals=terminals,
        start=adapter.start_state,
        end=adapter.end_state,
        delta=tuple(delta_rows),
        accepting=tuple(accepting),
        accepts_sets=tuple(accepts_sets),
        min_terminals=tuple(mins),
        state_index=index,
    )


def minimize_myhill_nerode(dfa: TerminalContinuationDFA) -> QuotientDFA:
    """Deterministic partition refinement (Moore) → Myhill–Nerode quotient."""

    n = dfa.n_states
    # Initial: accepting bit + accepts-set + min_terminals (sound distinguisher).
    buckets: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for i in range(n):
        key = (
            dfa.accepting[i],
            frozenset(dfa.accepts_sets[i]),
            dfa.min_terminals[i],
        )
        buckets[key].append(i)
    partition = [0] * n
    for block_id, members in enumerate(buckets.values()):
        for i in members:
            partition[i] = block_id

    def refine() -> bool:
        nonlocal partition
        next_buckets: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for i in range(n):
            sig = tuple(
                None if nxt is None else partition[nxt] for nxt in dfa.delta[i]
            )
            next_buckets[(partition[i], sig)].append(i)
        new_partition = [0] * n
        for block_id, members in enumerate(next_buckets.values()):
            for i in members:
                new_partition[i] = block_id
        changed = new_partition != partition
        partition = new_partition
        return changed

    for _ in range(n + 2):
        if not refine():
            break
    else:  # pragma: no cover
        raise RuntimeError("Myhill–Nerode refinement did not converge")

    n_blocks = max(partition) + 1 if n else 0
    representative = [-1] * n_blocks
    for i, block in enumerate(partition):
        if representative[block] < 0:
            representative[block] = dfa.states[i]

    # Quotient DFA rows (by representative).
    q_states = tuple(range(n_blocks))
    q_index = {b: b for b in q_states}
    q_delta: list[tuple[int | None, ...]] = []
    q_accepting: list[bool] = []
    q_accepts: list[frozenset[str]] = []
    q_mins: list[int] = []
    for block in q_states:
        src = dfa.state_index[representative[block]]
        q_accepting.append(dfa.accepting[src])
        q_accepts.append(dfa.accepts_sets[src])
        q_mins.append(dfa.min_terminals[src])
        q_delta.append(
            tuple(None if nxt is None else partition[nxt] for nxt in dfa.delta[src])
        )

    start_block = partition[dfa.state_index[dfa.start]]
    end_block = partition[dfa.state_index[dfa.end]]
    q_dfa = TerminalContinuationDFA(
        states=q_states,
        terminals=dfa.terminals,
        start=start_block,
        end=end_block,
        delta=tuple(q_delta),
        accepting=tuple(q_accepting),
        accepts_sets=tuple(q_accepts),
        min_terminals=tuple(q_mins),
        state_index=q_index,
    )
    # Map partition over original colors in state order.
    color_partition = tuple(partition[dfa.state_index[s]] for s in dfa.states)
    return QuotientDFA(
        dfa=q_dfa,
        source_states=dfa.states,
        partition=color_partition,
        n_blocks=n_blocks,
        representative=tuple(representative),
    )


def language_equivalent(
    control: TerminalContinuationDFA, treatment: QuotientDFA
) -> dict[str, Any]:
    """Check abstracted languages agree under the quotient map."""

    disagreements: list[str] = []
    part = {
        state: block
        for state, block in zip(
            treatment.source_states, treatment.partition, strict=True
        )
    }
    for i, state in enumerate(control.states):
        block = part[state]
        src = treatment.dfa.state_index[block]
        if control.accepting[i] != treatment.dfa.accepting[src]:
            disagreements.append(f"accepting:{state}")
        if control.accepts_sets[i] != treatment.dfa.accepts_sets[src]:
            disagreements.append(f"accepts_set:{state}")
        if control.min_terminals[i] != treatment.dfa.min_terminals[src]:
            disagreements.append(f"min_terminals:{state}")
        for j, term in enumerate(control.terminals):
            c_nxt = control.delta[i][j]
            t_nxt = treatment.dfa.delta[src][j]
            if c_nxt is None and t_nxt is None:
                continue
            if c_nxt is None or t_nxt is None:
                disagreements.append(f"delta_none:{state}:{term}")
                continue
            if part[control.states[c_nxt]] != t_nxt:
                disagreements.append(f"delta:{state}:{term}")
    return {
        "equivalent": not disagreements,
        "disagreement_count": len(disagreements),
        "disagreements_sample": disagreements[:32],
    }


def complete_domain_parity(
    control: TerminalContinuationDFA,
    treatment: QuotientDFA,
    *,
    fixture_states: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Proof-bearing CompleteDomain parity on frozen fixture states."""

    states = tuple(fixture_states) if fixture_states is not None else control.states
    part = {
        state: block
        for state, block in zip(
            treatment.source_states, treatment.partition, strict=True
        )
    }
    term_to_id = {name: idx for idx, name in enumerate(control.terminals)}
    reports: list[dict[str, Any]] = []
    all_ok = True
    for state in states:
        if state not in control.state_index:
            all_ok = False
            reports.append({"state": state, "ok": False, "reason": "unknown_state"})
            continue
        ci = control.state_index[state]
        legal_names = control.accepts_sets[ci]
        legal_ids = tuple(sorted(term_to_id[n] for n in legal_names if n in term_to_id))
        control_ev = build_complete_domain(
            f"control:{state}", legal_ids, legal_ids
        )
        block = part[state]
        ti = treatment.dfa.state_index[block]
        treat_names = treatment.dfa.accepts_sets[ti]
        treat_ids = tuple(sorted(term_to_id[n] for n in treat_names if n in term_to_id))
        treat_ev = build_complete_domain(
            f"treatment:{block}", treat_ids, treat_ids
        )
        ok = (
            control_ev.is_proved_complete()
            and treat_ev.is_proved_complete()
            and legal_ids == treat_ids
            and legal_names == treat_names
        )
        # forged Boolean alone must not authorize
        forged = CompleteDomainEvidenceV1(
            state_id=f"forged:{state}",
            candidates=(),
            legal_members=(),
            checked_sound=False,
            checked_complete=False,
            legacy_coverage_complete=True,
        )
        if forged.is_proved_complete() or forged.authorizes_pruning(
            expected_state_id=f"forged:{state}"
        ):
            ok = False
        all_ok = all_ok and ok
        reports.append(
            {
                "state": state,
                "block": block,
                "ok": ok,
                "control_authority": control_ev.authority_name(),
                "treatment_authority": treat_ev.authority_name(),
                "n_candidates": len(legal_ids),
            }
        )
    return {
        "parity_ok": all_ok,
        "n_fixtures": len(states),
        "n_passed": sum(1 for r in reports if r.get("ok")),
        "reports": reports,
    }


def measure_cold_load(
    blob: bytes, *, trials: int = 31
) -> dict[str, float]:
    """Primary metric helper: cold JSON artifact load/init time (ms)."""

    times: list[float] = []
    for _ in range(max(1, trials)):
        t0 = time.perf_counter()
        obj = json.loads(blob)
        # touch start transition row so decode isn't elided
        _ = obj.get("delta") or obj.get("dfa", {}).get("delta")
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    mid = times[len(times) // 2]
    return {
        "p50_ms": mid,
        "min_ms": times[0],
        "max_ms": times[-1],
        "trials": float(len(times)),
        "bytes": float(len(blob)),
    }


def run_research08_pilot(
    adapter: StaticLalrAdapter,
    *,
    enabled: bool = False,
    cold_trials: int = 31,
) -> dict[str, Any]:
    """Execute the default-off RESEARCH-08 pilot against a static adapter."""

    if not enabled:
        return {
            "schema": "research08_pilot_result/v1",
            "enabled": False,
            "default_off": DEFAULT_OFF,
            "skipped": True,
            "reason": "pilot disabled (default_off); pass enabled=True under research harness",
        }
    control = build_terminal_continuation_dfa(adapter)
    treatment = minimize_myhill_nerode(control)
    equiv = language_equivalent(control, treatment)
    # Frozen fixtures: start, end, and a dense sample of colors.
    fixtures = tuple(
        sorted({control.start, control.end, *control.states[:: max(1, len(control.states) // 16)]})
    )
    parity = complete_domain_parity(control, treatment, fixture_states=fixtures)
    control_blob = control.serialized_bytes()
    treatment_blob = treatment.serialized_bytes()
    control_cold = measure_cold_load(control_blob, trials=cold_trials)
    treatment_cold = measure_cold_load(treatment_blob, trials=cold_trials)
    state_reduction = control.n_states - treatment.n_blocks
    cold_improved = treatment_cold["p50_ms"] < control_cold["p50_ms"] and (
        treatment_cold["bytes"] < control_cold["bytes"]
    )
    zero_disagreement = bool(equiv["equivalent"]) and bool(parity["parity_ok"])
    decision = (
        "accept_research_only"
        if zero_disagreement and cold_improved and state_reduction > 0
        else "reject"
    )
    return {
        "schema": "research08_pilot_result/v1",
        "enabled": True,
        "default_off": DEFAULT_OFF,
        "production_authority": False,
        "request_local_overlays_isolated": True,
        "control": {
            "n_states": control.n_states,
            "n_terminals": len(control.terminals),
            "bytes": len(control_blob),
            "cold_load": control_cold,
        },
        "treatment": {
            "n_states": treatment.n_blocks,
            "n_terminals": len(treatment.dfa.terminals),
            "bytes": len(treatment_blob),
            "cold_load": treatment_cold,
            "state_reduction": state_reduction,
        },
        "language_equivalence": equiv,
        "complete_domain_parity": {
            "parity_ok": parity["parity_ok"],
            "n_fixtures": parity["n_fixtures"],
            "n_passed": parity["n_passed"],
        },
        "primary_metric": {
            "name": "cold_artifact_load_initialization_time",
            "control_p50_ms": control_cold["p50_ms"],
            "treatment_p50_ms": treatment_cold["p50_ms"],
            "delta_ms": control_cold["p50_ms"] - treatment_cold["p50_ms"],
            "improved": cold_improved,
        },
        "secondary": {
            "language_equivalence_failures": equiv["disagreement_count"],
            "state_count_control": control.n_states,
            "state_count_treatment": treatment.n_blocks,
            "artifact_bytes_control": len(control_blob),
            "artifact_bytes_treatment": len(treatment_blob),
        },
        "activation_telemetry": {
            "pilot": "myhill_nerode_minimize",
            "scope": "static_regular_subset",
            "abstraction": "terminal_continuation_singleton_stack",
            "coverage_states": control.n_states,
            "quotient_blocks": treatment.n_blocks,
            "enabled": True,
        },
        "zero_semantic_disagreement": zero_disagreement,
        "decision": decision,
    }


__all__ = [
    "DEFAULT_OFF",
    "QUOTIENT_SCHEMA",
    "SCHEMA",
    "QuotientDFA",
    "TerminalContinuationDFA",
    "build_terminal_continuation_dfa",
    "complete_domain_parity",
    "language_equivalent",
    "measure_cold_load",
    "minimize_myhill_nerode",
    "run_research08_pilot",
]
