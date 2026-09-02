#!/usr/bin/env python3
"""N8 regular-layer gap probe: how much of OpenUI legality is a finite automaton?

Read-only literature-ledger audit for the two-tower / automaton-constrained
decoding transfer table in ``docs/design/research-lineage.md`` (section
"Two-tower diffusion and automaton-constrained decoding transfer audit").
An exact DFA/NFA constrained sampler (arXiv:2607.07026, catalog row
``arxiv-2607.07026``) can at most replace the *regular* (lexeme-level) layer
of the legality authority.  This probe measures, on gold programs, how often
the regular layer alone is a strict over-approximation of the CFG-scoped
``CompletionDomainV1``.

For every record the gold ``openui`` program is encoded with
``DSLNativeTokenizer`` and walked position by position.  At each prefix:

- **automaton set** = ``OpenUIIncrementalEngine.next_terminals()`` expanded to
  token ids through the engine's verified token->terminal map (punctuation,
  bool/null, string-literal rows, framed-literal openers, ``kind_ids`` for the
  broad content classes, ``eos_id`` when ``$END`` is accepted);
- **domain set** = first token id of every candidate returned by the pack's
  ``CompletionDomainV1`` builder with a ``remaining_tokens`` horizon;
- a decision is **non-singleton** when the domain has >= 2 first-token
  candidates, and counts as a *gap* when ``automaton set`` is a strict
  superset of ``domain set``.

No model, checkpoint, metric, gate, or lever is touched; nothing is written
unless ``--out`` is given.  Bounded by ``--time-budget`` (seconds) so it obeys
the hard run cap; the report states how many records were actually walked.

    PYTHONPATH=src python -m scripts.audit_regular_layer_gap --limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_RECORDS = (
    "src/slm_training/resources/data/eval/"
    "e938_role_safe_all_targets_smoke24_v1/suites/smoke/records.jsonl"
)


class _RowState:
    """Minimal request-local holder so the packed completion session is reused."""

    completion_session: object | None = None
    completion_authority_key: tuple[object, ...] | None = None
    completion_state_id: int | None = None
    completion_batch_cache = None
    completion_domain_cache = None

    def __init__(self) -> None:
        self.completion_prefix_states: dict[tuple[int, ...], int] = {}

    def bind_completion_session(
        self,
        session: object,
        authority_key: tuple[object, ...],
        state_id: int,
        prefix_ids: tuple[int, ...],
    ) -> None:
        if session is not self.completion_session:
            self.completion_prefix_states = {}
        self.completion_session = session
        self.completion_authority_key = authority_key
        self.completion_state_id = int(state_id)
        self.completion_prefix_states[tuple(int(t) for t in prefix_ids)] = int(state_id)


def _terminal_to_ids(tokenizer: Any, engine: Any) -> dict[str, frozenset[int]]:
    """Invert the engine's verified token->terminal map into terminal->ids."""
    mapping = engine._direct_map(tokenizer)  # noqa: SLF001 - probe reads the live map
    if mapping is None:
        raise RuntimeError("engine has no verified direct map for this tokenizer")
    out: dict[str, set[int]] = {}
    for tid, term in mapping["punct"].items():
        out.setdefault(term, set()).add(int(tid))
    for table in (mapping["bool"], mapping["null"], mapping["str_lit_ids"]):
        for tid, term in table.items():
            out.setdefault(term, set()).add(int(tid))
    for kind, term in mapping["kind_terminals"].items():
        out.setdefault(term, set()).update(int(t) for t in tokenizer.kind_ids(kind))
    out.setdefault("NUMBER", set()).add(int(mapping["lit_num"]))
    if mapping["lit_str"] is not None:
        out.setdefault("STRING", set()).add(int(mapping["lit_str"]))
    out["$END"] = {int(tokenizer.eos_id)}
    return {term: frozenset(ids) for term, ids in out.items()}


def walk_record(
    record: dict[str, Any],
    tokenizer: Any,
    term_to_ids: dict[str, frozenset[int]],
    *,
    remaining_tokens: int,
) -> dict[str, Any]:
    from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
    from slm_training.dsl.grammar.fastpath.token_map import token_surface_piece
    from slm_training.dsl.grammar_capabilities import CompletionDomainRequestV1
    from slm_training.dsl.pack import _openui_completion_domain

    placeholders = tuple(str(p) for p in record.get("placeholders", ()))
    ids = [
        int(i)
        for i in tokenizer.encode(
            str(record["openui"]) + "\n", add_special=False, placeholders=placeholders
        )
    ]
    engine = OpenUIIncrementalEngine()
    engine.set_prefix("")
    row_state = _RowState()
    positions = 0
    singleton = 0
    forced_by_automaton = 0
    nonsingleton = 0
    gap = 0
    equal = 0
    dfa_sizes: list[int] = []
    dom_sizes: list[int] = []
    status_counts: dict[str, int] = {}
    gold_outside_complete = 0
    gold_misses: list[dict[str, Any]] = []
    names = tokenizer.id_to_token
    for p in range(len(ids) + 1):
        positions += 1
        terms = engine.next_terminals()
        dfa = set()
        for term in terms:
            dfa.update(term_to_ids.get(term, ()))
        domain = _openui_completion_domain(
            CompletionDomainRequestV1(
                prefix_ids=tuple(ids[:p]),
                tokenizer=tokenizer,
                slot_contract=placeholders,
                remaining_tokens=remaining_tokens,
                state=row_state,
            )
        )
        status = str(domain.status)
        status_counts[status] = status_counts.get(status, 0) + 1
        first = {int(c.token_ids[0]) for c in domain.candidates if c.token_ids}
        gold = int(tokenizer.eos_id) if p == len(ids) else ids[p]
        if status == "complete" and gold not in first:
            gold_outside_complete += 1
            gold_misses.append(
                {
                    "position": p,
                    "gold": str(names.get(gold, gold)),
                    "domain": sorted(str(names.get(t, t)) for t in first)[:12],
                    "domain_size": len(first),
                }
            )
        if len(first) <= 1:
            singleton += 1
            if engine.is_deterministic_next() is not None:
                forced_by_automaton += 1
        else:
            nonsingleton += 1
            if dfa > first:
                gap += 1
            elif dfa == first:
                equal += 1
            dfa_sizes.append(len(dfa))
            dom_sizes.append(len(first))
        if p < len(ids):
            fed = engine.feed_token_id(tokenizer, ids[p])
            if fed is None:
                fed = engine.advance_checked(token_surface_piece(tokenizer, ids[p]))
            if fed is False:
                raise RuntimeError(
                    f"{record.get('id')}: grammar rejected gold token at position {p}"
                )
    return {
        "id": record.get("id"),
        "gold_tokens": len(ids),
        "positions": positions,
        "singleton_or_empty": singleton,
        "singleton_forced_by_automaton": forced_by_automaton,
        "nonsingleton": nonsingleton,
        "nonsingleton_automaton_strictly_larger": gap,
        "nonsingleton_sets_equal": equal,
        "mean_automaton_set": (sum(dfa_sizes) / len(dfa_sizes)) if dfa_sizes else None,
        "mean_domain_set": (sum(dom_sizes) / len(dom_sizes)) if dom_sizes else None,
        "domain_status_counts": status_counts,
        "gold_outside_complete_domain": gold_outside_complete,
        "gold_misses": gold_misses,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, default=Path(DEFAULT_RECORDS))
    parser.add_argument("--limit", type=int, default=None, help="Walk only the first N records.")
    parser.add_argument("--remaining-tokens", type=int, default=32)
    parser.add_argument(
        "--time-budget",
        type=float,
        default=150.0,
        help="Stop starting new records after this many seconds (hard run cap guard).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args(argv)

    from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        records = records[: args.limit]
    tokenizer = DSLNativeTokenizer.build()
    engine = OpenUIIncrementalEngine()
    engine.set_prefix("")
    term_to_ids = _terminal_to_ids(tokenizer, engine)

    started = time.monotonic()
    per_record: list[dict[str, Any]] = []
    skipped: list[str] = []
    for record in records:
        if time.monotonic() - started > args.time_budget:
            skipped.append(str(record.get("id")))
            continue
        per_record.append(
            walk_record(record, tokenizer, term_to_ids, remaining_tokens=args.remaining_tokens)
        )
    elapsed = time.monotonic() - started

    def _sum(key: str) -> int:
        return int(sum(int(r[key]) for r in per_record))

    gap_rows = [r for r in per_record if r["nonsingleton"]]
    weighted_dfa = sum(r["mean_automaton_set"] * r["nonsingleton"] for r in gap_rows)
    weighted_dom = sum(r["mean_domain_set"] * r["nonsingleton"] for r in gap_rows)
    nonsingleton = _sum("nonsingleton")
    status_counts: dict[str, int] = {}
    for r in per_record:
        for k, v in r["domain_status_counts"].items():
            status_counts[k] = status_counts.get(k, 0) + int(v)
    report = {
        "schema": "regular_layer_gap_probe/v1",
        "records_file": str(args.records),
        "records_available": len(records),
        "records_walked": len(per_record),
        "records_skipped_by_time_budget": skipped,
        "remaining_tokens": args.remaining_tokens,
        "elapsed_seconds": round(elapsed, 1),
        "gold_tokens": _sum("gold_tokens"),
        "positions": _sum("positions"),
        "singleton_or_empty": _sum("singleton_or_empty"),
        "singleton_forced_by_automaton": _sum("singleton_forced_by_automaton"),
        "nonsingleton": nonsingleton,
        "nonsingleton_automaton_strictly_larger": _sum("nonsingleton_automaton_strictly_larger"),
        "nonsingleton_sets_equal": _sum("nonsingleton_sets_equal"),
        "gap_fraction": (
            _sum("nonsingleton_automaton_strictly_larger") / nonsingleton if nonsingleton else None
        ),
        "mean_automaton_set_at_nonsingleton": (weighted_dfa / nonsingleton) if nonsingleton else None,
        "mean_domain_set_at_nonsingleton": (weighted_dom / nonsingleton) if nonsingleton else None,
        "domain_status_counts": status_counts,
        "gold_outside_complete_domain": _sum("gold_outside_complete_domain"),
        "per_record": per_record,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
