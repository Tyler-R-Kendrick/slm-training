# Autotrain c1738 runtime-repair study

**Outcome:** the repair converts escaped alarms into typed timeout evidence and
increases exact constrained progress, but does not yet complete
`smoke_hero_01` inside the frozen 24-second document wall. It is harness
progress, not model-quality or ship evidence.

## Scratch result matrix

| Exact variant | Tokens at wall | Forwards | Compiler ms | States | Witnesses | Forks | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cold setup separated | 37 | 26 | 21,396.637 | 128,093 | 9,060 | 145,179 | typed timeout |
| immutable semantic hash | 50 | 36 | 20,457.300 | 132,384 | 9,679 | 149,955 | typed timeout |
| minimal official root projection | 56 | 38 | 20,197.334 | 144,422 | 10,368 | 162,666 | typed timeout |
| persistent official streaming root probe | 59 | 39 | 20,059.013 | 144,470 | 10,438 | 162,754 | typed timeout |
| callback-free parser configuration sharing | 63 | 42 | 19,787.817 | 144,524 | 10,526 | 162,854 | typed timeout |
| copy-on-write fed-token history | 66 | 45 | 19,562.599 | 144,567 | 10,592 | 162,933 | typed timeout |

All rows use the same c1737 control checkpoint, `smoke` offset 0, one record,
strict compiler-tree policy, CPU, and a 24-second diagnostic deadline. The
later rows traverse more of the same deterministic output, so raw state totals
are not direct speed ratios.

## What changed

- Immutable `SemanticState` hashes and tokenizer token/kind projections are
  memoized without changing equality, legal domains, proof budgets, or order.
- Process-cold parser/schema/static-map initialization runs before the
  per-document wall and is disclosed as `decode_initialization_ms`.
- Root-presence checks still use official `@openuidev/lang-core`; the bridge
  returns only the required boolean and reuses its official streaming parser.
- Descendant control-only Lark forks share their immutable callback-free parse
  configuration while retaining fork-local state/value stacks and lexer
  threads. This changes allocation only; exact domain parity remains green.
- Parser forks also share append-only fed-token histories until the first
  branch-local terminal commit, when both buffers detach before mutation. The
  ownership regression test and exact domain-parity suite remain green.
- Repeating `SIGALRM` delivery is disarmed before timeout bookkeeping. The
  prior failure exited 142 with an uncaught second alarm; the repaired runs
  exit 0 with `decode_outcome=runtime_timeout` and complete AgentV artifacts.

## Rejected or neutral hypotheses

- Cross-query witness-subproblem caching had negligible reuse and was removed;
  only completed identical top-level verdicts remain cached. `UNKNOWN` is
  never cached.
- A SemanticArena transition cache and first-path-limited forest were neutral
  or slower and were reverted.
- Closure-preferred witness order changed the exact parity outcome and was
  reverted immediately. Search order remains authoritative.
- Transferring either the final verified branch state or all intermediate
  verified branch states into the session cache changed exact candidate or
  witness ordering in differential parity tests. Both variants were reverted.

## Frozen replay after repair

| Cycle | Arm | Complete | Timeout | p50 incl. incomplete | Forwards | States | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c1739 | matched control | 2/3 | 1 | 20,108.53 ms | 118 | 310,372 | incomplete |
| c1739 | compact canvas | 2/3 | 1 | 19,685.63 ms | 119 | 313,492 | incomplete |
| c1740 | matched control | 2/3 | 1 | 19,078.93 ms | 134 | 313,834 | incomplete |
| c1740 | compact canvas | 2/3 | 1 | 18,827.20 ms | 131 | 313,793 | incomplete |

This replay confirms that the accepted repairs preserve typed evidence and
increase exact progress, but remain insufficient to make the authoritative
comparison scoreable. The corrected SDLC classifier reports the cycle as
non-positive. c1740 also rejects the candidate's approximately 4.04% partial
MPR/ms gain because it is below the preregistered 5% minimum effect.

## Next hypotheses

1. Reduce mechanical Lark parser-state copy and allocation cost without
   changing traversal, candidates, budgets, or cache-observable state.
2. Batch or incrementally retain official root probes across terminal witness
   siblings; the profile still attributes about 2.9 seconds to 189 bridge
   probes.
3. Reduce compiler state construction and parser-copy cost before changing any
   timeout. The compiler consumes about 20 seconds of the 24-second wall,
   whereas neural work is roughly 3.7 seconds.
4. Keep Lean/full formal validation in the delivery gate; no heuristic or
   incomplete proof may widen a legal domain.

Machine-readable evidence is in
[`autotrain-cycle-1738-runtime-repair.json`](autotrain-cycle-1738-runtime-repair.json).
