# Autotrain c1727: persistent-prefix replay and hero profile

**Verdict:** persistent completion prefixes preserved the exact candidate
trajectory and slightly reduced compiler time, but did not complete the frozen
batch-size-1 arm. AgentV finalized two documents and one typed runtime timeout.
The quality result remains incomplete and no promotion is authorized.

## Result matrix

| Arm | Records | Parse | Binder F1 | Meaningful | Structure | p50 completed | Timeout | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 3/3 | 1.0 | 1.0 | 0.3333 | 0.3656 | 2,844.47 ms | 0 | Complete fixture control; ship gates fail |
| frozen batch1 | 2/3 | 1.0 | 1.0 | 0.5 | 0.1869 | 8,579.53 ms | 1 | Incomplete quality; further runtime repair required |

The candidate rates exclude the timed-out hero document. On the completed subset,
structure remains 49% below control and latency remains 3.02 times control.

## Repair matrix

| Candidate signal | c1726 before | c1727 after | Change / interpretation |
| --- | ---: | ---: | --- |
| p50 completed | 8,163.93 ms | 8,579.53 ms | +5.09%; host variance dominates |
| p50 including incomplete | 19,345.73 ms | 19,036.14 ms | -1.60% |
| total decode work | 51.510 s | 51.617 s | +0.21% |
| compiler | 41.019 s | 40.584 s | -1.06%; insufficient |
| backbone | 10.016 s | 10.535 s | +5.18% |
| neural forwards | 115 | 115 | Exact model-compute parity |
| completed documents | 2 / 3 | 2 / 3 | Repair insufficient |

The timed-out hero followed the same exact token trajectory. The state-count increase
is the same deeper-progress-at-deadline effect observed in c1726, not a widened
domain. Persistent prefixes are retained as a representation simplification, but
they are rejected as the dominant timeout repair.

## Diagnostic hero profile

An exact one-record `cProfile` reproduction used the same checkpoint, suite offset,
24-second cooperative deadline, constrained decode, and honest slot contract. It is
a diagnostic profile (`n=1`), not a quality or latency comparison: profiling overhead
raised wall time to 31.631 seconds and the record remained a typed timeout.

| Cumulative site | Calls | Profile time | Signal |
| --- | ---: | ---: | --- |
| completion forest / domain | 6 | 22.035 s | Dominant exact witness work |
| terminal witness recursion | 13,813 | 21.838 s | Dominant repair boundary |
| outgoing-domain construction | 492 | 12.956 s | Repeated authority queries |
| completion advance path | 13,814 | 8.675 s | Transition hot path |
| token `kind_of` | 301,888 | 2.835 s | Repeated immutable enum construction |
| semantic-kind projection | 136,321 | 2.530 s | Same token-kind authority consumer |
| parser control copy | 34,717 | 2.428 s | Secondary structural-copy cost |
| lazy module imports | diagnostic | 3.560 s | Cold-start signal; measure separately |

The next smallest exact repair precomputes immutable `TokenKind` enum values once per
tokenizer. It does not change token ids, candidate ordering, grammar authority,
deadlines, node budgets, or model forwards. Cold import/preflight work stays inside
the current measurement until a separate startup contract reports it explicitly.

## Next-run priorities

1. Replay the exact frozen manifest after token-kind caching. Require three completed
   AgentV dispositions and zero decode timeouts before using quality metrics.
2. Confirm the hero trajectory and 115 neural forwards remain unchanged; divergence
   is an authority regression.
3. If the timeout survives, preregister separate cold-start and warm-steady-state
   timing fields before moving lazy initialization outside the record budget.
4. Then target parser control-fork copying or repeated terminal-witness construction,
   whichever remains dominant under an unprofiled exact replay.
5. Once runtime completes, return to a size-matched model/data hypothesis for the
   structural regression.

Hosted Lean and lean-formal jobs passed for the c1726 repair. No checkpoint was
created or promoted in c1727, so no model-card or README checkpoint update is
required. Machine-readable evidence is in
[`autotrain-cycle-1727-persistent-prefix-replay.json`](autotrain-cycle-1727-persistent-prefix-replay.json).
