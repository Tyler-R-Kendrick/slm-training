# Autotrain c1726: disposable-branch repair result

**Verdict:** the disposable-branch snapshot repair reduced compiler runtime, but it
did not complete the frozen batch-size-1 measurement. AgentV again finalized two
documents and one typed runtime timeout. The quality result remains incomplete and
no promotion is authorized.

## Result matrix

| Arm | Records | Parse | Binder F1 | Meaningful | Structure | p50 completed | Timeout | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 3/3 | 1.0 | 1.0 | 0.3333 | 0.3656 | 2,958.47 ms | 0 | Complete fixture control; ship gates fail |
| frozen batch1 | 2/3 | 1.0 | 1.0 | 0.5 | 0.1869 | 8,163.93 ms | 1 | Incomplete quality; further runtime repair required |

The candidate rates still exclude the timed-out hero document. On the completed
subset, structure remains 49% below control and latency remains 2.76 times control.

## Repair matrix

| Candidate signal | c1725 before | c1726 after | Change / interpretation |
| --- | ---: | ---: | --- |
| p50 completed | 8,986.38 ms | 8,163.93 ms | -9.15%; real local improvement |
| p50 including incomplete | 19,868.99 ms | 19,345.73 ms | -2.63% |
| total decode work | 53.079 s | 51.510 s | -2.96% |
| compiler | 42.483 s | 41.019 s | -3.45%; still dominant |
| backbone | 10.125 s | 10.016 s | -1.08% |
| neural forwards | 115 | 115 | Exact model-compute parity |
| completed documents | 2 / 3 | 2 / 3 | Repair insufficient |

The timed-out hero selection trajectory is identical through position 73. Aggregate
state and fork counts are slightly larger after the repair because the cooperative
deadline permits more of the same exact witness search to run before interruption;
that is progress depth, not a widened domain or changed token choice.

## Next canonical repair

Every accepted transition currently creates a full tuple copy of the entire token
prefix for its newly interned state. c1726 created 268,755 states while only 13,866
states required witness expansion, so eager prefix duplication performs work for a
large majority of states that never need a prefix scan.

The next repair stores each child as an immutable parent-state plus token link and
materializes the exact prefix tuple only when the official authority builder asks for
it. The seed prefix remains concrete; interning keys, parser and semantic states,
candidate ordering, node-budget debit, typed `UNKNOWN`, and every output payload stay
unchanged. A regression test proves transition-only children remain deferred until
`prefix_ids_of` requests the exact tuple.

## Next-run priorities

1. Replay the exact c1726 frozen manifest after persistent-prefix delivery. Require
   three completed AgentV dispositions and zero decode timeouts before quality use.
2. Confirm the hero token trajectory remains identical and neural forwards remain
   115; any divergence requires an authority investigation.
3. If the timeout survives, profile parser control-fork list copying and direct-map
   cache ownership. Do not widen the 24-second document deadline or witness budget.
4. Once runtime completes, move to a size-matched model/data hypothesis for the
   structural regression rather than another infrastructure replay.

Both hosted Lean jobs passed for the c1725 repair. No checkpoint was created or
promoted in c1726, so no model-card or README checkpoint update is required.
Machine-readable evidence is in
[`autotrain-cycle-1726-disposable-branch-repair.json`](autotrain-cycle-1726-disposable-branch-repair.json).
