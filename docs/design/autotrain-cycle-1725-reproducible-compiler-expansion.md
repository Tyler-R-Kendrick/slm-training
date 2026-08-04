# Autotrain c1725: reproducible compiler expansion

**Verdict:** the frozen batch-size-1 arm reproduced the c1724 finalized timeout
with identical grammar-work counts. This rules out a transient supervisor or host
failure and isolates deterministic completion-forest expansion as the next model-build
repair boundary. Quality remains incomplete and no promotion is authorized.

## Result matrix

| Arm | Records | Parse | Binder F1 | Meaningful | Structure | p50 completed | Timeout | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 3/3 | 1.0 | 1.0 | 0.3333 | 0.3656 | 2,940.73 ms | 0 | Complete fixture control; ship gates fail |
| frozen batch1 | 2/3 | 1.0 | 1.0 | 0.5 | 0.1869 | 8,986.38 ms | 1 | Incomplete quality; runtime repair required |

The batch-size-1 quality rates cover only two completed documents. They cannot be
compared as though the timed-out third document succeeded. Even on that favorable
subset, structural similarity is 49% below control and p50 latency is 3.06 times
control.

## Reproduction matrix

| Candidate signal | c1724 | c1725 | Interpretation |
| --- | ---: | ---: | --- |
| completed / expected | 2 / 3 | 2 / 3 | Same typed timeout boundary |
| neural forwards | 115 | 115 | Exact match |
| unique completion states | 264,558 | 264,558 | Exact match |
| parser forks | 278,189 | 278,189 | Exact match |
| transition hit / miss | 2,087 / 265,991 | 2,087 / 265,991 | Exact match |
| witness states expanded | 13,728 | 13,728 | Exact match |
| compiler | 42.714 s | 42.483 s | Stable deterministic dominant cost |
| backbone | 10.473 s | 10.125 s | Stable secondary cost |

The identical discrete counts across independent cycles are stronger evidence than
wall-clock timing alone: the frozen model repeatedly asks the canonical compiler to
construct the same expansion graph. No deadline, grammar gate, search budget, or
unconstrained fallback should be widened.

## Canonical runtime repair

The hot path already forks each candidate parser branch before a verified direct
token feed. The direct-feed API then took another rollback snapshot, including a
second Lark control-state clone, even though a rejected candidate branch is discarded
immediately. That nested snapshot is observable only as runtime and allocation cost.

The repair marks only these already-isolated completion-session and completion-forest
branches as disposable on rejection. Reusable engine callers retain transactional
rollback by default. Candidate order, V1 payloads, node-budget debit, tri-state
`UNKNOWN`, exact grammar authority, and fail-closed behavior are unchanged. Regression
coverage proves the disposable path takes no rollback snapshot while its source engine
remains exact and reusable.

## Next-run priorities

1. Replay the identical frozen manifest after this repair. Require all three AgentV
   dispositions to complete with zero decode timeouts before comparing quality.
2. Compare discrete grammar-work counts to c1724/c1725. They should remain identical;
   a changed count would indicate an authority or ordering regression, not a speedup.
3. Measure compiler time and p50 including incomplete records. If the nested-copy
   removal is insufficient, profile the remaining control-fork and prefix materialization
   costs before proposing any semantic quotient.
4. Only after runtime completes, preregister a size-matched model/data experiment for
   the observed structural regression. Do not buy quality with capacity or gate drift.

No checkpoint was created or promoted, so no model-card or README checkpoint update
is required. Machine-readable evidence is in
[`autotrain-cycle-1725-reproducible-compiler-expansion.json`](autotrain-cycle-1725-reproducible-compiler-expansion.json).
