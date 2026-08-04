# Autotrain c1724: finalized internal decode timeout

**Verdict:** supervisor repair confirmed; model comparison remains incomplete. The
70-second symmetric arm allocation let AgentV finalize all three record dispositions
for both size-matched, 1,608,962-parameter checkpoint replays. The frozen batch-size-1
arm produced two completed documents and one typed runtime timeout, so its reported
quality rates cover only the completed subset and cannot authorize promotion.

## Result matrix

| Arm | Records | Parse | Binder F1 | Meaningful | Structure | p50 completed | Timeout | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 3/3 | 1.0 | 1.0 | 0.3333 | 0.3656 | 2,913.22 ms | 0 | Complete fixture control; ship gates fail |
| frozen batch1 | 2/3 | 1.0 | 1.0 | 0.5 | 0.1869 | 9,109.77 ms | 1 | Incomplete quality; runtime repair required |

The apparently favorable parse, binder, and meaningful rates are not enough: they
exclude the timed-out document. On the completed subset, structural similarity is
49% below control and p50 latency is 3.13 times control. The batch-size-1 arm is not
a quality win and is not ship evidence.

## Runtime matrix

| Signal | control | frozen batch1 | Ratio / interpretation |
| --- | ---: | ---: | --- |
| total decode work | 8.981 s | 53.665 s | 5.98x |
| compiler | 4.041 s | 42.714 s | 10.57x; dominant cost |
| backbone | 4.595 s | 10.473 s | 2.28x |
| neural forwards | 58 | 115 | 1.98x |
| unique completion states | 31,405 | 264,558 | 8.42x |
| parser forks | 32,690 | 278,189 | 8.51x |
| transition hit / miss | 398 / 31,575 | 2,087 / 265,991 | Candidate miss rate remains about 99.2% |
| witness states expanded | 1,395 | 13,728 | 9.84x |

This confirms the c1723 supervisor fix: the evaluator completed in about 56 seconds
inside its 70-second campaign wall and emitted AgentV plus gates artifacts. It also
isolates the next failure inside constrained model-build decoding. No deadline,
grammar gate, or unconstrained fallback should be widened.

## E2E control repair

The old handoff treated a finalized AgentV timeout scoreboard like a supervisor
interruption and spent another generic frozen retry. The canonical supervisor now
recognizes the narrower boundary only when all of these hold:

1. AgentV reports zero execution errors and authoritative gates fail.
2. Completed plus incomplete document counts equal suite `n`.
3. At least one incomplete document has the typed runtime-timeout disposition.

That boundary remains measurement-incomplete for quality, but immediately routes a
receipt-blocking `model_build` repair to `improve-openui-harnesses`; the exact frozen
retry stays queued behind the repair. This lets runtime evidence improve the harness
without mislabeling the model or repeating an unchanged timeout.

## Next-run priorities

1. Instrument completion branching by candidate role and prefix position; prioritize
   the 8.4x state and 8.5x parser-fork expansion over neural-prefill work.
2. Preregister an exact quotient/closure or continuation-control hypothesis only if
   ordered V1 payloads, node-budget debit, and `UNKNOWN` fail-closed semantics remain
   unchanged and have Lean coverage.
3. Replay this exact frozen arm after a measured canonical runtime repair. Require
   three completed documents and zero decode timeouts before comparing quality.
4. If runtime completes, address the observed structural regression with a
   size-matched model/data experiment; do not buy quality with capacity or gate drift.

Both Lean jobs passed on the c1723 supervisor repair. No checkpoint was created or
promoted in c1724, so no model-card or README checkpoint update is required.
Machine-readable evidence is in
[`autotrain-cycle-1724-finalized-decode-timeout.json`](autotrain-cycle-1724-finalized-decode-timeout.json).
