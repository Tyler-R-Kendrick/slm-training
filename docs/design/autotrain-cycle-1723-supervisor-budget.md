# Autotrain c1723: supervisor arm-budget exhaustion

**Verdict:** fixture/scratch measurement incomplete. Both 1,608,962-parameter
training stages reused immutable, size-matched checkpoints. The control completed
honest smoke evaluation and failed ship gates. The frozen batch-size-1 arm was
interrupted before its evaluator could finalize a scoreable result, so no model
comparison or promotion is authorized.

## Result matrix

| Arm | Training | Smoke result | Runtime signal | Disposition |
| --- | --- | --- | --- | --- |
| control | Reused c1717 control | n=3; parse 1.0; binder F1 1.0; meaningful 0.3333; structure 0.3656; p50 2,643.74 ms | 4.052 s compiler, 4.199 s backbone, 31,405 states, 58 forwards | Honest gates fail; fixture model evidence only |
| frozen batch1 | Reused c1716 batch1 | Non-scoreable partial progress: 2 records complete and `smoke_callout_01` active | 34.421 s compiler, 7.854 s backbone, 214,614 states, 225,127 parser forks, 83 forwards | Harness repair required before exact replay |

The handoff is `inconclusive`, preserves frozen manifest
`660b0d42508d9bbcda9b91f43dec4b3a2a050d4d0e783d3e9b5adc66c1e353f2`,
and emits a receipt-blocking `repair_harness` action before the queued exact retry.
The partial `DecodeProgressV1` artifact has `measurement_complete=false` and
`scoreable=false`; its three stats rows are two completed documents plus one active
partial document, never three quality observations.

## Harness diagnosis

| Budget boundary | c1723 value | Consequence |
| --- | ---: | --- |
| Repository command cap | 180 s | Canonical hard ceiling |
| Driver arm campaign wall | 51.667 s | One third of usable harness wall |
| Actual candidate stage interval | 44.520 s | Reuse bookkeeping plus bounded-process interrupt grace reduced evaluator time |
| Partial decode work captured | 42.699 s | Evaluator was interrupted while final record remained active |
| Prior complete same-checkpoint probe | about 57 s | The arm can finish within the repository cap but not the allocated share |

The driver divided the usable wall into three equal shares and then also required a
separate 15-second finalization reserve. That stranded roughly 37 seconds even
though the two decision arms are serialized. The candidate consequently received
only about 44.5 seconds of execution before the bounded-process interrupt. This is
an orchestration-budget defect, not evidence against the model.

The canonical repair allocates the usable wall across the two decision arms after
retaining the finalization reserve: 70 seconds per arm under the current constants.
Before either arm starts, the driver still proves that both full shares plus the
reserve fit in the remaining cycle wall. The frozen arm's checkpoint, seed, data,
24-second per-record decode timeout, endpoint, levers, and manifest digest remain
unchanged. Measurement remains serialized and fail closed.

## Diagnostic matrix

| Signal | control | batch1 partial | What it says |
| --- | ---: | ---: | --- |
| compiler / backbone | 4.052 / 4.199 s | 34.421 / 7.854 s | Compiler expansion remains the dominant model-path cost |
| neural forwards | 58 | 83 | Forward count does not explain the stopped wall |
| unique completion states | 31,405 | 214,614 | Grammar completion state volume remains the principal optimization signal |
| parser forks | 32,690 | 225,127 | Parser branching closely follows unique-state growth |
| transition hit / miss | 398 / 31,575 | 1,476 / 215,766 | Candidate transitions are approximately 99.3% misses |
| witness states expanded | 1,395 | 10,424 | Exact witness exploration compounds on the longer program |

The budget repair is necessary to obtain a complete measurement; it does not solve
the underlying completion-state explosion. Those compiler signals remain the next
performance hypothesis area after the exact replay establishes a scoreable baseline.

## Next-run priorities

1. Replay the exact frozen arm under the repaired symmetric supervisor budget and
   require a complete AgentV scoreboard before any model disposition.
2. If the evaluator completes with a typed per-record timeout, treat that as complete
   timeout evidence and prioritize exact branching reduction—not another wall change.
3. Measure branching by candidate role and prefix position, then preregister a
   size-matched continuation/EOS calibration arm without weakening I1-I6 legality.
4. Require Lean and Lean-formal CI for any exact completion optimization; `UNKNOWN`
   remains fail closed and no heuristic may become grammar authority.

No checkpoint was created or promoted, so no model-card or README checkpoint update
is required. Machine-readable values and claim boundaries are in
[`autotrain-cycle-1723-supervisor-budget.json`](autotrain-cycle-1723-supervisor-budget.json).
