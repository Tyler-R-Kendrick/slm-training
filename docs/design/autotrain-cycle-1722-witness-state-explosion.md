# Autotrain c1722: terminal-witness state explosion

**Verdict:** fixture/scratch measurement incomplete. Both 1,608,962-parameter
training stages reused immutable, size-matched checkpoints. The control completed
honest smoke evaluation and failed ship gates. The batch-size-1 arm was interrupted
before a scoreable evaluation, so no arm comparison or promotion is authorized.

## Result matrix

| Arm | Training | Smoke result | Runtime signal | Disposition |
| --- | --- | --- | --- | --- |
| control | Reused c1717 control | n=3; parse 1.0; binder F1 1.0; meaningful 0.3333; structure 0.3656; p50 2,768.64 ms | 4.062 s compiler, 3.497 s backbone, 31,405 completion states, 58 forwards | Honest gates fail; model evidence only |
| batch1 | Reused c1716 batch1 | Non-scoreable partial progress: 2 records complete and `smoke_callout_01` active | 34.716 s compiler, 7.382 s backbone, 221,688 states, 232,277 parser forks, 83 forwards | Frozen replay still required |

The handoff remains `inconclusive` and binds retry measurement to frozen manifest
`1a29206c015305e3714134675bf867f7c7351bac755ccaedde021675258b06e1`.
The partial sidecar has schema `DecodeProgressV1`, `measurement_complete=false`,
and `scoreable=false`; its three stats rows are two completed documents plus the
active partial document, not three quality observations.

## Diagnostic matrix

| Signal | control | batch1 partial | What it says |
| --- | ---: | ---: | --- |
| compiler / backbone | 4.062 / 3.497 s | 34.716 / 7.382 s | Compiler dominates the stopped arm |
| neural forwards | 58 | 83 | Forward count did not grow in proportion to wall time |
| unique completion states | 31,405 | 221,688 | Speculative grammar state volume is the principal scaling signal |
| parser forks | 32,726 | 232,277 | Branch construction scales with the state explosion |
| transition hit / miss | 213 / 31,575 | 1,476 / 222,879 | About 99.3% of batch1 transitions were misses |
| witness states expanded | 1,395 | 10,461 | Bounded top-level searches multiply into many forest builds |

This is a workload diagnosis, not a proof that one leaf function is the hotspot.
It does establish that another neural-prefill optimization is the wrong next target.

## Rejected hypothesis

A dirty, same-checkpoint diagnostic added exact, request-local terminal-witness memo
entries keyed by state, room, and remaining node budget. Completed supported and
unsupported results replayed their historical node debit; `UNKNOWN` remained
uncacheable and fail-closed. Exact V1 differential tests and the mathlib-free Lean
mirror passed, but the same three-document replay still timed out one document:

| Memo probe | Result |
| --- | ---: |
| hits / misses | 16 / 233,190 |
| historical nodes charged on hits | 71 |
| compiler / backbone | 42.727 / 10.262 s |
| unique states / parser forks | 263,128 / 276,732 |

The hypotheses' reuse rate is too small to be meaningful, so the implementation and
its Lean theorem mirror were removed rather than shipped. The local OpenUI Mathlib
build reached 1,004/2,954 jobs before the canonical cap and was interrupted; it is
not proof evidence. No theorem, axiom, model parameter, gate, deadline, or grammar
authority changed.

## Next-run priorities

1. Measure branching by candidate role and prefix position, especially 64-way
   structural-id and 17--27-way component sites, without persisting model-facing
   external names as authority.
2. Preregister a size-matched model experiment for excessive continuation/EOS
   calibration: the stopped checkpoint emits long nested programs, causing exact
   witness work to compound. Keep legality and finalize certification unchanged.
3. Pursue an exact quotient or certified closure proof only if it preserves ordered
   V1 payloads and has a Lean statement covering payload realization, budget debit,
   and `UNKNOWN` fail-closed behavior.
4. Require both Lean suites, complete frozen replay, AgentV, and honest ship gates
   before promotion. No checkpoint was written, so no model-card update is required.

Eval commit: `cb4e49e1a41e7df55519b0a9e89b4c071a914842`
(`harness.model_build.eval=v73`, `model.twotower=v275`). Machine-readable values
and artifact boundaries are in
[`autotrain-cycle-1722-witness-state-explosion.json`](autotrain-cycle-1722-witness-state-explosion.json).
