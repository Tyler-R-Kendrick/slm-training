# Continuous autotrain: 2026-08-04 (session zflz88, scheduled) cycle 3 — decode-timeout reproduces on retry (2nd occurrence)

**Loop:** `continuous-openui-scheduled-zflz88`
**Campaign:** `continuous-loop-20260804-continuous-openui-schedu-486913c8-c3`
**Integration commit:** `2ee5621a` (post-c2-docs)
**Intent:** `retry_measurement` — identical frozen `c2-{control,component-plan}` arms replayed per the c2 handoff.

**Verdict:** reproduces the identical `decode_timeout_count=3/3` on both arms.

| Arm | Params | decode_timeout_count | compiler_ms_mean (partial) |
| --- | ---: | ---: | ---: |
| control | 1,755,764 | 3/3 | 23,285.4 ms |
| component-plan | 1,755,764 | 3/3 | 23,227.2 ms |

## Second consecutive occurrence — confirms the c2 diagnosis

This is a straight retry of the exact frozen c2 arms, same commit lineage,
same code. It reproduces the same measurement-incomplete result. That rules
out a one-off container blip on c2 specifically and confirms the diagnosis
in
[`continuous-openui-scheduled-zflz88-c2-results.md`](continuous-openui-scheduled-zflz88-c2-results.md):
this container's current CPU headroom cannot fit a 1,755,764-param
train(20 steps)+eval(3 records) inside the fixed 180s `MAX_RUN_MINUTES`
cap today, on either arm.

Per the `autotrain` continuous loop law, the same hard blocker reported
three consecutive times with no new information is required before
declaring the loop blocked. This is occurrence 2 of 3. Rather than spend a
third identical retry (which would add confirmation but no new
information), this session pivots any further cycles to a distinct,
smaller-capacity hypothesis so real model signal is produced instead of
exhausting the session on a resource ceiling no code change can fix.

## SDLC Phase A

**Non-positive** (`measurement_incomplete`). No stack layer opens.

## Next priorities

1. Do not retry the identical 1,755,764-param pair a third time this
   session; a future session on a less-loaded container is the right venue.
2. Screen a distinct, smaller-capacity (≤1.6M param) hypothesis for any
   remaining cycles this session.
3. If a future session reproduces this on size-matched (>1.6M param) arms a
   third independent time under unchanged code, escalate to
   `improve-openui-harnesses` for a genuine decode-budget allocation fix
   (e.g. per-arm wall-time reservation proportional to param count).

Machine evidence:
[`continuous-openui-scheduled-zflz88-c3-results.json`](continuous-openui-scheduled-zflz88-c3-results.json).
