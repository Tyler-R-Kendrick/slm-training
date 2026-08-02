# Autotrain c1746: promotion-suite wall timeout reproduces after replay repair

**Verdict:** the c1741-c1745 `_apply_frozen_replay` slug fix works — cycle 6
correctly replayed the frozen `component-edge` promotion arm (no more
`unsupported automatic frozen replay arm` crash) — but the `smoke,held_out`
promotion-suite evaluation for both arms hit the same repository
`MAX_RUN_MINUTES` wall cap seen in c1744 and was killed before a scoreboard
was written. This is the **second** consecutive occurrence of this exact
blocker on the `continuous-loop-20260802-continuous-openui-local-8c0b60dd`
promotion lineage (c4, c6); the loop rule stops only after three, so this
cycle documents and continues rather than blocking.

## Result matrix

| Arm | Suites | Status | Verdict |
| --- | --- | --- | --- |
| control | smoke,held_out | `wall_timeout`, empty metrics, `killed` | incomplete |
| component-edge | smoke,held_out | `wall_timeout`, empty metrics, `killed` | incomplete |

`held_out.structural_similarity` improvement is 0.0 (control=candidate=
0.4167, both partial/pre-timeout) — not a measured model delta.

## Signals and next run

- Root cause is very likely the same class as
  [`autotrain-cycle-1723-supervisor-budget.md`](autotrain-cycle-1723-supervisor-budget.md):
  the `smoke,held_out` promotion suite does not fit the per-arm wall share
  under `MAX_RUN_MINUTES=3` on this container's CPU decode speed, independent
  of the frozen-replay slug bug fixed this cycle.
- If a **third** consecutive `wall_timeout` occurs on this lineage, escalate
  to a typed `repair_harness` action for the promotion-suite wall-budget
  allocation (do not keep blindly retrying the identical timeout a fourth
  time).
- The frozen-replay slug repair itself is confirmed working infrastructure;
  no further action needed there.

No checkpoint was promoted; `docs/MODEL_CARD.md` / README are unchanged.
