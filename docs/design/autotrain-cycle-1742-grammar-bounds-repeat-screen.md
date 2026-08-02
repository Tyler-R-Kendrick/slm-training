# Autotrain c1742: grammar_completion_bounds confirmed null on a second seed

**Verdict:** running clean end to end (AgentV toolchain healthy after the
`npm ci` fix documented in
[c1741](autotrain-cycle-1741-grammar-bounds-latency-regression.md)),
`grammar_completion_bounds` again ties the matched control on every quality
metric and is slower on latency — 5.99% at seed 100002, versus 12.29% at seed
100001. Two independent seeds now reject the hypothesis.

## Result matrix

| Arm | Params | Complete | Timeout | Parse | Meaning | Structure | Binder F1 | Reward | p50 (ms) | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,608,962 | 3/3 | 0 | 1.000 | 0.000 | 0.0964 | 0.0000 | 0.000 | 1,126.16 | fail (fixture n=3) |
| grammar_completion_bounds | 1,608,962 | 3/3 | 0 | 1.000 | 0.000 | 0.0964 | 0.0000 | 0.000 | 1,193.63 | fail (fixture n=3) |

Both arms complete the full 3-record smoke suite with zero decode timeouts.
Ship gates fail on both for the expected fixture-scale reasons
(`smoke:insufficient_n`, n=3 vs the ≥20 floor) plus the standard quality
thresholds; this is not a production-readiness claim either way.

## Signals and next run

- `grammar_completion_bounds` is now rejected on two independent seeds
  (100001, 100002): quality ties exactly on every metric and latency
  regresses in both.
- No canonical harness repair indicated; the toolchain ran clean with zero
  infra errors this cycle.
- Per the driver's own ranked priorities, the next cycle should test a
  distinct, size-matched **component-plan** hypothesis rather than repeating
  the now-exhausted `grammar_completion_bounds` arm.
- Both checkpoints are local fixture-scratch artifacts (`outputs/`,
  gitignored, no sync) — neither reusable, promotable, nor ship.

Machine-readable evidence is in
[`autotrain-cycle-1742-grammar-bounds-repeat-screen.json`](autotrain-cycle-1742-grammar-bounds-repeat-screen.json).
