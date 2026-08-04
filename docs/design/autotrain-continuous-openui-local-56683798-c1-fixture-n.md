# Autotrain continuous-openui-local-56683798 cycle 1: fixture-n rejection on current main

**Verdict:** reject (fixture ship-gate, not a model regression). First cycle of
a fresh loop lineage (`continuous-openui-local-56683798`, chosen to avoid
colliding with the concurrently-active `continuous-openui-local` lineage that
already landed cycles via [#1425](https://github.com/Tyler-R-Kendrick/slm-training/pull/1425)
/ [#1429](https://github.com/Tyler-R-Kendrick/slm-training/pull/1429) on this
same objective), run against current `main` (`0abaf07`) after this session's
own AgentV/NODE_OPTIONS fix ([PR #1354](https://github.com/Tyler-R-Kendrick/slm-training/pull/1354))
was closed as superseded by the already-landed #1429/#1360/#1425.

Both 21-step CPU scratch arms (1,608,962 trainable params, `wf_smoke_v2`)
trained and completed a full AgentEvals scoreboard:

| Arm | Parse | Meaningful | Struct sim | Binder F1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 1.0 | 0 | .0575 | .63333 | 4,153.06 |
| bounds | 1.0 | 0 | .0575 | .63333 | 4,715.05 |

Matched-knob arms, identical `smoke.structural_similarity` — no attributable
delta this cycle. This confirms the already-landed harness fix works cleanly
on current `main`: measurement completed, ship gates correctly reject on
`smoke:insufficient_n` (`n=3` vs required `>=20`) and the quality thresholds,
which is expected for a tiny screening run, not an infrastructure failure or
a regression.

Non-positive per SDLC Phase A (`fixture_insufficient_n_alone`): no stacked
PR, local docs/commit only. Next queued hypothesis is the distinct,
size-matched `component-plan` quality lever.

Machine evidence:
[`autotrain-continuous-openui-local-56683798-c1-fixture-n.json`](autotrain-continuous-openui-local-56683798-c1-fixture-n.json).
