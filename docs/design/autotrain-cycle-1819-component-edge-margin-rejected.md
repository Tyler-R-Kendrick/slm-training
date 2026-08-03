# Autotrain c1819: component-edge margin regresses quality

**Verdict:** reject `component-edge-margin` at fixture scale. The frozen replay
completed both size-matched arms, and the candidate is faster, but it halves
meaningful-program rate and sharply regresses the declared structural primary
and protected binder F1.

| Arm | Params | Loss | Edge rows at final step | Smoke structure | MPR | Binder F1 | AST / canonical | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reused control | 1,608,962 | 15.24373 | 0 | .404433 | .6667 | .95238 | 0 / 0 | 3772.85 |
| component-edge margin | 1,608,962 | 69.44171 | 2 | .174167 | .3333 | .63333 | 0 / 0 | 1074.57 |

The candidate objective was active. At step 20 it aligned two deterministic
`component_bound` rows over a mean 32 legal candidates, with alignment loss
`38.6861`, CE `19.0933`, margin loss `19.5928`, and a `.5` violation rate. It
adds no parameters and changes neither decoder authority nor the legal domain.
Its faster fixture decode is therefore useful runtime evidence, but not a
quality win.

The v116 driver originally labeled the cycle positive from a 75.6% MPR/ms ratio
gain even while recording the MPR, structure, and binder regressions. That
disposition is invalidated by campaign harness v117: efficiency may not
override regressed meaningful-program rate, the role-owned quality primary, or
protected non-regression metrics. The candidate must not enter confirmation or
promotion.

Both arms parse all three documents but fail unchanged gates, including
evidence volume and AST/canonical equality. The candidate checkpoint
`68c7d02b...cc02f0d` and reused control `b97e7424...b502215` are explicit
no-sync scratch artifacts, never reusable, promotable, syncable, or shippable.
Lean is `not_applicable:retry_measurement`; no theorem or promotion claim is
made.

The next run should prioritize a genuinely new, denser structured objective
only after the v117 classification repair is integrated and the stale c1819
queue disposition is reclassified.

Machine evidence:
[`autotrain-cycle-1819-component-edge-margin-rejected.json`](autotrain-cycle-1819-component-edge-margin-rejected.json).
