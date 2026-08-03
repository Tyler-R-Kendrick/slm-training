# Continuous autotrain: 2026-08-03 session 2, cycle 2 (positive, but not a new delta)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c2`
**Base commit:** `5512bad7` (this session's cycle-1 docs commit, on top of `9dcfa7e6`)

| Arm | Params | parse | MPR | structural_similarity | comp-type recall | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,755,764 | 1.0 | 0.0 | .3267 | 0 | 9940.47 |
| component-plan | 1,755,764 | 1.0 | 0.0 | .3828 | .1667 | 7800.08 |

**Verdict: positive on the primary (+0.0561), but not a new tracked delta.**
The driver's arm seed (`100002`) is derived deterministically from
`cycle_index`, not drawn fresh — so this cycle reproduces the already-merged
[#1369](https://github.com/Tyler-R-Kendrick/slm-training/pull/1369) `c2`
finding bit-for-bit on every deterministic metric (only wall-clock latency
differs). It is **not** the independent fresh-seed confirmation that #1369's
own next-priorities called for. The driver's own SDLC Phase A classification
agrees: `positive=true`, `has_tracked_delta=false`,
`stack_action=positive_no_tracked_delta_skip_stack` — **no stacked PR** for
this cycle.

## What this means for promotion

`component-plan` still cannot be promoted: the outstanding requirement is a
measurement under a genuinely different seed, and this driver invocation
does not vary the seed by itself. A future cycle needs an explicit seed
override (or a hypothesizer change that draws seeds independently per
session) to produce that confirmation.

## Next priorities

1. Re-run `component-plan` vs control with an explicitly different seed for
   a real independent confirmation.
2. Keep promotion formal preflight locked until that confirmation exists.
3. Merge [#1351](https://github.com/Tyler-R-Kendrick/slm-training/pull/1351)
   (AgentV NODE_OPTIONS + missing-SDK fix) — this session worked around the
   same defect locally rather than duplicate that diff.

Machine evidence:
[`continuous-openui-20260803-s2-c2-results.json`](continuous-openui-20260803-s2-c2-results.json).
