# Continuous autotrain: 2026-08-03 cycle 2, second container (positive, already-delivered corroboration)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `dd2628de` (docs commit from this container's c1 replay, on
top of `main` tip `318492c5`, which already includes PR #1369/#1370)

See
[`continuous-openui-20260803-c1-container2-results.md`](continuous-openui-20260803-c1-container2-results.md)
for why this container's local cycle counter restarted at `c1`/`c2` and
reused the same campaign-id strings as an earlier container's same-day
cycles.

| Arm | Params | Seed | structural_similarity | component_type_recall | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 1,755,764 | 100002 | .32667 | 0 | 11120.23 |
| component-plan | 1,755,764 | 100002 | .38280 | .16667 | 9101.53 |

**Verdict:** `component-plan` again beats its size-matched control on the
declared primary (+.05613, identical to the earlier-container measurement in
[`continuous-openui-20260803-c2-results.md`](continuous-openui-20260803-c2-results.md)),
which was already delivered as a stacked PR
([#1369](https://github.com/Tyler-R-Kendrick/slm-training/pull/1369),
`9dcfa7e68eab9a1055a6585da759511a3b755bd0`, merged via
[#1370](https://github.com/Tyler-R-Kendrick/slm-training/pull/1370)).
`meaningful_program_rate`, `binder_reference_f1`, and `placeholder_fidelity`
are again 0 on both arms. Ship gates fail as expected (`insufficient_n`,
missing `held_out`/`adversarial`/`ood`/`rico_held` suites).

## SDLC Phase A

**Positive**, but `stack_action=positive_no_tracked_delta_skip_stack`
(`has_tracked_delta=false`): the driver itself recognizes this win has no new
code/docs delta beyond what PR #1369 already delivered, so **no new stack
layer opens for this cycle**. This corroboration is one additional data point
toward the still-open fresh-seed-confirmation requirement for `component-plan`
(see next priorities below), not a fresh delivery.

## Next priorities (ranked by the driver)

1. Confirm the `component-plan` candidate on a fresh seed with the exact
   size-matched treatment/control recipe before promotion (confidence 0.95).
2. Keep promotion formal preflight locked until fresh confirmation
   establishes a champion (confidence 1.0, lean assumption).

Machine evidence:
[`continuous-openui-20260803-c2-container2-results.json`](continuous-openui-20260803-c2-container2-results.json).
