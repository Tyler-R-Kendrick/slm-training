# Continuous autotrain: 2026-08-03 cycle 5 — frozen-replay confirmation of component-plan structural win

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `bcda93e0` (`repair_harness` resolution commit)

**Verdict:** frozen replay (`cycle_intent=retry_measurement`) of campaign c2's
control/component-plan pair completed cleanly — no decode timeout this
time — and confirms `component-plan` beats its matched control on the
declared primary. Positive per SDLC Phase A, but `has_tracked_delta=false`:
the win is not new code, so no separate stacked PR — documented here on the
same PR (#1374) as the c3/c4 cycles above.

| Arm | Params | Seed | structural_similarity | component_type_recall | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 1,755,764 | 100002 | .32667 | .16667 | 16747.64 |
| component-plan | 1,755,764 | 100002 | .38280 | .16667 | 11584.63 |

Primary improvement `+.05613`; p50 latency improves 30.8%. Both arms parse
all 3 documents at 1,755,764 trainable params. `meaningful_program_rate`,
`binder_reference_f1`, and `placeholder_fidelity` remain **0 on both arms**
— the win is confined to structural similarity, component-type recall, and
latency, not full program correctness.

Ship gates fail as expected: `insufficient_n` (n=3, need 20), and
`held_out`/`adversarial`/`ood`/`rico_held` suites were not run.

## Relation to prior evidence

This replay reused c2's already-trained checkpoints for eval only (no
retrain — c2 itself trained successfully; only its eval timed out, see
[`continuous-openui-20260803-c4-results.md`](continuous-openui-20260803-c4-results.md)).
The resulting metrics are byte-for-byte identical to an earlier, unrelated
session's [`continuous-openui-20260803-c2-results.json`](continuous-openui-20260803-c2-results.json)
record for the same hypothesis, seed, and trainable-param count — consistent
with this being a deterministic recipe (fixed fixture data, fixed seed)
rather than coincidence. This is now at least the second independent
measurement in the same structural direction.

## SDLC Phase A

**Positive** (`primary_metric_win`), but `has_tracked_delta=false` /
`stack_action=positive_no_tracked_delta_skip_stack`: the win itself carries
no new code or docs delta beyond this record, so no separate stack layer is
opened for it.

Full JSON: [`continuous-openui-20260803-c5-results.json`](continuous-openui-20260803-c5-results.json).
