# Autotrain continuous-openui-local c4: first complete measurement (fixture, ship gates rejected)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

**Verdict:** cycle 4 of loop `continuous-openui-local` (campaign
`continuous-loop-20260802-continuous-openui-local-8c0b60dd-c4`) is this loop
instance's first cycle to produce a **complete** measurement — both arms
trained, evaluated through AgentV, and hit an honest ship-gate rejection
rather than an infrastructure crash. This confirms the c3 NODE_OPTIONS fix
(`f8abb9d7`) actually unblocks the pipeline end to end.

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c4` |
| Treatment | `grammar_completion_bounds` knob, control vs bounds |
| Train | `wf_smoke_v2`, steps=20/21, batch=2, seed=100001, cpu |
| Eval | `e938_role_safe_all_targets_v2`, suite=smoke |
| Params | 1,608,962 (both arms, size-matched) |

## Result

Both arms produced byte-identical smoke metrics:

| Metric | Control | Bounds | Gate | Pass? |
| --- | --- | --- | --- | --- |
| `n` | 3 | 3 | need ≥20 | ✗ insufficient_n |
| `parse_rate` | 1.0 | 1.0 | — | — |
| `structural_similarity` | 0.0575 | 0.0575 | need ≥0.35 | ✗ |
| `meaningful_program_rate` | 0.0 | 0.0 | need ≥0.66 | ✗ |
| `ast_beq_rate` | 0.0 | 0.0 | need ≥0.2 | ✗ |
| `canonical_beq_rate` | 0.0 | 0.0 | need ≥0.1 | ✗ |
| `binder_reference_f1` | 0.633 | 0.633 | — | — |
| `latency_ms_p50` | 903.7 | 972.1 | — | — |

`held_out` / `adversarial` / `ood` / `rico_held` suites are all
`missing_suite` at this `train_version`. Ship gates reject honestly on
fixture-scale `n=3` (insufficient_n alone) — expected at screening scale, not
a quality regression.

## Delivery classification (SDLC Phase A)

**Not positive** — `fixture_insufficient_n_alone` is explicitly excluded from
the positive-result gate in `autotrain-iteration-delivery.md`. No new stack
layer for this cycle's model result; docs + local commit only.

This is distinct from cycle 3's harness repair (`f8abb9d7`), which **is**
independently positive under the "executable unblocking" criterion (a
harness fix removed a hard blocker and the identical arm then completed with
a usable scoreboard) and is delivered on its own commit/PR regardless of
this cycle's fixture-scale model outcome.

## Next priority

`grammar_completion_bounds` is exhausted for this loop instance (candidate
knob produced no measurable delta at n=3). Next hypothesis:
`component-plan`, per the driver's ranked successor priorities.

Machine evidence:
[`autotrain-cycle-continuous-openui-local-c4-first-complete-measurement.json`](autotrain-cycle-continuous-openui-local-c4-first-complete-measurement.json).
