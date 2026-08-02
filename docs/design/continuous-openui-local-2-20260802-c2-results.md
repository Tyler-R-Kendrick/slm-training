# Continuous autotrain cycle 2 results (2026-08-02, loop `continuous-openui-local-2`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local-2` (predecessor cycle: c1) |
| Campaign | `continuous-loop-20260802-continuous-openui-local--c94ddb78-c2` |
| Cycle intent | Run the driver's own rank-1 next-run priority from cycle 1 (`component-plan`) against a freshly matched control |
| Upstream / integration | `b8188a49` / `c42a9a63` |
| Device | CPU |
| Train | `wf_smoke_v2`, `steps=20` (recorded 22), seed 100002 |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke`, `n=3` |

## Lever this cycle: `component-plan`, the driver's own choice

Per this cycle's scope, the driver was left to pick freely rather than being
forced onto a specific lever. It selected `component-plan`
(`component_plan_loss_weight`) -- exactly the rank-1 `next_experiment`
priority it emitted at the end of cycle 1
([`continuous-openui-local-2-20260802-c1-results.md`](continuous-openui-local-2-20260802-c1-results.md)).
No override was applied.

`_feasible_eval_version` again picked `eval_version=e938_role_safe_all_targets_v2`
(smoke, `n=3`) automatically. Both arms completed cleanly with
`decode_timeout_count=0` / `decode_timeout_rate=0.0` on both arms.

## Smoke results (`n=3`)

| Metric | control | component-plan (candidate) | delta |
| --- | --- | --- | --- |
| `parse_rate` | 1.0 | 1.0 | 0.0 |
| `meaningful_program_rate` | 0.0 | 0.0 | 0.0 |
| `structural_similarity` | 0.3267 | 0.0964 | **-0.2303 (regression)** |
| `binder_reference_f1` | 0.0 | 0.0 | 0.0 |
| `ast_beq_rate` | 0.0 | 0.0 | 0.0 |
| `canonical_beq_rate` | 0.0 | 0.0 | 0.0 |
| `reward_score` | 0.0 | 0.0 | 0.0 |
| `component_type_recall` | 0.1667 | 0.0833 | -0.0833 |
| `placeholder_fidelity` | 0.0 | 0.0 | 0.0 |
| `latency_ms_p50` | 12582.04 | 1319.82 | -11262.22 (see caveat) |
| `latency_ms_p95` | 14336.07 | 1356.65 | -12979.42 (see caveat) |

`structural_similarity` (the campaign's primary metric) **fell** from the
control (0.3267) to the `component-plan` candidate (0.0964), a genuine
`-0.2303` regression -- not a bit-identical null like cycle 1's `bounds`
lever. `component_type_recall` also regressed. Training loss diverged as
expected between arms (`last_loss` control=14.390, candidate=19.916, both
`steps_actual=22`, `stopped_on=steps`), confirming the lever was genuinely
exercised and this is a real recipe effect, not a config no-op.

**Latency caveat:** the control arm's own `latency_ms_p50` this cycle
(12582ms) is far outside this lineage's prior control latencies (~1.3-1.6s
in cycle 1), which points to local CPU-sandbox host contention during that
specific arm's run rather than a real recipe effect. Latency is **not**
used as supporting evidence for this cycle's classification -- the primary
signal is the `structural_similarity` regression alone.

## SDLC Phase A: `NON_POSITIVE`

```text
SDLC_PHASE_A NON_POSITIVE campaign=continuous-loop-20260802-continuous-openui-local--c94ddb78-c2
  reason=fixture_insufficient_n:c20260802-continuous-openui-local--c94ddb78-c2-control
  reason=fixture_insufficient_n:c20260802-continuous-openui-local--c94ddb78-c2-component-plan
  reason=primary_metric_null_or_worse:smoke.structural_similarity:control=0.32666666666666666 candidate=0.0964 improvement=-0.23026666666666668
  reason=fixture_insufficient_n_alone
```

`sdlc_delivery.json`: `positive=false`, `stack_layer=false`,
`stack_action=no_stack_layer_non_positive`, `measurement_complete=true`,
`fixture_volume_gate_hits=2`. Both arms honestly reject the full ship-gate
scoreboard on `smoke:insufficient_n (actual=3 need>=20)` plus the expected
missing-suite gates (`held_out`, `adversarial`, `ood`, `rico_held`) -- fixture
scope only, exactly as intended, never a ship claim.

Unlike cycle 1's bit-identical null (`bounds` lever, delta exactly 0.0), this
is a **confirmed quality regression**: `component-plan` supervision at this
recipe (seed 100002, `steps=20`) measurably hurts `structural_similarity` and
`component_type_recall` versus the matched control, at this fixture scale.
This mirrors the same lever's confirmed regression in the parked
`continuous-openui-local` lineage's cycle 5 frozen replay (structure
0.2308->0.1725) -- a second, independent confirmation across lineages that
`component_plan_loss_weight` regresses structural quality on this size
model/fixture rather than merely being under-powered to detect an effect.

## Handoff actions

`cycle_handoff.json` emitted two actions, no `repair_harness` (empty
`harness_signals: []` -- this cycle produced no canonical-family signal to
diagnose):

1. `document` (owner `documenting-experiment-results`) -- **acknowledged this
   cycle** with this doc pair as evidence.
2. `next_experiment` (owner `autotrain`, disposition `experiment_next`,
   proposed `component-edge`) -- a steering action, not a predecessor
   prerequisite per `contracts.md`; left un-acked and queued for cycle 3 of
   this lineage, which this session does not run (single supervised cycle
   only, per scope).

`checkpoint_documentation_required=true` -- `docs/MODEL_CARD.md` and the
README model-card summary updated with a screening-note history line (not a
new roster entry; `n=3`, non-positive/regressed, fixture scale only).

## Recommendation for cycle 3

1. **model:** run the `component-edge` candidate next (driver's own rank-1
   priority) -- distinct untested lever, avoids repeating the just-rejected
   `component-plan` regression.
2. **evaluation:** keep `e938_role_safe_all_targets_v2` (smoke, `n=3`) as the
   fixed matched suite for this lineage. Do **not** retarget
   `test_data_scaleup_v1` here -- that stays the separately tracked
   decode-compiler follow-up owned by the `continuous-openui-local` lineage /
   PR #1307.
3. **infrastructure:** no signal to escalate; `_feasible_eval_version`
   performed correctly again. Flag the control arm's outlier latency this
   cycle as sandbox-noise to watch for in future cycles, not a repair target.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local--c94ddb78-c2/`
- Runs: `.../runs/c20260802-continuous-openui-local--c94ddb78-c2-{control,component-plan}/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- JSON twin: `continuous-openui-local-2-20260802-c2-results.json`
- Predecessor cycle: [`continuous-openui-local-2-20260802-c1-results.md`](continuous-openui-local-2-20260802-c1-results.md)
