# Continuous autotrain cycle 4 results (2026-08-02, loop `continuous-openui-local-2`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local-2` (predecessor cycle: c3) |
| Campaign | `continuous-loop-20260802-continuous-openui-local--c94ddb78-c4` |
| Cycle role / intent | `promotion` / `confirm` -- first champion-confirmation attempt this lineage |
| Cycle intent | Confirm c3's driver-tagged POSITIVE `component-edge` efficiency-win against a freshly matched control |
| Upstream / integration | `b8188a49` / `791b0a0e` |
| Device | CPU |
| Train | `wf_smoke_v2`, `steps=20` (recorded 20), record_count=101 |
| Eval | `e938_role_safe_all_targets_v2`, suites `smoke,held_out` |
| Budget | `max_wall_minutes=1.1667` (~70s) per experiment |

## This is a champion-confirmation cycle, not a fresh screen

c3's `component-edge` lever was auto-enqueued into
`outputs/autoresearch/loops/continuous-openui-local-2/champion_queue.jsonl`
(`entry_id=champ-continuous-openui-local-2-3-0d77af2fb9002464`) on the strength
of the driver's own `positive=true` efficiency-win tag from c3
([`continuous-openui-local-2-20260802-c3-results.md`](continuous-openui-local-2-20260802-c3-results.md)).
That same c3 report reclassified the result `NON_POSITIVE` as a pure latency
blip -- every quality metric was bit-identical and the primary endpoint moved
by exactly `0.0`. c4 is the driver's queued attempt to confirm that champion
with a proper `--ship-gates` scoreboard (`eval_suites=smoke,held_out`, wider
than c1-c3's smoke-only screens) rather than a new lever being freely picked.

## Smoke suite completed and tied again (`n=3`)

| Metric | control | confirm | delta |
| --- | --- | --- | --- |
| `parse_rate` | 1.0 | 1.0 | 0.0 |
| `meaningful_program_rate` | 0.3333 | 0.3333 | 0.0 |
| `structural_similarity` | 0.4167 | 0.4167 | **0.0 (tied)** |
| `binder_reference_f1` | 0.9524 | 0.9524 | 0.0 |
| `ast_beq_rate` | 0.0 | 0.0 | 0.0 |
| `canonical_beq_rate` | 0.0 | 0.0 | 0.0 |
| `reward_score` | 0.9360 | 0.9360 | 0.0 |
| `component_type_recall` | 0.25 | 0.25 | 0.0 |
| `placeholder_fidelity` | 0.9167 | 0.9167 | 0.0 |
| `latency_ms_p50` | 9946.03 | 11152.92 | +1206.89 (confirm slower) |
| `latency_ms_p95` | 12897.92 | 12339.89 | -558.03 |

Training completed cleanly on both arms (`steps_actual=20`,
`stopped_on=steps`, `record_count=101`, control `last_loss=13.395`, confirm
`last_loss=14.686` -- the lever was genuinely exercised at the loss level).
The smoke suite decoded fully on both arms with `decode_timeout_count=0` /
`decode_timeout_rate=0.0`.

## But the confirmatory `held_out` suite never finished on either arm

Both arms' `evaluate_model --suites smoke,held_out` stage was killed by the
harness's own wall-time guard mid-suite:

```
error: stage exceeded wall-time limit: .../scripts.evaluate_model --test-dir
  e938_role_safe_all_targets_v2 ... --suites smoke,held_out
  --decode-timeout-seconds 24.0 --local-files-only
exit_code: null
status: stopped
```

on `c20260802-continuous-openui-local--c94ddb78-c4-control` and
`c20260802-continuous-openui-local--c94ddb78-c4-confirm` alike. No full
`--ship-gates` scoreboard was ever written for either arm
(`measurement_complete=false`). This is a stage-level wall-clock budget
exhaustion -- `held_out` has more documents than `smoke`'s `n=3` and the
per-experiment budget here (`max_wall_minutes=1.1667`, ~70s) was not enough
to also finish `held_out` after the ~10-12s smoke suite and full training.

**Important honesty note:** `sdlc_delivery.json` and `cycle_handoff.json`
report `held_out.structural_similarity: control=0.4166666666666667
candidate=0.4166666666666667 improvement=0.0`. This is **not** a genuine
held_out measurement -- it is the completed *smoke*-suite value carried
forward by the harness as a placeholder when `held_out` never wrote. Do not
read this as "held_out also tied"; read it as "held_out was never measured
this cycle." The driver's own reasons list correctly also flags
`measurement_incomplete` and `wall_timeout` for both arms alongside that
number, so nothing here contradicts the driver -- this doc is simply making
explicit what the placeholder value means.

## SDLC Phase A: driver and loop policy agree -- `NON_POSITIVE`

```text
SDLC_PHASE_A NON_POSITIVE campaign=continuous-loop-20260802-continuous-openui-local--c94ddb78-c4
  reason=measurement_incomplete:c20260802-continuous-openui-local--c94ddb78-c4-control:missing_scoreboard
  reason=measurement_incomplete:c20260802-continuous-openui-local--c94ddb78-c4-confirm:missing_scoreboard
  reason=wall_timeout:3e160bc142d5d83191b7dc7f9b38c3438df78548b0558bf26f05221224ad3366
  reason=wall_timeout:27c833cdcdec3dab0f315660294b1b1513d4fec01480668c6f1747452f8e6d27
  reason=primary_metric_null_or_worse:held_out.structural_similarity:control=0.4166666666666667 candidate=0.4166666666666667 improvement=0.0
```

`sdlc_delivery.json`: `positive=false`, `stack_layer=false`,
`stack_action=no_stack_layer_non_positive`, `measurement_complete=false`,
`fixture_volume_gate_hits=0`. Unlike c3 (where the driver tagged `positive`
and this doc reclassified `NON_POSITIVE`), here the driver and the loop's
quality policy agree from the start: there is no efficiency-win rule in play
because the measurement never completed, and a missing scoreboard is never
positive evidence either way.

`climb_state=rejected`, `ship_state=blocked` in `cycle_handoff.json` --
correctly conservative given the incomplete measurement.

## Handoff actions

`cycle_handoff.json` emitted two actions, no `repair_harness` (empty
`harness_signals: []` -- the wall-timeout is a budget/scale issue, not a
canonical-family bug to diagnose):

1. `retry_measurement` (owner `autotrain`, `frozen_manifest_sha256
   =d136c228a5a967e6884abdbfd2d5de46f6700bd1a110ccdd88fc772a07d30a1f`) --
   a steering/execution action per `contracts.md`, not a predecessor
   prerequisite; left queued. Per `contracts.md`, an unacknowledged
   `retry_measurement` is consumed automatically before any new model
   hypothesis: the next supervised cycle should replay this identical frozen
   arm rather than propose a fresh lever.
2. `document` (owner `documenting-experiment-results`) -- **acknowledged
   this cycle** with this doc pair as evidence.

No `deliver_stack` action this cycle (unlike c3) -- `stack_action
=no_stack_layer_non_positive` is the plain non-positive path with nothing to
even attempt acknowledging.

`checkpoint_documentation_required=true` -- `docs/MODEL_CARD.md` and the
README model-card summary updated with a screening-note history line (not a
new roster entry; confirmation still pending, `confirm_attempts=1`,
`promote_attempts=0` in `champion_queue.jsonl`).

## Recommendation for cycle 5

1. **infrastructure (priority):** consume the queued `retry_measurement`
   first -- replay the identical frozen manifest
   (`d136c228a5a967e6884abdbfd2d5de46f6700bd1a110ccdd88fc772a07d30a1f`) rather
   than proposing a new lever, per `contracts.md`. Consider narrowing
   `eval_suites` back to `smoke`-only for the retry (matching c1-c3, which
   all completed comfortably) or raising the per-experiment wall budget --
   smoke alone took ~10-12s of the ~70s budget, so `held_out` alone pushed
   the stage over the limit.
2. **model:** no new model evidence either way this cycle. c3's
   `component-edge` lever remains an **unconfirmed** driver-tagged
   `POSITIVE` (already reclassified `NON_POSITIVE` by this lineage's own
   quality policy on the smoke-only evidence) pending a completed confirm
   measurement. Do not count this cycle's tied `held_out` numbers as a
   second independent null -- they are an unmeasured placeholder, not data.
3. **evaluation:** `e938_role_safe_all_targets_v2` remains the right
   snapshot; the blocker this cycle is wall-clock budget for a two-suite
   confirm run, not the eval snapshot choice.
4. **lever-bank health (carried from c3, still open):** with c1 (`bounds`,
   tied null), c2 (`component-plan`, confirmed regression), and c3
   (`component-edge`, tied null on smoke, still unconfirmed on held_out) now
   touched in this lineage, and the parked `continuous-openui-local` lineage
   having independently screened a largely overlapping set across its own 13
   cycles, the pool of genuinely fresh single-lever screening candidates
   remains thin. This cycle's infrastructure timeout doesn't change that
   audit recommendation -- still worth confirming before c6 whether fresh
   single-lever candidates remain, or whether the loop should move to
   combination/second-order arms, or invest in a larger per-experiment wall
   budget so two-suite confirmations can complete.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local--c94ddb78-c4/`
- Runs: `.../runs/c20260802-continuous-openui-local--c94ddb78-c4-{control,confirm}/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- Champion queue entry: `outputs/autoresearch/loops/continuous-openui-local-2/champion_queue.jsonl` (`entry_id=champ-continuous-openui-local-2-3-0d77af2fb9002464`)
- JSON twin: `continuous-openui-local-2-20260802-c4-results.json`
- Predecessor cycle: [`continuous-openui-local-2-20260802-c3-results.md`](continuous-openui-local-2-20260802-c3-results.md)
