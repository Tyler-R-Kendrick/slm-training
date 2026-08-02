# Continuous autotrain cycle 3 results (2026-08-02, loop `continuous-openui-local-2`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local-2` (predecessor cycle: c2) |
| Campaign | `continuous-loop-20260802-continuous-openui-local--c94ddb78-c3` |
| Cycle intent | Run the driver's own rank-1 next-run priority from cycle 2 (`component-edge`) against a freshly matched control |
| Upstream / integration | `b8188a49` / `0bbdbb6c` |
| Device | CPU |
| Train | `wf_smoke_v2`, `steps=20` (recorded 20), record_count=101 |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke`, `n=3` |

## Lever this cycle: `component-edge`, the driver's own choice

The task left the driver free to pick rather than forcing a specific lever.
The hypothesis matrix's `recommended_experiment_id` was `component-edge`
(`component_edge_decode_weight` / `structural_aux_head_profile=component-edge`)
and the driver selected it -- exactly the rank-1 `next_experiment` priority
it emitted at the end of cycle 2
([`continuous-openui-local-2-20260802-c2-results.md`](continuous-openui-local-2-20260802-c2-results.md)).
No override was applied.

`_feasible_eval_version` again picked `eval_version=e938_role_safe_all_targets_v2`
(smoke, `n=3`) automatically. Both arms completed cleanly with
`decode_timeout_count=0` / `decode_timeout_rate=0.0` on both arms.

## Smoke results (`n=3`)

| Metric | control | component-edge (candidate) | delta |
| --- | --- | --- | --- |
| `parse_rate` | 1.0 | 1.0 | 0.0 |
| `meaningful_program_rate` | 0.3333 | 0.3333 | 0.0 |
| `structural_similarity` | 0.2308 | 0.2308 | **0.0 (tied)** |
| `binder_reference_f1` | 0.7333 | 0.7333 | 0.0 |
| `ast_beq_rate` | 0.0 | 0.0 | 0.0 |
| `canonical_beq_rate` | 0.0 | 0.0 | 0.0 |
| `reward_score` | 0.8407 | 0.8407 | 0.0 |
| `component_type_recall` | 0.1667 | 0.1667 | 0.0 |
| `placeholder_fidelity` | 0.6389 | 0.6389 | 0.0 |
| `latency_ms_p50` | 3733.40 | 3284.41 | -448.99 (-12.0%) |
| `latency_ms_p95` | 3735.64 | 3851.11 | +115.47 |

Every smoke quality metric came back **bit-identical** between control and
candidate -- not just the primary metric. Training loss did diverge between
arms (control `last_loss=15.298`, candidate `last_loss=16.483`, both
`steps_actual=20`, `stopped_on=steps`, `record_count=101`), confirming the
lever was genuinely exercised at the loss level, but that divergence did not
translate into any change in the smoke-suite decode outputs at this fixture
scale. `latency_ms_p50` dropped by ~12%, but `latency_ms_p95` moved in the
opposite direction, so the latency signal is not a clean unidirectional win
either.

## SDLC Phase A: driver says `POSITIVE`, loop policy reclassifies `NON_POSITIVE`

```text
SDLC_PHASE_A POSITIVE campaign=continuous-loop-20260802-continuous-openui-local--c94ddb78-c3
  reason=fixture_insufficient_n:c20260802-continuous-openui-local--c94ddb78-c3-component-edge
  reason=fixture_insufficient_n:c20260802-continuous-openui-local--c94ddb78-c3-control
  reason=efficiency_win:mpr_per_ms:8.928412e-05->0.00010148956:gain_fraction=0.1367034:minimum=0.05
  reason=quality_held:parse=1.0 mpr=0.3333333333333333
  reason=primary_metric_null_or_worse:smoke.structural_similarity:control=0.23083333333333333 candidate=0.23083333333333333 improvement=0.0
  reason=fixture_insufficient_n_alone
```

`sdlc_delivery.json`: `positive=true`, `stack_layer=false`,
`stack_action=positive_no_tracked_delta_skip_stack`, `measurement_complete=true`,
`fixture_volume_gate_hits=2`. Both arms honestly reject the full ship-gate
scoreboard on `smoke:insufficient_n (actual=3 need>=20)` plus the expected
missing-suite gates (`held_out`, `adversarial`, `ood`, `rico_held`) -- fixture
scope only, never a ship claim.

The driver's automated classifier applied its `mpr_per_ms` efficiency-win
rule: quality held (`parse=1.0`, `mpr=0.3333` exactly at the `>=1/3` floor)
and latency improved by a `gain_fraction` of `0.1367` (>= the `0.05`
minimum), so it tagged the cycle `positive=true`.

**This report reclassifies the cycle as `NON_POSITIVE` for delivery
purposes**, per this loop's own quality-aware policy that "a pure latency
blip is not positive." The efficiency-win rule is meant to reward latency
wins that come alongside *held* quality on a metric that moved for some
other reason -- here, literally every quality metric without exception
(`parse_rate`, `meaningful_program_rate`, `structural_similarity`,
`binder_reference_f1`, `ast_beq_rate`, `canonical_beq_rate`, `reward_score`,
`component_type_recall`, `placeholder_fidelity`) is bit-identical between
arms, and the primary endpoint (`structural_similarity`) moved by exactly
`0.0`. There is no quality signal at all attached to this latency change --
it is a pure latency blip by definition, and `latency_ms_p95` even moved the
wrong direction. This reclassification changes nothing operationally: the
driver's own `stack_layer` was already `false`
(`positive_no_tracked_delta_skip_stack` -- no tracked code delta to stack
regardless), so no stacked PR is skipped or opened differently under either
reading.

Unlike cycle 2's confirmed `component-plan` regression, this is a **tied
null**, structurally similar to cycle 1's `bounds` lever (bit-identical
primary metric) but this time on a different lever
(`component_edge_decode_weight`) and with training loss visibly diverging
between arms -- so the lever was exercised, it just did not move the
smoke-suite decode outputs at `n=3`, `steps=20` scale.

## Handoff actions

`cycle_handoff.json` emitted three actions, no `repair_harness` (empty
`harness_signals: []` -- this cycle produced no canonical-family signal to
diagnose):

1. `document` (owner `documenting-experiment-results`) -- **acknowledged this
   cycle** with this doc pair as evidence.
2. `next_experiment` (owner `autotrain`, disposition `experiment_next`,
   proposed re-running `component-edge`) -- a steering action, not a
   predecessor prerequisite per `contracts.md`; left un-acked. Per the
   lever-bank note below, cycle 4 should rotate to a fresh untested lever
   rather than repeating this one.
3. `deliver_stack` (owner `sdlc`) -- **not applicable, left un-acked**: per
   the `NON_POSITIVE` reclassification above (and matching the driver's own
   `stack_layer=false` either way), no reviewable delta is stacked this
   cycle. The harness's own `_validate_action_evidence` correctly refuses to
   let `deliver_stack` be acknowledged (`completed` or `blocked`) with
   anything other than a commit already merged into `origin/main` -- this
   session neither pushes nor opens a PR, so no such commit exists, and the
   action is intentionally left pending rather than force-acked.

`checkpoint_documentation_required=true` -- `docs/MODEL_CARD.md` and the
README model-card summary updated with a screening-note history line (not a
new roster entry; `n=3`, latency-only tied null, fixture scale only).

## Recommendation for cycle 4

1. **model:** two of the driver's own top-priority levers this lineage
   (`component-plan` in c2, `component-edge` in c3) are now screened -- one
   confirmed regression, one tied null. Rotate to a genuinely untested lever
   from the remaining `_SCREENING_ARM_BANK` (e.g. `bounds`/`canvas`/`both`/
   `steps`/`batch1`/`component-inventory`/`binder-topology`/
   `component-structure`) rather than re-running either of the two already
   screened this lineage.
2. **evaluation:** keep `e938_role_safe_all_targets_v2` (smoke, `n=3`) as the
   fixed matched suite for this lineage. Do **not** retarget
   `test_data_scaleup_v1` here -- that stays the separately tracked
   decode-compiler follow-up owned by the `continuous-openui-local` lineage /
   PR #1307.
3. **infrastructure:** no signal to escalate; `_feasible_eval_version`
   performed correctly again, and this cycle's latencies (3.28-3.73s on both
   arms) sit comfortably within lineage norms -- no host-contention outlier
   this time, unlike cycle 2's control arm.
4. **lever-bank health:** with cycle 1 (`bounds`, tied null), cycle 2
   (`component-plan`, confirmed regression, cross-lineage confirmed), and
   cycle 3 (`component-edge`, tied null) now screened here, and the parked
   `continuous-openui-local` lineage having independently screened a largely
   overlapping set across its own 13 cycles, the pool of genuinely fresh,
   untested single-lever screening candidates across both lineages combined
   is visibly thinning. Recommend a driver-level audit of
   `_SCREENING_ARM_BANK` coverage before cycle 5 to confirm fresh single-lever
   candidates remain, or whether the loop should move to combination /
   second-order arms.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local--c94ddb78-c3/`
- Runs: `.../runs/c20260802-continuous-openui-local--c94ddb78-c3-{control,component-edge}/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- JSON twin: `continuous-openui-local-2-20260802-c3-results.json`
- Predecessor cycle: [`continuous-openui-local-2-20260802-c2-results.md`](continuous-openui-local-2-20260802-c2-results.md)
