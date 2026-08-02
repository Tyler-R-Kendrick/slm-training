# Continuous autotrain cycle 1 results (2026-08-02, loop `continuous-openui-local-2`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local-2` (fresh lineage, `predecessor_campaign_id=null`) |
| Campaign | `continuous-loop-20260802-continuous-openui-local--c94ddb78-c1` |
| Cycle intent | Screen an untested `_SCREENING_ARM_BANK` lever against the original feasible eval fixture |
| Upstream / integration | `b8188a49` / `ee6ce85c` |
| Device | CPU |
| Train | `wf_smoke_v2`, `steps=20` (21 actual, stopped on steps) |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke`, `n=3` |

## Why a new loop lineage

The prior lineage (`continuous-openui-local`) is correctly self-gated: its
cycle 12 exhausted `max_consecutive_frozen_replays=2` on
`test_data_scaleup_v1` (smoke `n=12`) and emitted an unacknowledged
`repair_harness` action naming `harness_family=model_build`
(decode-compiler performance in `src/slm_training/models/twotower.py`,
`compiler_ms` ~97% of total decode time at ~14.3s/record). Per
`storage._PREREQUISITE_ACTION_KINDS`, `_require_predecessor_actions` blocks
**any** successor cycle in that lineage -- confirmed empirically -- until
that repair has an evidence-bound receipt. Decode-compiler work is
deliberately out of scope for this cycle (invariant-sensitive: "constrained
decoding is the product") and stays tracked separately in the already-open
PR #1307.

`continuous-openui-local-2` starts a fresh lineage (`predecessor_campaign_id:
null`) so no cross-lineage gate applies, and keeps the continuous-training
mandate moving on the remaining untested `_SCREENING_ARM_BANK` levers.

## This cycle: `bounds` lever, driver-picked by thrash rotation

`THRASH_ROTATE cycle=1 recommended=bounds skip=[]` -- the hypothesizer
proposed 11 candidates (`canvas`, `both`, `batch1`, `bounds`,
`component-plan`, `component-edge`, `component-inventory`,
`component-structure`, `binder-topology`, `steps`, `control`) and picked
`bounds` (`grammar_completion_bounds`) as the untested lever to run this
cycle; `binder-topology` was already tested in the prior lineage.

**`_feasible_eval_version` worked as designed.** The prior lineage's c12
harness fix (commit `14ec931`) picked `eval_version=e938_role_safe_all_targets_v2`
(smoke, `n=3`) automatically for this fresh cycle -- exactly the suite that
fits the ~70s/arm wall budget. Result: **both arms completed with zero
decode timeouts** (`decode_timeout_count=0`, `decode_timeout_rate=0.0` for
both `control` and `bounds`), a direct positive confirmation of that guard
working on a genuinely fresh (non-frozen) cycle, in contrast to every
`test_data_scaleup_v1` attempt in the old lineage which hit `wall_timeout`
at `processed_record_n=3/12`.

## Smoke results (`n=3`)

| Metric | control | bounds (candidate) | delta |
| --- | --- | --- | --- |
| `parse_rate` | 1.0 | 1.0 | 0.0 |
| `meaningful_program_rate` | 0.0 | 0.0 | 0.0 |
| `structural_similarity` | 0.0575 | 0.0575 | **0.0 (bit-identical)** |
| `binder_reference_f1` | 0.6333 | 0.6333 | 0.0 |
| `ast_beq_rate` | 0.0 | 0.0 | 0.0 |
| `canonical_beq_rate` | 0.0 | 0.0 | 0.0 |
| `reward_score` | 0.0 | 0.0 | 0.0 |
| `component_type_recall` | 0.0 | 0.0 | 0.0 |
| `latency_ms_p50` | 1557.01 | 1560.99 | +3.98ms |
| `latency_ms_p95` | 1580.79 | 1715.00 | +134.21ms |

`structural_similarity` (the campaign's primary metric) is bit-identical
between arms -- `grammar_completion_bounds` produced no measurable structural
effect at this `n=3` fixture scale. `latency_ms_p95` moved unfavorably
(+134ms) with zero accompanying quality change, which the classifier
correctly treats as **not positive** (a pure latency blip with no meaning is
not a win, per the quality-aware tradeoff rule).

## SDLC Phase A: `NON_POSITIVE`

```text
SDLC_PHASE_A NON_POSITIVE campaign=continuous-loop-20260802-continuous-openui-local--c94ddb78-c1
  reason=fixture_insufficient_n:c20260802-continuous-openui-local--c94ddb78-c1-control
  reason=fixture_insufficient_n:c20260802-continuous-openui-local--c94ddb78-c1-bounds
  reason=primary_metric_null_or_worse:smoke.structural_similarity:control=0.0575 candidate=0.0575 improvement=0.0
  reason=fixture_insufficient_n_alone
```

`sdlc_delivery.json`: `positive=false`, `stack_layer=false`,
`stack_action=no_stack_layer_non_positive`, `measurement_complete=true`,
`fixture_volume_gate_hits=2`. Both arms honestly reject the full ship-gate
scoreboard on `smoke:insufficient_n (actual=3 need>=20)` plus the expected
missing-suite gates (`held_out`, `adversarial`, `ood`, `rico_held`) -- fixture
scope only, exactly as intended, never a ship claim.

## Handoff actions

`cycle_handoff.json` emitted two actions, no `repair_harness` (empty
`harness_signals: []` -- this cycle produced no canonical-family signal to
diagnose):

1. `document` (owner `documenting-experiment-results`) -- **acknowledged this
   cycle** with this doc pair as evidence.
2. `next_experiment` (owner `autotrain`, disposition `experiment_next`,
   proposed `component-plan`) -- a steering action, not a predecessor
   prerequisite per `contracts.md`; left un-acked and queued for cycle 2 of
   this lineage, which this session does not run (single supervised cycle
   only, per scope).

`checkpoint_documentation_required=true` -- `docs/MODEL_CARD.md` and the
README model-card summary updated with a screening-note history line (not a
new roster entry; `n=3`, non-positive, fixture scale only).

## Recommendation for cycle 2

1. **model:** run the `component-plan` candidate next (driver's own rank-1
   priority) -- distinct untested lever, avoids repeating the just-exhausted
   `bounds` null.
2. **evaluation:** keep `e938_role_safe_all_targets_v2` (smoke, `n=3`) as the
   fixed matched suite for this lineage. Do **not** retarget
   `test_data_scaleup_v1` here -- that stays the separately tracked
   decode-compiler follow-up owned by the `continuous-openui-local` lineage /
   PR #1307.
3. **infrastructure:** no signal to escalate. `_feasible_eval_version`
   performed correctly on a fresh cycle; nothing new to repair.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local--c94ddb78-c1/`
- Runs: `.../runs/c20260802-continuous-openui-local--c94ddb78-c1-{control,bounds}/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- JSON twin: `continuous-openui-local-2-20260802-c1-results.json`
- Predecessor lineage (blocked, tracked separately): [`continuous-openui-local-20260802-c12-results.md`](continuous-openui-local-20260802-c12-results.md)
