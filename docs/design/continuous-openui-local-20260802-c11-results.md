# Continuous autotrain cycle 11 results (2026-08-02, loop `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c11` |
| Cycle intent | `retry_measurement` — driver-initiated frozen replay of cycle 10, **not a new hypothesis** |
| Upstream / integration | `b8188a49` / `b0550d1a` |
| Device | CPU |

## What happened

After acknowledging cycle 10's `document` action (`ack-action --action-index 1
--evidence docs/design/continuous-openui-local-20260802-c10-results.{md,json}`),
the driver was invoked once more via the same non-tracked eval-version
monkeypatch launcher used for cycle 10, intending to probe the locally-built
`smoke=6` reduced-scope snapshot (`outputs/data/eval/test_data_scaleup_v1_smoke6_probe/`).

Instead, the driver's own pending `retry_measurement` action from cycle 10's
handoff (`kind=retry_measurement, owner=autotrain, reason="measurement
incomplete; replay the identical frozen arm (0/2)"`) **took priority**: it
replayed the **identical frozen cycle-10 spec** against
`test_data_scaleup_v1` (`smoke=12`), resuming from cycle 10's saved
checkpoint (`--checkpoint .../c10-*/checkpoints/last.pt`) rather than
starting a fresh hypothesis. **The `smoke=6` probe was never exercised** —
frozen replay reuses the exact locked `eval_version` baked into the frozen
manifest and never calls `default_eval_version()`, so the monkeypatch had no
effect this round.

## Outcome

`measurement_incomplete` again. Both arms' `evaluate_model` stage again hit
the ~70s per-arm wall-time budget. Checkpoint resume saved train time and let
decode progress slightly further than cycle 10 — `processed_record_n=3` of
12 for both arms (vs. `2` of 12 in cycle 10) — but nowhere close to
completing the `n=12` smoke suite within one cycle.

`SDLC Phase A`: `positive=false`, `stack_layer=false`,
`action=no_stack_layer_non_positive`. Reasons: `measurement_incomplete` (both
arms, `missing_scoreboard`), `wall_timeout` + `empty_metrics` (both arms),
`primary_metric_unavailable`.

The driver's handoff again queues a second automatic `retry_measurement`
(`reason="measurement incomplete; replay the identical frozen arm (1/2)"`).

## Session stopping decision

**This session stops here — it does not continue to cycle 12.** Two
consecutive frozen-replay attempts at `smoke=12` (cycles 10 and 11) both hit
the same structural wall-time ceiling, with only marginal progress from
checkpoint reuse (`2/12 → 3/12` records decoded per arm). This confirms the
finding is not a one-off fluke: `n=12` genuinely does not fit the harness's
current per-arm wall budget on CPU, even across repeated attempts with warm
checkpoints. Extrapolating the `2→3` record trend, a full `n=12` decode pass
for one arm alone would need several more retry cycles at this rate —
disproportionate to what a single continuous-loop session should spend
chasing an already-established conclusion.

Also: the driver's own `retry_measurement` action (not a new hypothesis)
does not exercise the locally-built `smoke=6` probe at all, so continuing to
invoke the driver without further intervention would not even test the
reduced-scope idea — that requires either clearing/expiring the
`retry_measurement` action or a future cycle explicitly starting a **fresh**
(non-frozen) campaign, both out of this cycle's scope to force.

## Recommendation for cycle 12+

1. **infrastructure (highest leverage):** either (a) let the queued
   `retry_measurement` continue in a future session with a **raised**
   per-arm wall budget for eval-scale screens (the underlying fix), or
   (b) explicitly start a fresh, non-frozen campaign against the pre-built
   `outputs/data/eval/test_data_scaleup_v1_smoke6_probe` (`smoke=6`) to
   finally test the reduced-scope hypothesis this cycle could not reach.
2. **infrastructure:** fix `test_continuous_classify_positive_entry` (see
   cycle 10 doc) so `_DEFAULT_EVAL_VERSION_CANDIDATES` can be durably
   updated in `engine.py` via a normal commit, removing the need for a
   per-session monkeypatch launcher.
3. **model:** `component-structure` is still untested at any suite scale
   with a completed measurement — not exhausted, just never measured.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c11/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c11-{control,component-structure}/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- JSON twin: `continuous-openui-local-20260802-c11-results.json`
- Predecessor: [cycle 10 results](continuous-openui-local-20260802-c10-results.md) (first `test_data_scaleup_v1` attempt, `2/12` records)
