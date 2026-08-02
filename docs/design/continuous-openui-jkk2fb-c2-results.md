# Continuous autotrain cycle 2 results (2026-08-02, loop `continuous-openui-jkk2fb`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-jkk2fb` |
| Campaign | `continuous-loop-20260802-continuous-openui-jkk2fb-9f2e5830-c2` |
| Upstream | `62f31556` |
| Integration | `b2068072` (this loop's own c1 docs commit) |
| Device | CPU |
| Steps | 20 (declared) / control ran 22 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke` |
| Wall cap | 3 minutes total, ~1.17 min per arm slice |
| Candidate hypothesis | `component-plan` head prebuilt (driver's rank-1 successor priority from cycle 1: bounds arm exhausted, test the next distinct size-matched quality hypothesis) |

## Run matrix

| Arm | Lever | Status | steps | smoke metrics |
| --- | --- | --- | --- | --- |
| control | baseline (no component-plan head) | training completed (`stopped_on=steps`, 22 steps, last_loss 14.39) but **hit the per-arm wall_timeout before `evaluate_model` ran** — no suite metrics produced | 22 | — |
| component-plan (candidate) | component-plan head prebuilt, 1,755,764 trainable params (size-matched to control) | trained; eval smoke metrics captured; ship-gate publish step failed | 20 (declared) | parse_rate 1.0, meaningful_program_rate 0.0, structural_similarity 0.0964, binder_reference_f1 0.0, latency_ms_p50 2088.0 |

`sdlc_delivery.json` reasons: `wall_timeout:223de081...`, `primary_metric_unavailable`.
`measurement_complete: false` — this cycle is an **incomplete measurement**, not a quality result: the
control arm's training finished normally but its evaluation was cut off by the wall cap before it
could publish any suite metrics, so there is no valid control-vs-candidate primary-metric comparison
this cycle (candidate-only numbers exist but cannot be compared against a matched control).

## Diagnostics

1. This is the second consecutive cycle where the ship-gate publish step failed
   (`AgentV SDK unavailable; run npm ci in the checkout or set AGENTV_RUNNER`) on the arm(s) that did
   reach evaluation — same known environment gap as cycle 1
   ([continuous-openui-jkk2fb-c1-results.md](continuous-openui-jkk2fb-c1-results.md)).
2. New signal this cycle: the **control arm's evaluation never started** — training alone (22 steps)
   consumed most of the ~1.17-minute per-arm wall slice on this CPU container, leaving no time for
   `evaluate_model` before the harness's own wall-cap guardrail stopped the arm (`status: stopped`,
   `exit_code: null` in `results.tsv`). No typed `HarnessSignalV1` was raised by the driver
   (`harness_signals: []` in the JSON output) — per `continuous.md` this is a **soft failure**
   ("Fixture `insufficient_n` / expected ship-gate fails on smoke-scale data ... timeouts without a
   win" are explicitly non-positive and never loop-terminating), not a reproduced canonical-harness
   defect requiring a repair lane. Per the "hard run cap" contract: *"A timed-out, interrupted, or
   killed run is never evidence."*
3. `fixture_volume_gate_hits: 0`.

## Classification (SDLC Phase A)

**Non-positive.** `measurement_complete: false` and `primary_metric_unavailable` — a wall-capped,
incomplete measurement is explicitly excluded from "positive" under
`autotrain-iteration-delivery.md` ("Fixture `insufficient_n` ... null lever deltas, wall timeouts with
no metric win" → not positive). No stacked PR opened; docs + local commit only.

## Next-run priorities (from the driver's ranked list)

1. **model** (rank 1, confidence 0.90): component-plan arm is exhausted at this scale — test the
   distinct size-matched `component-edge` hypothesis next
   (`c20260802-continuous-openui-jkk2fb-9f2e5830-c2-component-edge`).
2. **evaluation** (rank 2, confidence 0.70): keep the matched control as the size-matched baseline
   every cycle.
3. **model** (rank 3, confidence 0.65): rotate the thrash recommendation across the lever bank.
4. **infrastructure** (rank 4, confidence 0.80): soft ship-gate fails on fixture n never stop the loop.
5. **model_build** (rank 5, confidence 0.55, speculative): confirmed champions promote under cadence;
   thrash only screens.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-jkk2fb-9f2e5830-c2/`
  (gitignored — not committed)
- Runs: `.../runs/c20260802-continuous-openui-jkk2fb-9f2e5830-c2-control/`,
  `.../runs/c20260802-continuous-openui-jkk2fb-9f2e5830-c2-component-plan/`
- Handoff: `.../cycle_handoff.json` (actions: `document`, `next_experiment`)
- JSON twin: `continuous-openui-jkk2fb-c2-results.json`
