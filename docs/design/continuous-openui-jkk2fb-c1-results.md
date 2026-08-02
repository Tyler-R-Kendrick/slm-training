# Continuous autotrain cycle 1 results (2026-08-02, loop `continuous-openui-jkk2fb`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-jkk2fb` |
| Campaign | `continuous-loop-20260802-continuous-openui-jkk2fb-9f2e5830-c1` |
| Source (upstream = integration) | `62f31556` |
| Device | CPU |
| Steps | 20 / batch default / seed default |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke`, n=3 |
| Wall cap | 3 minutes (actual cycle wall: ~18s) |
| Environment | fresh `.venv` (Python 3.12, `torch==2.5.1+cu124`, `pip install -e ".[torch]"`) — not pre-installed in this session's checkout |

## Run matrix

| Arm | Lever | smoke n | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | `grammar_completion_bounds=False` | 3 | 1.0 | 0.0 | 0.0575 | 0.6333 | 2469.22 | trained; eval smoke metrics captured, ship-gate publish step failed |
| bounds (candidate) | `grammar_completion_bounds=True` | 3 | 1.0 | 0.0 | 0.0575 | 0.6333 | 2380.30 | trained; eval smoke metrics captured, ship-gate publish step failed |

Primary metric (`smoke.structural_similarity`, direction=increase): control 0.0575, candidate 0.0575,
**improvement = 0.0**. `meaningful_program_rate` is 0.0 for both arms (flat, not just latency). The
only arm-to-arm difference is latency (candidate 88.92 ms faster p50), which alone is **not** a
positive result under the quality-aware tradeoff classifier — `parse_held`/`mpr_held` hold, but there
is no quality improvement to spend the latency budget on, and no latency-primary win claim is being
made here (primary metric for this cycle's classification was `smoke.structural_similarity`, correctly
flat at 0.0).

## Diagnostics

1. **Both arms' eval runs failed the honest ship-gate publish step**: `RuntimeError: AgentV SDK is
   unavailable; run npm ci in the checkout or set AGENTV_RUNNER` from
   `src/slm_training/evals/agentv.py:32`. This is a known environment gap noted by prior scheduled
   sessions (see `docs/design/autotrain-loop-ledger-20260725.md` tail: "plus `npm ci` for the AgentV
   publish step — neither was pre-installed in this session's checkout"). `npm`/`node` are present
   (`/opt/node22/bin/{npm,node}`, `package.json` declares `@agentv/core`, `agentv`, `@playwright/mcp`,
   `@playwright/test`) but `npm ci` was not run this session to keep the iteration inside the wall/time
   budget — the smoke-suite metrics (`eval_smoke.json`, n=3, `parse_rate`/`structural_similarity`/
   `binder_reference_f1`/`meaningful_program_rate`) were computed and captured **before** that publish
   step, so this cycle's classification is on real, honest per-arm numbers even though the full
   multi-suite ship scoreboard did not publish. Ship-gate status stays **incomplete**, not passed.
2. `grammar_completion_bounds=True` did not move `structural_similarity` or `meaningful_program_rate`
   at all on this fixture/step budget — identical to 4 decimal places on both metrics. Only latency
   moved, and the classifier correctly rejected that as non-positive (SDLC Phase A reason:
   `primary_metric_null_or_worse:smoke.structural_similarity:control=0.0575 candidate=0.0575
   improvement=0.0`).
3. `sdlc_delivery.json` confirms `measurement_complete: true`, `fixture_volume_gate_hits: 0`,
   `positive: false`, `stack_action: "no_stack_layer_non_positive"`.

## Classification (SDLC Phase A)

**Non-positive.** Primary metric (`smoke.structural_similarity`) is flat at 0.0 improvement; the only
observed delta is a latency blip (2469.22 ms → 2380.30 ms) with empty quality meaning
(`meaningful_program_rate` 0.0 on both arms) — explicitly excluded by
`scripts/run_autotrain_continuous._classify_metric_tradeoff` and the `autotrain-iteration-delivery`
"not positive" list ("Fixture `insufficient_n` / expected ship-gate fails on smoke-scale data").
No stacked PR opened for this cycle; docs + local commit only.

## Next-run priorities (from the driver's ranked list)

1. **model** (rank 1, confidence 0.90): bounds arm is exhausted at this fixture scale — test the
   size-matched `component-plan` quality hypothesis next
   (`c20260802-continuous-openui-jkk2fb-9f2e5830-c1-component-plan`).
2. **evaluation** (rank 2, confidence 0.70): keep the matched control as the size-matched baseline
   every cycle.
3. **model** (rank 3, confidence 0.65): rotate the thrash recommendation across the lever bank
   (not bounds-only) — the completed candidate cannot be reselected without a new hypothesis.
4. **infrastructure** (rank 4, confidence 0.80): soft ship-gate fails on fixture n never stop the loop.
5. **model_build** (rank 5, confidence 0.55, speculative): confirmed champions promote under cadence;
   thrash only screens.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-jkk2fb-9f2e5830-c1/`
  (gitignored — not committed)
- Runs: `.../runs/c20260802-continuous-openui-jkk2fb-9f2e5830-c1-control/`,
  `.../runs/c20260802-continuous-openui-jkk2fb-9f2e5830-c1-bounds/`
- Handoff: `.../cycle_handoff.json` (actions: `document`, `next_experiment`)
- JSON twin: `continuous-openui-jkk2fb-c1-results.json`
