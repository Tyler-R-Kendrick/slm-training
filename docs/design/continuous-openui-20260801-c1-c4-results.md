# Continuous autotrain cycles 1-4 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-scheduled` |
| Campaigns | `continuous-loop-20260801-c1` .. `c4` |
| Source | `2a64db996754d2ce25edcc0d82ee448f7843cd85` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Wall cap | 3 minutes per run |

Driven by `python -m scripts.run_autotrain_continuous --loop-id continuous-openui-scheduled --max-cycles N --train-version wf_smoke_v2 --steps 20`, run in two batches (c1-c3, then c4 after the self-heal below).

## Run matrix

| Campaign | Role | Primary metric | Control | Candidate | parse_rate | mpr | Delta | Positive |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c1 (bounds) | screening | smoke.latency_ms_p50 | 3159.94 ms | 3346.18 ms | 1.0 → 1.0 | 0.0 → 0.0 | +186.24 ms (slower) | **No** |
| c2 (canvas) | screening | smoke.latency_ms_p50 | 22072.23 ms | 23405.01 ms | 1.0 → 1.0 | 0.0 → 0.0 | +1332.78 ms (slower) | **No** |
| c3 (bounds+canvas) | screening | smoke.latency_ms_p50 | 7825.55 ms | 7987.23 ms | 1.0 → 1.0 | 0.0 → 0.0 | +161.68 ms (slower) | **No** |
| c4 (steps) | promotion | held_out.structural_similarity (unavailable, fixture n) | 21636.11 ms | 11930.55 ms | 1.0 → 1.0 | 0.333 → 0.333 | mpr_per_ms 1.54e-05 → 2.79e-05 | **Yes**, but `fixture_insufficient_n` + no tracked delta → `positive_no_tracked_delta_skip_stack` |

None of the three lever screens (grammar_completion_bounds, compact_active_canvas, both) reduced smoke p50 latency; all three candidates were slower than their matched control. `meaningful_program_rate` was 0.0 across c1-c3 (20 steps is too few to produce a meaningful program on this fixture) so these are pure latency screens, not quality wins.

## Environment self-heal (no tracked source changed)

c1-c3's held-out/AgentV-backed eval arms raised:

```
RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER
```

from `src/slm_training/evals/agentv.py::_agentv_runtime` — `node_modules/@agentv/core` was
missing in this checkout, contributing to the `empty_metrics:<hash>` reasons on all three
cycles. Fix: `npm ci` (package.json already pins `@agentv/core`; nothing tracked changed,
`node_modules/` is gitignored). After the fix, c4 ran with AgentV available:
`meaningful_program_rate` moved from 0.0 (c1-c3) to 0.333 (c4 both arms) and
`mpr_per_ms` improved 1.54e-05 → 2.79e-05. This is **not** a controlled A/B of the fix —
c4 used a different knob (`steps`) and a different random draw than c1-c3 — so treat it as
"the eval path now works end-to-end," not as a measured effect of installing the SDK.

## Diagnostics

1. `grammar_completion_bounds`, `compact_active_canvas`, and their combination did not
   improve smoke decode p50 latency under this recipe (c1-c3); all three were screening-role
   cycles and non-positive per `sdlc autotrain-iteration-delivery`.
2. c4 is a fixture-scale efficiency win on `mpr_per_ms` but both arms hit
   `fixture_insufficient_n`, and there is no tracked code/docs delta from this cycle to stack
   — `run_autotrain_continuous.py` correctly classified it `positive_no_tracked_delta_skip_stack`
   (positive result recorded, but nothing to open a stacked PR for).
3. No cycle here clears ship gates; all remain fixture/scratch honesty.

## Next-run priorities

1. Re-test `grammar_completion_bounds` / `compact_active_canvas` at a higher step budget
   within the wall cap now that AgentV eval runs end-to-end.
2. Do not promote from c1-c4; every arm is `fixture_insufficient_n` or non-positive.
3. Consider a continuous-loop preflight that fails closed with a clear message when
   `node_modules/@agentv/core` is missing, instead of surfacing as opaque `empty_metrics`
   hashes in the delivery ledger.

## Artifacts

- Campaigns: `outputs/autoresearch/continuous-loop-20260801-c1/` .. `c4/`
- Delivery ledger: `outputs/autoresearch/sdlc_delivery_ledger.jsonl` (last 4 entries)
- JSON twin: `continuous-openui-20260801-c1-c4-results.json`
