# Continuous autotrain cycle 2 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c2` |
| Source | `c1c4eca349b66f05684975575a3640ced50051ea` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |
| Hypothesis | `compact_active_canvas` reduces smoke `latency_ms_p50` vs. matched control without lowering `parse_rate` |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c2-control | canvas off | 3 | 1.0 | 0.0 | 20845.04 | eval completed; ship gates fail (insufficient n + quality) |
| c2-canvas | `compact_active_canvas=true` | 3 | 1.0 | 0.0 | 21207.37 | eval completed; ship gates fail (same) |

Primary delta (canvas − control) p50 latency: **+362.33 ms** (candidate slower;
null/negative lever delta).

## Diagnostics

1. AgentV SDK is installed and working this cycle (see
   `continuous-openui-20260801-c1-results.md` for the self-heal); both arms
   produced full `@agentv/core` scoreboards and ship-gate outcomes instead of
   crashing.
2. Control fails ship gates purely on fixture scale: `insufficient_n`
   (n=3 < 20) plus every quality threshold that depends on volume. This is
   the expected outcome for a 20-step `wf_smoke_v2` fixture run, not a model
   regression.
3. `compact_active_canvas` did **not** reduce smoke p50 latency under this
   size-matched 20-step recipe (delta +362.33 ms, i.e. slower); `parse_rate`
   and `meaningful_program_rate` were unchanged (1.0→1.0, 0.0→0.0).
4. SDLC Phase A classification: **non-positive** — `fixture_insufficient_n`
   on both arms plus `primary_metric_null_or_worse` for the candidate. No
   stack layer opened (`stack_layer=false`,
   `outputs/autoresearch/sdlc_delivery_ledger.jsonl`).

## Next-run priorities

1. **model:** do not promote `compact_active_canvas` from this screening
   result; the delta is negative, not just insufficient-n.
2. **model:** re-screen other continuous hypothesis-matrix candidates
   (`bounds`, `steps`, `batch1`) from this loop before spending more cycles
   on canvas.
3. **evaluation:** keep ship gates honest; fixture `insufficient_n` stays an
   expected diagnostic, never a loop terminator.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c2/`
- Runs: `.../runs/c20260801-c2-control/`, `.../runs/c20260801-c2-canvas/`
- JSON twin: `continuous-openui-20260801-c2-results.json`
