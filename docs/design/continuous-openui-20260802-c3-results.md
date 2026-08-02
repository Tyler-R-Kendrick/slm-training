# Continuous autotrain cycle 3 results (2026-08-02)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c3` |
| Source | `1a92671c` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Params | smoke n | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency_ms_p50 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c3-control | 1,766,987 | 3 | 1.0 | 0.3333 | 0.2308 | 0.7333 | 4,267.38 | eval completed; ship gates fail (insufficient n) |
| c3-component-edge | 1,766,987 | 3 | 1.0 | 0.3333 | 0.2308 | 0.7333 | 3,818.46 | eval completed; ship gates fail (insufficient n) |

Every quality metric is **exactly identical** between arms; the only
delta is p50 latency, down 4,267.38 → 3,818.46 ms (**-10.5% wall**,
**+11.76%** in `mpr_per_ms` throughput). SDLC Phase A's efficiency-win
rule (quality held, latency gain ≥ 5% floor) classified this cycle
**POSITIVE**.

## Diagnostics

1. This is a real efficiency signal, not a quality-tradeoff win: every
   smoke metric ties exactly, so the latency delta cannot be explained by
   the candidate producing worse/simpler output.
2. The driver's own `sdlc_delivery.json` recorded
   `stack_action=positive_no_tracked_delta_skip_stack` — that reflects the
   working tree being clean at the moment of classification (before this
   documentation existed), not a judgment against stacking. Per
   `continuous.md`'s "documentation creates a reviewable delta" rule, this
   doc commit is the tracked delta and the cycle is eligible for a stacked
   layer.
3. Fixture `n=3` still fails the `insufficient_n` ship-gate floor
   (`need>=20`); this remains a screening signal only, not a ship claim.

## Next-run priorities

1. **model:** confirm the component-edge latency win on a new seed / larger
   step budget before treating it as a durable recipe default.
2. **evaluation:** keep ship gates honest — this is a fixture-scale
   efficiency screening signal, not a promotion or ship claim.
3. **model:** rotate the next screening arm to a distinct lever per the
   hypothesizer feedback queue rather than repeating this arm.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c3/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c3-control/`,
  `.../runs/c20260802-continuous-openui-local-8c0b60dd-c3-component-edge/`
- JSON twin: `continuous-openui-20260802-c3-results.json`
