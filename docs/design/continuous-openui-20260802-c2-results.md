# Continuous autotrain cycle 2 results (2026-08-02)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c2` |
| Source | `895f0b729c5a5db1c9a2ed4a0f56d8f5b4b3e5a1` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Params | smoke n | parse_rate | structural_similarity | component_type_recall | latency_ms_p50 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c2-control | 1,755,764 | 3 | 1.0 | 0.3267 | 0.1667 | 12,662.88 | eval completed; ship gates fail (insufficient n + quality) |
| c2-component-plan | 1,755,764 | 3 | 1.0 | 0.0964 | 0.0833 | 1,406.41 | eval completed; ship gates fail (same) |

Primary delta (component-plan − control) `smoke.structural_similarity`:
**-0.2303** (-70.5% relative). The size-matched candidate is a real quality
regression, not a null tie — SDLC Phase A classified the cycle
`NON_POSITIVE` correctly.

## Diagnostics

1. This cycle's AgentV eval ran to completion for the first time in this
   loop (the AgentV SDK bootstrap from cycle 1 held), so these numbers are a
   real measurement, not the crash-fallback constant seen in cycle 1.
2. The component-plan head both trains lower quality *and* runs the smoke
   decode ~9x faster p50 (1.41s vs 12.66s) than the size-matched control.
   That latency gap is large enough to warrant its own investigation before
   being read as a real levers effect — it may be a warm/cold decode-path
   artifact rather than something the component-plan head itself causes.
3. Fixture `n=3` is far below the ship `insufficient_n` floor (`need>=20`),
   so both arms fail ship gates by construction; this is expected at
   fixture scale and not a loop terminator.

## Next-run priorities

1. **model:** reject the component-plan head at this step count/seed; do
   not carry it forward as a candidate.
2. **model:** run the distinct size-matched `component-edge` quality
   hypothesis next (queued by the hypothesizer).
3. **evaluation:** keep the matched control as the baseline every cycle.
4. **infrastructure:** the ~9x control-vs-candidate p50 latency gap on an
   otherwise size-matched pair deserves a dedicated latency screen before
   being cited as a levers effect.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c2/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c2-control/`,
  `.../runs/c20260802-continuous-openui-local-8c0b60dd-c2-component-plan/`
- JSON twin: `continuous-openui-20260802-c2-results.json`
