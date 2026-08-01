# Continuous autotrain cycle — 2026-08-01, campaign `continuous-loop-20260801-c2`

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260801-c2` (cycle 2, predecessor `continuous-loop-20260801-c1`) |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |
| Steps | 20 / batch 2 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c20260801-c2-control | canvas off | 3 | 1.0 | 0.0 | 21586.91 | eval completed; ship gates fail (insufficient n + quality) |
| c20260801-c2-canvas | canvas **on** | 3 | 1.0 | 0.0 | 21350.87 | eval completed; ship gates fail (same) |

Primary delta (canvas − control) p50 latency: **-236.04 ms** (canvas faster).

## SDLC Phase A classification

- `positive: false`, `stack_layer: false` — **no stacked PR opened this cycle.**
- Reasons: `fixture_insufficient_n` on both arms (n=3, need >= 20), and
  `latency_win_rejected_low_mpr` (meaningful_program_rate 0.0 < 0.333 quality
  floor) — the 236ms latency win is not counted as positive per the
  quality-aware tradeoff policy (a pure latency win with zero meaningful-
  program rate is explicitly excluded).

## Diagnostics

1. Unlike the 2026-07-30 c2 cycle, `eval_version` resolved correctly to
   `e938_role_safe_all_targets_v2` on the first try — the previously
   documented default-`v1`-suite footgun did not recur (harness fixes in
   #1242–#1245 plus explicit `--primary-metric`/`--objective` wiring cover it).
2. `compact_active_canvas=True` shows a directionally faster smoke p50
   (21350.87ms vs 21586.91ms) but `meaningful_program_rate` is 0.0 for both
   arms at this 20-step/batch-2 fixture size — too small to say anything
   about program quality, so the latency delta stays a screening signal only.
3. Ship gates correctly fail closed on `insufficient_n` (fixture n=3 vs
   required n>=20) and the `held_out`/`adversarial`/`ood`/`rico_held` suites
   report `missing_suite` at this train_version/eval_version pairing — none
   of this is promotion evidence.

## Next-run priorities

1. **model:** re-run `compact_active_canvas` at a step/data budget large
   enough to raise `meaningful_program_rate` above 0 and suite `n` above the
   fixture floor before trusting the latency delta.
2. **infrastructure:** none outstanding for this path this cycle (see the
   companion `continuous-openui-20260801-c1-results.md` for the separate
   AgentV SDK / `npm ci` setup gap hit earlier in this session).
3. **evaluation:** keep ship gates fail-closed; do not promote off n=3
   fixture evidence.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c2/`
- Runs: `.../runs/c20260801-c2-control/`, `.../runs/c20260801-c2-canvas/`
- JSON twin: `continuous-openui-20260801-c2-results.json`
- SDLC Phase A ledger: `outputs/autoresearch/continuous-loop-20260801-c2/sdlc_delivery.json`
