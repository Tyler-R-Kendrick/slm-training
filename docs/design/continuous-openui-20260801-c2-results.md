# Continuous autotrain cycle 2 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c2` |
| Predecessor | `continuous-loop-20260801-c1` |
| Source | `2a64db996754d2ce25edcc0d82ee448f7843cd85` |
| Device | CPU |
| Steps | 42 / batch 2 / seed 100002 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c20260801-c2-control | bounds off, canvas off | 3 | 1.0 | **1.0** | 17947.37 | eval completed; ship gates fail (insufficient n only) |
| c20260801-c2-canvas | bounds off, canvas **on** | 3 | 1.0 | **1.0** | 19051.98 | eval completed; ship gates fail (same) |

Primary delta (canvas − control) p50 latency: **+1104.61 ms** (positive = canvas slower).
`c20260801-c2-bounds`, `-both`, `-steps`, `-batch1` were proposed by the
hypothesizer but not executed this cycle (budget: control + 1 candidate;
`canvas` was picked over `bounds`).

## SDLC Phase A classification

**Non-positive.** `sdlc_delivery.json` → `positive=false`, `stack_layer=false`.
`compact_active_canvas` is a clean latency regression at held quality
(`mpr` 1.0→1.0, `parse` 1.0→1.0) — not a null delta, an actual loss. Fixture
`insufficient_n` (n=3) still blocks any ship-quality claim independent of
direction.

No stacked PR opened. Local commit only.

## Diagnostics (headline finding)

Doubling `steps` from 21 (cycle 1) to 42 (this cycle) moved
`meaningful_program_rate` from **0.0 → 1.0** on the unmodified control arm.
This confirms the cycle-1 reading: the -262.65ms "win" for
`grammar_completion_bounds` at 21 steps was a quality-blind blip on garbage
output, correctly rejected by the `mpr >= 1/3` classifier floor. At 42 steps
the model produces structurally meaningful programs on every smoke case, and
decode latency scales up accordingly (17.9s vs 3.3s p50) because more of the
grammar is actually being walked instead of terminating early on
degenerate output.

`compact_active_canvas` was *not* tested at 21 steps in cycle 1 (bounds was
the cycle-1 candidate); at 42 steps it is now measured and is a clear
regression, not a wash.

## Next-run priorities

1. **model:** run `c20260801-c2-bounds` (`grammar_completion_bounds=true`,
   `steps=42`) against this cycle's control to check whether the bounds lever
   still helps latency once `meaningful_program_rate` is held at 1.0 (the
   comparison cycle 1 could not make honestly).
2. **model:** drop `compact_active_canvas` as a latency candidate at this
   recipe — no quality upside, measured latency cost.
3. **evaluation:** fixture `n=3` will keep failing the volume gate regardless
   of lever choice; that is expected and not a blocker.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c2/` (local, not tracked)
- Runs: `.../runs/c20260801-c2-control/`, `.../runs/c20260801-c2-canvas/`
- JSON twin: `continuous-openui-20260801-c2-results.json`
- SDLC delivery ledger: `outputs/autoresearch/continuous-loop-20260801-c2/sdlc_delivery.json`
- Predecessor: `continuous-openui-20260801-c1-results.md`
