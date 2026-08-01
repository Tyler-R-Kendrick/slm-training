# Continuous autotrain cycle 2 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260801-c2` |
| Source | `24c20769c366aeb9e9f7a98eb72089b3a97859c7` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c20260801-c2-control | canvas off | 3 | 1.0 | 0.0 | 19905.00 | eval completed; ship gates fail (insufficient n + quality) |
| c20260801-c2-canvas | canvas **on** | 3 | 1.0 | 0.0 | 22230.22 | eval completed; ship gates fail (same) |

Primary delta (canvas − control) p50 latency: **+2325.22 ms** (positive = canvas slower).

## Diagnostics

1. AgentV ship-gate publication succeeded this cycle after `npm ci`
   (`NODE_OPTIONS=--max-old-space-size=8192` override) resolved the
   `AgentV SDK is unavailable` failure observed in cycle 1.
2. `compact_active_canvas=True` did **not** improve smoke decode p50 under
   this recipe; both arms still fail `smoke:insufficient_n actual=3 need>=20`
   and every quality-threshold gate (`meaningful_program_rate`,
   `structural_similarity`, `component_type_recall`, `ast_beq_rate`,
   `canonical_beq_rate`, `placeholder_fidelity`, `reward_score`).
3. `held_out` / `adversarial` / `ood` / `rico_held` suites are absent from
   this eval snapshot (`missing_suite`) — smoke-only, wiring evidence.

## Next-run priorities

1. **model:** re-test `compact_active_canvas` only after a higher-step/higher-n
   budget clears fixture `insufficient_n`.
2. **evaluation:** keep ship gates honest; do not weaken for continuous smoke.
3. **infrastructure:** the ambient `NODE_OPTIONS` in this container ships a
   malformed quoted `--import tsx` value that node's CLI parser rejects
   verbatim (`node: --import tsx is not allowed in NODE_OPTIONS`); override to
   `--max-old-space-size=8192` before invoking npm/node in the loop worktree.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c2/`
- JSON twin: `continuous-openui-20260801-c2-results.json`
