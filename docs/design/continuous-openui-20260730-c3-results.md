# Continuous autotrain cycle 3 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260801-c1` |
| Source | `c1c4eca349b66f05684975575a3640ced50051ea` |
| Device | CPU |
| Steps | 21 / batch 2 / seed 100001 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c20260801-c1-control | bounds off | 3 | 1.0 | 0.0 | 2961.24 | eval completed; ship gates fail (insufficient n + quality) |
| c20260801-c1-bounds | bounds **on** | 3 | 1.0 | 0.0 | 2955.83 | eval completed; ship gates fail (same) |

Primary delta (bounds − control) p50 latency: **-5.41 ms** (bounds slightly faster). SDLC Phase A classifies this **non-positive**: `latency_win_rejected_low_mpr` (meaningful_program_rate=0.0 on both arms, mpr floor is 1/3). A latency win with zero held quality signal is not a real win.

## Diagnostics

1. Two environment gaps blocked earlier attempts at this cycle in this fresh container, both resolved by running setup steps README already documents (`pip install -e ".[dev,torch,web]"`, `npm ci`) rather than any harness bug:
   - Fresh `uv venv` install had no `torch` (`train_model.detect_device` import failed) — fixed by installing the `dev` extra.
   - Fresh checkout had no AgentV SDK (`evaluate_model`'s `publish_agentv_evaluation` raised `AgentV SDK is unavailable; run npm ci`) — fixed by running `npm ci` (with `NODE_OPTIONS` unset to work around a malformed value already present in this shell's environment).
2. After both were repaired, the identical cycle spec ran clean end-to-end: train + eval on both arms, honest ship-gate fail on fixture `n=3` (`need>=20`) plus missing `held_out`/`adversarial`/`ood`/`rico_held` suites — expected for a `wf_smoke_v2`/21-step screening cycle, not a regression.
3. `grammar_completion_bounds=True` shows a small p50 latency improvement (-5.41ms) but `meaningful_program_rate` is 0.0 on both arms, so the delta carries no quality signal and the existing classifier correctly rejects it as positive.

## Next-run priorities

1. **model:** re-test `grammar_completion_bounds` vs control once a cycle produces `meaningful_program_rate > 0`, so the mpr floor has real signal to gate on.
2. **evaluation:** keep ship gates honest; do not weaken for continuous smoke.
3. Do not promote RL; ship gates fail by design on fixture n.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c1/` (not tracked — `outputs/` is gitignored)
- Runs: `.../runs/c20260801-c1-control/`, `.../runs/c20260801-c1-bounds/`
- JSON twin: `continuous-openui-20260730-c3-results.json`
