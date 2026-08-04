# Continuous autotrain cycle 2 results (2026-07-30)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260730-c2` |
| Source | `c3df22b34e213cd59560e1f7d99efbf0cbaa12e3` |
| Device | CPU |
| Steps | 20 / batch 2 / seed 7 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` (after v1 path failure) |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c2-control | bounds off | — | — | — | — | **failed** missing suite smoke under `v1` |
| c2r-control-e938 | bounds off, e938 | 3 | 1.0 | 0.0 | 1962.86 | eval completed; ship gates fail (insufficient n + quality) |
| c2r-bounds-e938 | bounds **on**, e938 | 3 | 1.0 | 0.0 | 2017.26 | eval completed; ship gates fail (same) |

Primary delta (bounds − control) p50 latency: **54.40 ms** (positive = bounds slower).

## Diagnostics

1. Continuous `compile_commands` defaults `--test-dir v1` when `eval_version` is unset; published fixtures live under names like `e938_role_safe_all_targets_v2`. First arm failed closed on path, not model quality.
2. After binding `eval_version`, both arms ran full multi-suite ship-gate eval. Fixture 20-step models correctly fail quality/volume gates; that is not promotion evidence.
3. `grammar_completion_bounds=True` did **not** improve smoke decode p50 under this recipe.

## Next-run priorities

1. **infrastructure / model_build:** fail closed or default continuous eval to a real published suite (reproduced on frozen path error).
2. **model:** re-run size-matched bounds/canvas only after (1), possibly higher steps within wall.
3. **evaluation:** keep ship gates honest; do not weaken for continuous smoke.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260730-c2/`
- Runs: `.../runs/c2r-control-e938/`, `.../runs/c2r-bounds-e938/`
- JSON twin: `continuous-openui-20260730-c2-results.json`
