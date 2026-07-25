# Autotrain loop ledger (fixture smoke)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.** Continuous loop under `MAX_RUN_MINUTES=3` / harness wall ≈2.58m.

| run_id | ok | steps | stopped_on | last_loss | wall_s | max_wall | scoreboard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `autotrain_wf_smoke_20260725` | True | 8 | steps | 41.52362060546875 | 55.16 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter2` | True | 8 | steps | 36.6005744934082 | 46.78 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter3` | True | 8 | steps | 24.184091567993164 | 44.7 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter4` | True | 8 | steps | 38.17085647583008 | 48.27 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter5` | True | 8 | steps | 44.06878662109375 | 47.84 | 2.5833333333333335 | True |

All runs: twotower / choice / scratch / cpu / fixture `wf_smoke_v1` / `--no-sync-checkpoints` / no `--ship-gates`.

Canonical CLI: `uv run --extra torch python -m scripts.slm` (`sft train` + `eval model --suites smoke --eval-limit 3`).
