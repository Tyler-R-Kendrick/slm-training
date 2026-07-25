# Autotrain loop ledger (fixture smoke)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.** Continuous loop under MAX_HARNESS_WALL_MINUTES.

| run_id | ok | steps | stopped_on | last_loss | wall_s | max_wall | scoreboard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `autotrain_wf_smoke_20260725` | True | 8 | steps | 41.52362060546875 | 55.16 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter2` | True | 8 | steps | 36.6005744934082 | 46.78 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter3` | True | 8 | steps | 24.184091567993164 | 44.7 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter4` | True | 8 | steps | 38.17085647583008 | 48.27 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter5` | True | 8 | steps | 44.06878662109375 | 47.84 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter6` | True | 8 | steps | 29.073862075805664 | 42.78 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter7` | True | 8 | steps | 26.87131690979004 | 43.78 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter8` | True | 8 | steps | 34.350059509277344 | 46.46 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter9` | True | 8 | steps | 39.343326568603516 | 57.68 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter10` | True | 8 | steps | 38.7589111328125 | 48.11 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter11` | True | 8 | steps | 30.61425018310547 | 45.57 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter12` | True | 8 | steps | 44.137203216552734 | 35.22 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter13` | True | 8 | steps | 34.19657516479492 | 46.4 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter14` | True | 8 | steps | 26.997821807861328 | 40.42 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter15` | True | 8 | steps | 37.53594207763672 | 42.18 | 2.5833333333333335 | True |
| `autotrain_wf_smoke_20260725_iter16` | True | 8 | steps | 33.343055725097656 | 40.55 | 2.5833333333333335 | True |

Canonical CLI: `uv run --extra torch python -m scripts.slm` smoke recipe. No ship-gates.
