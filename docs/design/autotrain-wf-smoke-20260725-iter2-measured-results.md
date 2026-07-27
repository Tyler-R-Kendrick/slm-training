# Autotrain workflow smoke — measured results (autotrain_wf_smoke_20260725_iter2)

**Honesty:** `fixture_or_scratch` wiring only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| recipe | smoke (loop iter2) |
| run_id | `autotrain_wf_smoke_20260725_iter2` |
| parent_run | `autotrain_wf_smoke_20260725` |
| train | fixture `wf_smoke_v1` (103 records, strict) |
| test | fixture `wf_smoke_v1` smoke suite |
| model | twotower / choice / scratch / cpu |
| steps | 8 (`--fast-train`, `--no-sync-checkpoints`) |
| seed | 1 |
| ship_gates | false (`--eval-limit 3 --suites smoke`) |
| train last_loss | 36.6005744934082 |
| stopped_on | steps |
| elapsed_wall_seconds | 46.778278316007345 |
| max_wall_minutes | 2.5833333333333335 |
| smoke n | 3 (diagnostic subset) |
| meaningful_program_rate | 0.0 |
| decode_timeout_count | 3 |
| AgentV criteria.pass | None |

## Phase status

| Phase | Status |
| --- | --- |
| data_train | skipped (artifacts present) |
| data_test | skipped (artifacts present) |
| quality | ok |
| sft | ok |
| eval | ok (wiring; criteria fail expected) |
| closeout | ok |
| verify | ok |

## Artifacts

- `outputs/data/train/wf_smoke_v1`
- `outputs/data/eval/wf_smoke_v1`
- `outputs/runs/autotrain_wf_smoke_20260725_iter2/train_summary.json`
- `outputs/runs/autotrain_wf_smoke_20260725_iter2/checkpoints/last.pt`
- `outputs/runs/autotrain_wf_smoke_20260725_iter2/scoreboard.json`
- `outputs/autotrain-workflow/autotrain_wf_smoke_20260725_iter2/quality_gate.json`
- `outputs/autotrain-workflow/autotrain_wf_smoke_20260725_iter2/state.json`
- `outputs/autotrain-workflow/autotrain_wf_smoke_20260725_iter2/report.md`

`gates.json` is only written with `--ship-gates` (not this recipe).
