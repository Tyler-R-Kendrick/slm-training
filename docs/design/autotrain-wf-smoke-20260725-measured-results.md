# Autotrain workflow smoke — measured results (autotrain_wf_smoke_20260725)

**Honesty:** `fixture_or_scratch` wiring only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| recipe | smoke |
| run_id | `autotrain_wf_smoke_20260725` |
| train | fixture `wf_smoke_v1` (103 records, strict profile) |
| test | fixture `wf_smoke_v1` smoke suite |
| model | twotower / choice / scratch / cpu |
| steps | 8 (`--fast-train`, `--no-sync-checkpoints`) |
| ship_gates | false (`--eval-limit 3 --suites smoke`) |
| train last_loss | 41.52362060546875 |
| smoke n | 3 (diagnostic subset) |
| meaningful_program_rate | 0.0 |
| decode_timeout_count | 3 |
| AgentV criteria.pass | False |

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
- `outputs/runs/autotrain_wf_smoke_20260725/train_summary.json`
- `outputs/runs/autotrain_wf_smoke_20260725/checkpoints/last.pt`
- `outputs/runs/autotrain_wf_smoke_20260725/scoreboard.json`
- `outputs/runs/autotrain_wf_smoke_20260725/agentv/`
- `outputs/autotrain-workflow/autotrain_wf_smoke_20260725/quality_gate.json`
- `outputs/autotrain-workflow/autotrain_wf_smoke_20260725/state.json`
- `outputs/autotrain-workflow/autotrain_wf_smoke_20260725/report.md`

`gates.json` is only written with `--ship-gates` (not this recipe).

## Harness unblocks applied during the run

See `state.json` → `harness_fixes_applied`. Component bumps: `model.twotower` v247, `harness.model_build.eval` v60.
