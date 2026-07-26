# Autotrain workflow smoke — measured results (autotrain_wf_smoke_20260726)

**Honesty:** `fixture_or_scratch` wiring only. **Not a ship claim.**

Loop iter51 — continuation of
[`autotrain-loop-ledger-20260725.md`](autotrain-loop-ledger-20260725.md)
(iter1–iter50) into
[`autotrain-loop-ledger-20260726.md`](autotrain-loop-ledger-20260726.md).
This run first reproduced, then verified the fix for, the regression in
[`autotrain-smoke-canonical-marker-regression-20260726.md`](autotrain-smoke-canonical-marker-regression-20260726.md)
(`harness.train_data` v20 → v21).

| Field | Value |
| --- | --- |
| recipe | smoke (loop iter51) |
| run_id | `autotrain_wf_smoke_20260726` |
| parent_run | `autotrain_wf_smoke_20260725` (prior day's loop head) |
| train | fixture `wf_v0` (rebuilt; 103 records, strict profile) |
| test | fixture `wf_v0` smoke suite |
| model | twotower / choice / scratch / cpu |
| steps | 8 (`--fast-train`, `--no-sync-checkpoints`) |
| seed | 0 |
| ship_gates | false (`--eval-limit 3 --suites smoke`) |
| train last_loss | 30.21893310546875 |
| stopped_on | steps |
| elapsed_wall_seconds (train) | 6.571167230000015 |
| max_wall_minutes | 2.5833333333333335 |
| smoke n | 3 (diagnostic subset) |
| meaningful_program_rate | 0.0 |
| decode_timeout_count | 3 |
| AgentV criteria.pass | False |

## Phase status

| Phase | Status |
| --- | --- |
| data_train | ran (rebuilt `wf_v0`; prior fixture output not present in this checkout) |
| data_test | ran (rebuilt `wf_v0`) |
| quality | ok |
| sft | ok (after fix; failed with `ValueError: persisted template markers must use opaque :slot_<ordinal> identities` before) |
| eval | ok (wiring; criteria fail expected) |
| closeout | ok |
| verify | ok |

## Artifacts

- `outputs/data/train/wf_v0`
- `outputs/data/eval/wf_v0`
- `outputs/runs/autotrain_wf_smoke_20260726/train_summary.json`
- `outputs/runs/autotrain_wf_smoke_20260726/checkpoints/last.pt`
- `outputs/runs/autotrain_wf_smoke_20260726/scoreboard.json`
- `outputs/runs/autotrain_wf_smoke_20260726/agentv/`

`gates.json` is only written with `--ship-gates` (not this recipe).

## Harness fix applied during this run

See
[`autotrain-smoke-canonical-marker-regression-20260726.md`](autotrain-smoke-canonical-marker-regression-20260726.md).
Component bump: `harness.train_data` v21.
