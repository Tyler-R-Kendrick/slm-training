# Autotrain workflow smoke — measured results (autotrain_wf_smoke_20260726_iter2)

**Honesty:** `fixture_or_scratch` wiring only. **Not a ship claim.**

Continuing the [2026-07-26 autotrain smoke loop](autotrain-loop-ledger-20260726.md)
from a second fresh container (no cached `outputs/`), this iteration verifies
the #978 marker-canonicalization fix end to end from a clean checkout: fresh
`slm data build-train` → `slm data build-test` → `slm sft train` (8 steps,
cpu, scratch) → `slm eval model` (`--eval-limit 3 --suites smoke`), all
succeeded with no manual workaround.

## Run

| Field | Value |
| --- | --- |
| recipe | smoke |
| run_id | `autotrain_wf_smoke_20260726_iter2` |
| train | fixture (fresh build, 103 records, `wf_smoke_v1_iter66`) |
| test | fixture disjoint suite (`wf_smoke_v1_iter66`, smoke n=3) |
| model | twotower / choice / scratch / cpu |
| steps | 8 (`--fast-train`, `--no-sync-checkpoints`) |
| seed | 1 |
| ship_gates | false (`--eval-limit 3 --suites smoke`) |
| train last_loss | 41.331092834472656 |
| stopped_on | steps |
| elapsed_wall_seconds (sft) | 3.054784452999911 |
| max_wall_minutes | 2.5833333333333335 |
| smoke n | 3 (diagnostic subset) |
| meaningful_program_rate | 0.0 |
| structural_similarity | 0.1464 |
| placeholder_fidelity | 0.5277777777777778 |
| decode_timeout_count | 0 |
| AgentV criteria.pass | false (expected at 8 steps from scratch) |

## Phase status

| Phase | Status |
| --- | --- |
| data_train | ok (fresh build; #978 canonicalization fix confirmed working) |
| data_test | ok (fresh build) |
| sft | ok |
| eval | ok (wiring; criteria fail expected at 8 steps) |
| closeout | ok (this doc + ledger) |

## Artifacts

- `outputs/data/train/wf_smoke_v1_iter66/manifest.json`
- `outputs/data/eval/wf_smoke_v1_iter66`
- `outputs/runs/autotrain_wf_smoke_20260726_iter2/train_summary.json`
- `outputs/runs/autotrain_wf_smoke_20260726_iter2/checkpoints/last.pt`
- `outputs/runs/autotrain_wf_smoke_20260726_iter2/scoreboard.json`

`gates.json` is only written with `--ship-gates` (not this recipe).
