# autotrain_wf_smoke_20260725_iter252

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_v1 last_loss=38.951087951660156 stopped_on=steps wall=6.095962753000094 max_wall=2.5833333333333335 n=3

Stacked on top of iter251's marker-canonicalization fix (PR #1006); this
iteration reuses the already-built `wf_smoke_v1` train/eval fixture corpus
(no rebuild, no harness change) to continue the smoke loop.

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

## Eval (suite=smoke, n=3, no --ship-gates)

| metric | value |
| --- | --- |
| meaningful_program_rate | 0.3333333333333333 |
| decode_timeout_count | 0 |
| placeholder_fidelity | 0.5277777777777778 |
| structural_similarity | 0.17416666666666666 |
| component_type_recall | 0.25 |
| reward_score | 0.7653333333333334 |

## Artifacts

- `outputs/runs/autotrain_wf_smoke_20260725_iter252/train_summary.json`
- `outputs/runs/autotrain_wf_smoke_20260725_iter252/checkpoints/last.pt`
- `outputs/runs/autotrain_wf_smoke_20260725_iter252/scoreboard.json`

`gates.json` is only written with `--ship-gates` (not this recipe). `outputs/`
is ephemeral per container; artifacts above are not committed.
