# wf_smoke_v2_verify_ae4b446

**Honesty:** fixture_or_scratch. **Not ship.**

train_dir=src/slm_training/resources/data/train/wf_smoke_v2
last_loss=32.610084533691406 stopped_on=steps wall=13.607101259999979
max_wall=2.5833333333333335 record_count=101

## Why this run exists

This scheduled run was asked to continue
`docs/design/autotrain-loop-ledger-20260725.md`'s fixture-smoke loop past
`iter947`. Before adding more rows, it investigated the loop's provenance and
found that the `wf_smoke_v2` fixture almost every historical row claims to
have trained against had **no commit in this repository's history prior to
`main` HEAD `ae4b446`** (`fix(train_data): unblock and continue autotrain
smoke loop through iter340`, PR #1015) — `git log --follow` on
`src/slm_training/resources/data/train/wf_smoke_v2/records.jsonl` returns
exactly one commit: `ae4b446` itself. Independently, this run reproduced
(100%, prior to that commit) the exact crash three prior sessions had
already hit and failed to land a fix for:

```text
ValueError: persisted template markers must use opaque :slot_<ordinal> identities
```

See the integrity notice at the top of `autotrain-loop-ledger-20260725.md`
for the full account. `ae4b446` (already merged by the time this notice
landed) fixes the actual root cause — `_normalize_record`
(`src/slm_training/harnesses/train_data/pipeline.py`) now canonicalizes
persisted markers to opaque `:slot_N` before returning, and republished a
genuinely canonical `wf_smoke_v2`. This doc records an independent
re-verification of that fix, run fresh against `main` exactly as merged (no
local patch), so the ledger has at least one row it can actually vouch for.

## Recipe

```bash
slm sft train --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps 8 \
  --run-id wf_smoke_v2_verify_ae4b446 --no-sync-checkpoints --device cpu
```

`build-train` was not re-run: `wf_smoke_v2` was used as already published by
`ae4b446`. Environment: `python3.12` venv (`.venv-smoke/`, untracked,
gitignored), `torch==2.5.1+cu124` (CPU execution).

## Result

`outputs/runs/wf_smoke_v2_verify_ae4b446/train_summary.json`:

| field | value |
| --- | --- |
| steps | 8 |
| stopped_on | steps |
| last_loss | 32.610084533691406 |
| record_count | 101 |
| elapsed_wall_seconds | 13.607101259999979 |
| max_wall_minutes | 2.5833333333333335 |
| device | cpu |
| code_commit | ae4b446870656090a2eb3cc167136367eaacb765 |

No `--ship-gates` scoreboard was requested; this is wiring verification only,
not a ship claim.

## Artifacts (not committed; `outputs/` is gitignored)

- `outputs/runs/wf_smoke_v2_verify_ae4b446/train_summary.json`
- `outputs/runs/wf_smoke_v2_verify_ae4b446/checkpoints/last.pt`
- `outputs/runs/wf_smoke_v2_verify_ae4b446/trace.json`
