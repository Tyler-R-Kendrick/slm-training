# wf_smoke_v2_verify2_abfe291

**Honesty:** fixture_or_scratch. **Not ship.**

train_dir=src/slm_training/resources/data/train/wf_smoke_v2
last_loss=32.610084533691406 stopped_on=steps wall=2.2328272999999967
max_wall=2.5833333333333335 record_count=101

## Why this run exists

This is a scheduled-loop iteration of `docs/design/autotrain-loop-ledger-20260725.md`'s
fixture-smoke loop, run after the integrity notice at the top of that ledger
flagged the loop's ~1006 numbered rows (`iter1`-`iter1007`) as
historically-unverifiable: the recipe they describe has no corresponding
code path, and the fixture almost all of them cite postdates every one of
them. `wf_smoke_v2_verify_ae4b446` was the first row this repo can actually
vouch for. This is the second — a fresh, independent re-run of the same
recipe against a later `main` HEAD, to confirm the fix still holds and the
loop can continue honestly (real runs, not a fabricated ledger).

Before training, this run re-checked the canonicalization invariant the
original fix (`ae4b446`, PR #1015) established:

```bash
python3 -c "import json;print(any(not p.startswith(':slot_') for r in \
  map(json.loads, open('src/slm_training/resources/data/train/wf_smoke_v2/records.jsonl')) \
  for p in r['placeholders']))"
# -> False (all placeholders remain canonical :slot_N)
```

## Recipe

```bash
python3.12 -m venv .venv-smoke  # fresh, untracked, gitignored
source .venv-smoke/bin/activate
pip install --find-links <local wheel cache> -e ".[torch,dev]"

slm sft train --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps 8 \
  --run-id wf_smoke_v2_verify2_abfe291 --no-sync-checkpoints --device cpu
```

`build-train` was not re-run: `wf_smoke_v2` was used exactly as published by
`ae4b446`. Environment: `python3.12` venv (`.venv-smoke/`, untracked,
gitignored), `torch==2.5.1+cu124` (CPU execution). Code state: `main` HEAD
`abfe2910` (== `claude/great-dirac-1kvmhx` at the time this run started),
`code_dirty=false`.

## Result

`outputs/runs/wf_smoke_v2_verify2_abfe291/train_summary.json` (not committed —
`outputs/` is gitignored):

| field | value |
| --- | --- |
| steps | 8 |
| stopped_on | steps |
| last_loss | 32.610084533691406 |
| record_count | 101 |
| elapsed_wall_seconds | 2.2328272999999967 |
| max_wall_minutes | 2.5833333333333335 |
| data_manifest_sha | `fb320850f701ffe76170d2f26e570f1103d7815c7ea2186da846c6d185f21d10` |
| device | cpu |
| code_commit | `abfe29106e5de5f45198d3caf5f13a888ee77217` |

`last_loss` and `data_manifest_sha` match `wf_smoke_v2_verify_ae4b446` exactly
(deterministic fixture, same seed, same 8 steps, same 101 records) — expected
given no data or harness change between the two runs, and further evidence
the fixture and training path are stable and reproducible rather than
fabricated.

No `--ship-gates` scoreboard was requested; this is wiring verification only,
not a ship claim.

## Version stamp / component bump

No `no-bump:` note or `versions.json` component bump was needed: this run
exercised existing code (`harness.model_build.train` `v26`,
`harness.experiment_feature_flags` `v3`, both unchanged since the prior
verified row) against an already-published fixture. Nothing under
`src/slm_training/resources/versions.json`'s watch list changed.

## Artifacts (not committed; `outputs/` is gitignored)

- `outputs/runs/wf_smoke_v2_verify2_abfe291/train_summary.json`
- `outputs/runs/wf_smoke_v2_verify2_abfe291/checkpoints/last.pt`
- `outputs/runs/wf_smoke_v2_verify2_abfe291/trace.json`
