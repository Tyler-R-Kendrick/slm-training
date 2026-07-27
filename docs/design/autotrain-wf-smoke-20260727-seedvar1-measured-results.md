# autotrain_wf_smoke_20260727_seedvar1

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_v2 last_loss=38.951087951660156 stopped_on=steps wall=2.07 record_count=101 seed=1

Independently run in this scheduled autotrain-loop session against `main` HEAD `0e54c5bd` (same already-published `wf_smoke_v2` fixture, no local patch), varying only `--seed` from the deterministic `iter1008`-`iter1022` batches (all seed 0) to check whether the recipe actually responds to it:

```bash
python -m scripts.train_model --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps 8 \
  --run-id autotrain_wf_smoke_20260727_seedvar1 --no-sync-checkpoints --device cpu --seed 1
```

Environment: fresh `.venv` (Python 3.12.3, `torch==2.5.1+cpu`, `pip install -e .`) —
created in this scheduled session, not committed to the repo (`.venv/` is gitignored).

`outputs/runs/autotrain_wf_smoke_20260727_seedvar1/train_summary.json` (not committed; `outputs/` is gitignored).
