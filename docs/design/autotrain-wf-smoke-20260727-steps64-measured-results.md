# autotrain_wf_smoke_20260727_steps64

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_v2 last_loss=3.9243931770324707 stopped_on=steps wall=13.13 record_count=101 seed=0 steps=64
example_token_loss_proxy: first_20_mean=65.47187957763671 last_20_mean=4.4185902833938595 count=250

Independently run in this scheduled autotrain-loop session against `main` HEAD `b908b543` (same already-published `wf_smoke_v2` fixture, no local patch), varying `--steps` from 8 to 64 (seed 0, otherwise identical recipe to `iter1008`-`iter1022`) to check whether the fixture recipe shows a real loss-decrease trend beyond the 8-step smoke probe, rather than just reproducing a fixed single-batch number:

```bash
python -m scripts.train_model --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps 64 \
  --run-id autotrain_wf_smoke_20260727_steps64 --no-sync-checkpoints --device cpu --seed 0
```

Environment: fresh `.venv` (Python 3.12.3, `torch==2.5.1+cpu`, `pip install -e .`) —
created in this scheduled session, not committed to the repo (`.venv/` is gitignored).

`outputs/runs/autotrain_wf_smoke_20260727_steps64/train_summary.json` (not committed; `outputs/` is gitignored).
