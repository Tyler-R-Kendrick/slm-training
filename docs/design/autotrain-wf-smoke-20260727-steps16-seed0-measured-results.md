# autotrain_wf_smoke_20260727_steps16_seed0

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_v2 steps=16 seed=0 last_loss=15.098441123962402 stopped_on=steps wall=7.03 record_count=101

Independently run in this scheduled autotrain-loop session against `main` HEAD `226df88` (already-published `wf_smoke_v2` fixture, no local patch) via:

```bash
python -m scripts.train_model --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps 16 \
  --run-id autotrain_wf_smoke_20260727_steps16_seed0 --no-sync-checkpoints --device cpu --seed 0
```

Environment: fresh `.venv-smoke` (Python 3.12.3, `torch==2.5.1+cu124`, `pip install -e ".[torch]"`) — created in this scheduled session, not committed to the repo (`.venv-smoke/` is gitignored).

`outputs/runs/autotrain_wf_smoke_20260727_steps16_seed0/train_summary.json` (not committed; `outputs/` is gitignored).
