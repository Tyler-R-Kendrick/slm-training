# autotrain_wf_smoke_20260727_iter1022

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_v2 last_loss=32.610084533691406 stopped_on=steps wall=2.1 max_wall=2.5833333333333335 record_count=101

Independently run in this scheduled autotrain-loop session against `main` HEAD `e9bab2b` (same already-published `wf_smoke_v2` fixture, no local patch) via:

```bash
python -m scripts.train_model --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps 8 \
  --run-id autotrain_wf_smoke_20260727_iter1022 --no-sync-checkpoints --device cpu --seed 0
```

`outputs/runs/autotrain_wf_smoke_20260727_iter1022/train_summary.json` (not committed; `outputs/` is gitignored).
