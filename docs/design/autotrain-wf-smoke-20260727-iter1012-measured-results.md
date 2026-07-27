# autotrain_wf_smoke_20260727_iter1012

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_v2 last_loss=32.610084533691406 stopped_on=steps wall=2.2370180519999963 max_wall=2.5833333333333335 record_count=101

Independently run in this session against `main` HEAD `abfe291` (post the `ae4b446` canonical-marker fix; see the ledger's integrity notice) via:

```bash
python -m scripts.train_model --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps 8 \
  --run-id autotrain_wf_smoke_20260727_iter1012 --no-sync-checkpoints --device cpu --seed 0
```

`outputs/runs/autotrain_wf_smoke_20260727_iter1012/train_summary.json` (not committed; `outputs/` is gitignored).
