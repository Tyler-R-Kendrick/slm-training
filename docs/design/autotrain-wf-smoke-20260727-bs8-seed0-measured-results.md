# autotrain_wf_smoke_20260727_bs8_seed0

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_v2 last_loss=28.973106384277344 stopped_on=steps batch_size=8 wall=7.63 max_wall=2.5833333333333335 record_count=101

Independently run in this scheduled autotrain-loop session against `main`
HEAD `5f94b92` (already merged), same fixture/recipe as
[bs1](autotrain-wf-smoke-20260727-bs1-seed0-measured-results.md) and
[bs2](autotrain-wf-smoke-20260727-bs2-seed0-measured-results.md), only
`--batch-size` changed:

```bash
python -m scripts.train_model --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps 8 --batch-size 8 \
  --run-id autotrain_wf_smoke_20260727_bs8_seed0 --no-sync-checkpoints --device cpu --seed 0
```

Environment: fresh `.venv` (Python 3.12, `torch==2.5.1+cu124`, `pip install -e ".[torch]"`),
created in this scheduled session, not committed to the repo (`.venv/` is
gitignored).

`outputs/runs/autotrain_wf_smoke_20260727_bs8_seed0/train_summary.json` (not
committed; `outputs/` is gitignored).
