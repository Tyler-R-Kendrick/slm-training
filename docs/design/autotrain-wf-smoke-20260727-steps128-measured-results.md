# autotrain_wf_smoke_20260727_steps128

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_v2 last_loss=5.877634525299072 stopped_on=steps wall=13.22 record_count=101 seed=0 steps=128
example_token_loss_proxy: first_20_mean=65.47187957763671 last_20_mean=3.269563728570938 count=500

Independently run in this scheduled autotrain-loop session against `main` HEAD `b908b543` (same already-published `wf_smoke_v2` fixture, no local patch), extending the step-scaling check to 128 steps (seed 0):

```bash
python -m scripts.train_model --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps 128 \
  --run-id autotrain_wf_smoke_20260727_steps128 --no-sync-checkpoints --device cpu --seed 0
```

Note: `last_loss` (5.88) is *higher* than the 64-step run's `last_loss` (3.92) even though the
smoothed `example_token_loss_proxy.last_20_mean` kept falling (4.42 -> 3.27) — `last_loss` is a
single final-minibatch value, not a running average, so it is noisy at this tiny fixture size
(101 records) and should not be read as a monotonic trend indicator on its own. See the
step-scaling section of the ledger for the full comparison and interpretation.

Environment: fresh `.venv` (Python 3.12.3, `torch==2.5.1+cpu`, `pip install -e .`) —
created in this scheduled session, not committed to the repo (`.venv/` is gitignored).

`outputs/runs/autotrain_wf_smoke_20260727_steps128/train_summary.json` (not committed; `outputs/` is gitignored).
