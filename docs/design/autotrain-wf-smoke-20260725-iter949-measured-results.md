# autotrain_wf_smoke_20260725_iter949

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_v2 last_loss=31.926565170288086 stopped_on=steps wall=1.6989173780000328 max_wall=2.5833333333333335 n=None scoreboard=False

Run against the canonicalized-marker fix (harness.train_data v22, this
session's PR). Command: `slm sft train --model twotower --context-backend
scratch --denoiser-backend scratch --steps 8 --seed 949 --run-id autotrain_wf_smoke_20260725_iter949
--no-full-state-checkpoint --no-sync-checkpoints` against
`slm data build-train --source fixture --version wf_smoke_v2 --no-publish`.
Artifacts at `outputs/runs/autotrain_wf_smoke_20260725_iter949/` (gitignored, not committed).
