# autotrain_wf_smoke_20260725_iter953

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_v2 last_loss=34.54191207885742 stopped_on=steps wall=1.7509064649999573 max_wall=2.5833333333333335 n=None scoreboard=False

Run against the canonicalized-marker fix (harness.train_data v22, this
session's PR). Command: `slm sft train --model twotower --context-backend
scratch --denoiser-backend scratch --steps 8 --seed 953 --run-id autotrain_wf_smoke_20260725_iter953
--no-full-state-checkpoint --no-sync-checkpoints` against
`slm data build-train --source fixture --version wf_smoke_v2 --no-publish`.
Artifacts at `outputs/runs/autotrain_wf_smoke_20260725_iter953/` (gitignored, not committed).
