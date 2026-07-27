# Joint seed x steps sweep (2026-07-27, scheduled autotrain-loop session)

**Honesty:** `fixture_or_scratch`. Not a ship claim.

Follow-up to the ledger's own "Next steps note" after batch #4: instead of
another single-variable check, this sweep varies `--seed` (1, 2, 3) *and*
`--steps` (4, 16) jointly against `main` HEAD `f3adde1b` (PR #1131, already
merged), same fixture/model/recipe as every prior batch:

```bash
python -m scripts.train_model \
  --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps <4|16> \
  --run-id <run_id> --no-sync-checkpoints --device cpu --seed <1|2|3>
```

Environment: fresh `.venv` (`python3.12 -m venv`, `pip install -e ".[dev,grammar]"`),
created in this scheduled session, not committed. Checked (not committed —
`outputs/` is gitignored) at `outputs/runs/<run_id>/train_summary.json`.

| run_id | steps | seed | stopped_on | last_loss | wall_s |
| --- | --- | --- | --- | --- | --- |
| `autotrain_wf_smoke_20260727_joint_seed1_steps4` | 4 | 1 | steps | 52.60234069824219 | 3.63 |
| `autotrain_wf_smoke_20260727_joint_seed2_steps4` | 4 | 2 | steps | 49.65409851074219 | 3.94 |
| `autotrain_wf_smoke_20260727_joint_seed3_steps4` | 4 | 3 | steps | 64.18412017822266 | 3.57 |
| `autotrain_wf_smoke_20260727_joint_seed1_steps16` | 16 | 1 | steps | 19.269824981689453 | 4.76 |
| `autotrain_wf_smoke_20260727_joint_seed2_steps16` | 16 | 2 | steps | 16.698604583740234 | 5.33 |
| `autotrain_wf_smoke_20260727_joint_seed3_steps16` | 16 | 3 | steps | 22.079015731811523 | 5.21 |

Combined with the already-verified single-variable rows (`seed=0` across
`steps` in {4, 8, 16, 32}; `steps=8` across `seed` in {0, 1, 2, 3}), the full
joint grid now covered is:

| steps \ seed | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| 4  | 56.89 | 52.60 | 49.65 | 64.18 |
| 8  | 32.61 | 38.95 | 38.32 | 36.62 |
| 16 | 15.10 | 19.27 | 16.70 | 22.08 |
| 32 | 8.39  | -     | -     | - |

**Result:** at every step count, seed-to-seed spread is real but bounded
(steps=4: 49.65-64.18, spread 14.53; steps=16: 15.10-22.08, spread 6.98) and
shrinks in absolute terms as steps increase, while the seed-0 loss is not
systematically the lowest or highest at any step count (steps=4: seed 0 is
second-highest; steps=16: seed 0 is lowest) -- i.e. seed does not
introduce a directional bias, only variance, consistent with it only
touching model initialization on this from-scratch `twotower` config. This
is genuinely new joint evidence (not a repeat of either single-variable
batch): it's the first check that seed-variance and step-count both act on
the same fixture without cancelling or compounding unexpectedly.

Still `fixture_or_scratch`: n=1 per (seed, steps) cell, a 101-record
fixture, `context-backend scratch`. No convergence, generalization, ranking,
or ship claim is made -- this only characterizes how much this fixed
recipe's loss moves under its own declared knobs.

**Next steps note (carried forward):** the smoke-loop's role as a harness
liveness + single/joint-variable check is now thoroughly covered. Per the
ledger's own prior note, the next scheduled iteration should move off this
fixed fixture and pick up one of the repo's actually-open threads (the
DSH5-10 SFT/preference-training scope -- noting the sixth slice, PR #1131,
found the existing `PreferencePair` composite-reward shape is a structural
mismatch for exact-state action-token rows, so the real next step there is
locating and wiring the `TypedOperatorPolicyScorer`
(`src/slm_training/harnesses/experiments/typed_operator_policy.py:316`) the
issue's own text names as the intended consumer -- or the next queued
`AP-007+` campaign arm).
