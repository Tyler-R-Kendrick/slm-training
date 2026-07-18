# Distillation phase

Trace-driven self-distillation. Owner: `src/slm_training/harnesses/distill/`.

## Prerequisites

- A checkpoint to roll out; versioned train/eval data for anchors and scoring.
- P1 companions when climbing: mixture search / scaling ladder (experiments
  phase).

## Commands

```bash
# P1: collect decision trajectories from a checkpoint
python -m scripts.collect_trajectories --checkpoint outputs/runs/<id>/last.pt \
  --out <trace-store> --limit 200

# P2: stratified selection, then SFT from selected traces (+ anchor mix)
python -m scripts.self_distill select --traces <trace-store> --out <corpus>
python -m scripts.self_distill train --corpus <corpus> \
  --checkpoint outputs/runs/<id>/last.pt --anchor-train-dir outputs/data/train/v1

# P3: resume-climb rollouts → trajectory RL → ship gates (RL gated)
python -m scripts.resume_climb --checkpoint <pt> --test-dir outputs/data/eval/v1 \
  --rl-readiness-report <approved.json>   # or --skip-rl for the SFT-only climb
```

## Key flags

`collect`: `--decode-policy`, `--counterfactual-*`, `--record-support`;
`self train`: `--lambda-anchor`, `--lambda-traj`, `--budget`, `--steps`;
`resume-climb`: `--samples-per-prompt`, `--rl-steps`, `--skip-rl`.

## Outputs

Trace stores and distilled checkpoints inside the owning run/campaign
(`outputs/runs/<id>/`).

## Gates & invariants

- Selection data stays disjoint from frozen evals; never train on held-out
  benchmark traces.
- Checkpoint hashes/lineage labels preserved; generated records are never
  silently relabeled as gold.
- The RL leg is fail-closed behind an approved `RLReadinessReport` (rl phase).

## Close out

- Iron law docs + model-card duty when a new serving checkpoint is written
  (`documenting-experiment-results`).
- Checks: `pytest -q tests/test_harnesses/distill tests/test_models/test_trace_store.py`.
