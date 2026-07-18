# RL phase

GRPO-lite online RL plus external backends. Owner:
`src/slm_training/harnesses/rl/`; integrations under
`src/slm_training/integrations/`.

## Prerequisites

- An **approved `RLReadinessReport`** — produced by
  `python -m scripts.autoresearch validate-rl` (autoresearch phase) and proving
  the frozen five-suite evaluation, full `rico_held`, honest ship gates, AgentV
  pass, and nonzero reward variance. There is no override.
- A reference checkpoint and train records.

## Commands

```bash
# GRPO-lite online RL (fail-closed on the readiness report)
python -m scripts.train_rl --checkpoint outputs/runs/<id>/last.pt \
  --train-records outputs/data/train/v1 \
  --rl-readiness-report <approved.json> --steps 15 --group-size 4

# External backends (env-configured wrappers; no argparse flags):
python -m scripts.run_nemo_rl    # NVIDIA NeMo-RL — see nemo-rl-hf-jobs-etiquette
python -m scripts.run_molt_rl    # MOLT
```

Prefer submitting/reconciling external jobs through the lifecycle CLI so lineage
stays canonical: `python -m scripts.model_cycle submit-nemo|reconcile-nemo|
submit-molt|reconcile-molt` (checkpoints phase).

## Key flags

`--kl-beta`, `--group-size`, `--lr`, `--limit`, `--ref-checkpoint`, `--out-dir`.

## Outputs

Trajectories and RL checkpoints under the owning run; external jobs reconcile
into canonical lineage.

## Gates & invariants

- RL readiness is fail-closed; a rejected report is evidence, never an obstacle
  to route around.
- Hardware smoke is never promotion evidence.
- No paid GPU or remote job without explicit user approval.

## Close out

- Iron law docs; model-card duty when a new serving checkpoint is written
  (`documenting-experiment-results`).
- Checks: `pytest -q tests/test_harnesses/rl
  tests/test_harnesses/quality/test_rl_curriculum_telemetry.py`.
