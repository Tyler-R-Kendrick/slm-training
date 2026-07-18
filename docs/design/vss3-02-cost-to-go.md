# VSS3-02: Train a cost-to-go energy scorer that ranks live solver candidates only

**Issue:** SLM-70  
**Status:** wiring / fixture evidence. No train, eval, benchmark, model, checkpoint, or ship claim.

## What was added

A cost-to-go energy scorer that plugs into the verified-scope solver as a permutation-only ranker.

Solver-side contract in `src/slm_training/dsl/solver/energy_ranker.py`:

- `CandidateEnergyInput` / `CandidateEnergyOutput` — typed scoring envelope.
- `CandidateEnergyScorer` protocol and `EnergyCandidateRanker` implementing `CandidateRanker`.
- Fail-closed fallback: wrong-length, non-finite, membership-changing, or exception-throwing scorer output returns the canonical value order and records the fallback.

Model-side head in `src/slm_training/models/twotower.py`:

- `TwoTowerConfig` fields `cost_to_go_loss_weight` and `cost_to_go_hidden_dim`.
- Optional `cost_to_go_head` MLP built with `isolated_aux_init` when enabled.
- `score_candidates(state, hole_id, values, context_prompt=None)` returns `CandidateEnergyOutput`.
- `cost_to_go_loss(rows, prompts=None)` regresses `log1p(nodes + verifier_calls + backtracks + depth)` for rows with `cost_observed=True` and adds a pairwise ranking loss over candidates sharing the same `(state_fingerprint, hole_id)`.
- `UNKNOWN`/censored rows (`cost_observed=False`) are masked from the regression loss.
- Prefix `"cost_to_go_head."` added to optimizer groups and `allowed_missing` checkpoint set.

Harness wiring:

- `src/slm_training/harnesses/model_build/config.py` and `factory.py` propagate the new config fields.
- `scripts/train_model.py` accepts `--cost-to-go-loss-weight` and `--cost-to-go-hidden-dim`.
- `src/slm_training/harnesses/model_build/cost_to_go_train.py` provides head-only training (freeze backbone, optimize head) and a checkpoint-save path.
- `scripts/train_cost_to_go.py` thin CLI over the head-only trainer.

## Verified

- `ruff check` passes on touched files.
- `python -m compileall` passes.
- `pytest tests/test_dsl/test_energy_ranker.py tests/test_models/test_cost_to_go.py tests/test_dsl/test_solver_controller.py -q` → 45 passed.
- `python -m scripts.repo_policy` ok.
- `git diff --check` clean.
- `.githooks/check-changed` → all checks passed (590 passed across targeted suites).

## Design boundaries preserved

- `EnergyCandidateRanker` validates that output is a permutation of the exact live values; it cannot add or drop candidates.
- The energy head is trained only from replay-verified `CandidateCostRow` records; `cost_observed=False` rows are not treated as negatives.
- No support certification, verifier bypass, or unconstrained fallback is introduced.
- Old checkpoints load cleanly with the new head allowed as missing.

## Caveats

- This is fixture wiring only: the head trains on tiny synthetic rows in tests and has not been trained on a real VSS3-01 corpus or evaluated on a solver benchmark.
- Candidate features are currently the raw `token_ids` payload from `DomainValue`; richer state/hole features are future work.
- No checkpoint ship, model-card, or readiness claim is made.
