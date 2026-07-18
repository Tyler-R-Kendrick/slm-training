# VSS3-02 (SLM-70): cost-to-go energy scorer — fixture note

**Date:** 2026-07-18
**Scope:** wiring + guarantee evidence for the learned solver energy scorer. **No
real checkpoint was trained; no GPU run, eval, or matrix executed. Fixture overfit
is not a ship claim and `MODEL_CARD.md` is intentionally not updated.**

## What was built

- [`models/solver_energy.py`](../../src/slm_training/models/solver_energy.py):
  - `CandidateEnergyScorer` — small MLP mapping `(state, hole, candidate)`
    features to a scalar energy (aux-adapter; not baked into the base state dict,
    so old checkpoints load unchanged).
  - `CandidateEnergyRanker` — a `CandidateRanker` that **orders** the exact live
    candidates and is guaranteed to return a **permutation**; a count mismatch,
    duplicate, NaN, or infinite energy triggers the deterministic identity
    fallback (counted in `fallback_count`) and never alters the hard domain.
  - Versioned work target (`v1`): `work = expanded_nodes + wᵥ·verifier_calls +
    w_b·backtracks + w_d·decisions`, `cost_target = log1p(work)`, consuming the
    replay-verified VSS3-01 `candidate_cost` rows.
  - Masked losses: Huber regression **only** where `cost_observed=true`; pairwise
    ranking over same-state/hole pairs with *distinct* observed costs.
- `TwoTowerConfig` gains disabled-by-default fields (`solver_energy_head`,
  `solver_ranker`, `solver_energy_hidden_dim`, `solver_energy_{loss,pairwise}_weight`,
  `solver_energy_cost_version`, `solver_energy_fallback`).
- [`tests/test_models/test_solver_energy.py`](../../tests/test_models/test_solver_energy.py)
  — 12 tests.

## Fixture overfit (CPU, deterministic)

A tiny 3-candidate state (works `5 / 10 / 20` → `log1p` targets) trained with Adam
for 300 steps under `torch.manual_seed(0)`: the regression loss dropped by **>10×**
and the ranker then ordered candidates by ascending work (cheapest `payload=2`
first, dearest `payload=3` last → `[2, 1, 3]`). This demonstrates the training path
wires end-to-end and produces a cost-ordered ranking; it is **not** evidence of
model quality.

## Guarantees proven by the fixture tests

- Scorer returns exactly one energy per live candidate.
- Ranking is a permutation — the hard candidate set is identical before/after.
- Missing / extra / duplicate / NaN / infinite energies fall back deterministically
  and cannot alter membership.
- Certified-removed values (absent from the live set handed to `rank`) can never be
  reintroduced.
- Regression loss masks `UNKNOWN`/censored rows; equal-cost supported alternatives
  are not forced into a one-hot order.
- Pairwise loss prefers lower observed cost; a fixed seed produces a stable order.
- No final-source/witness text can enter the scorer input contract.
- Config is disabled by default, so existing configs/checkpoints load unchanged.

## Honesty / non-goals

Low energy means "expected to reach a verified terminal with less exact search
work," never "correct." The scorer has no authority over legality, membership,
support certification, `UNKNOWN`, or the final verifier. No learned pruning, no
sequence-density EBM, no topology-diffusion integration, no checkpoint claim. Full
production wiring (model-build harness CLI training path, in-checkpoint head
migration, `eval_policy` metrics) is deliberately out of this fixture increment and
is not implied to exist.

## Verification

```
python -m pytest tests/test_models/test_solver_energy.py -q          # 12 passed
python -m pytest tests/test_harnesses/distill/test_solver_trace.py -q  # 4 passed (config compat)
python -m ruff check src/slm_training/models/solver_energy.py \
    tests/test_models/test_solver_energy.py src/slm_training/models/twotower.py  # clean
python -m scripts.repo_policy                                        # ok
git diff --check                                                     # clean
```
