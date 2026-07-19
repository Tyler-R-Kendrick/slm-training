# EFS3-01: Solver-state supervision source comparison (SLM-118)

**Linear issue:** SLM-118
**Branch:** `agent/slm-118-efs3-01-solver-state-supervision`
**Date:** 2026-07-19
**Status:** wiring fixture / data scaffolding; SLM-118 acceptance incomplete

Evidence: [iter-efs3-01-solver-state-supervision-20260719.json](iter-efs3-01-solver-state-supervision-20260719.json).
Harness: [`src/slm_training/evals/solver_state_supervision.py`](../../src/slm_training/evals/solver_state_supervision.py),
fixture runner: [`scripts/run_solver_state_supervision_fixture.py`](../../scripts/run_solver_state_supervision_fixture.py).
Tests: [`tests/test_evals/test_solver_state_supervision.py`](../../tests/test_evals/test_solver_state_supervision.py).

## What changed

Added eval/data wiring for the EFS3-01 hypothesis that solver-state supervision
corpora can be built from distinct sources and mixed DAgger-style, while
preserving honesty invariants.

- `src/slm_training/evals/solver_state_supervision.py`
  - `SupervisionSource` enum: `gold | on_policy | mixed`.
  - `SolverStateTrainingExampleV1` — serializable row carrying a solver-state
    fingerprint, legal actions, acceptable actions, support verdict,
    observed/censored cost-to-go, split/lineage identity, and replay-certification
    flag.
  - `SolverStateMixSpec` — named mix with source weights, seed, and optional
    per-source cap; weights normalize to a probability vector.
  - `MixResult` / `CompareResult` — containers for one mix and the canonical
    three-way comparison.
  - `build_solver_state_mix()` — builds one corpus with cross-split leakage
    rejection by `split_group_id`, held-out split exclusion, deterministic
    sampling, and per-row relabeling to the effective supervision source.
  - `compare_solver_state_supervision()` — convenience wrapper producing the
    pure-gold, pure-on-policy, and 50/50 mixed corpora with a version stamp.
- `scripts/run_solver_state_supervision_fixture.py`
  - Synthetic fixture generating solver-state rows from a mix of replay-certified
    gold states and on-policy rollout states, then emitting the three canonical
    mixes.
- `tests/test_evals/test_solver_state_supervision.py`
  - Regression tests for pure-source selection, 50/50 mix fraction, cross-split
    leak rejection, `UNKNOWN` verdict preservation, and JSON round-trip.
- `src/slm_training/resources/versions.json`
  - Bumped `evals.scoring` to `v3`.

## Fixture run

Command:

```bash
python -m scripts.run_solver_state_supervision_fixture --run-id iter-efs3-01-20260719
```

Recipe: CPU; synthetic solver states; no checkpoint load; deterministic seed 2026.

### Mix summary

| Mix | Rows | Rejected | Source tag |
| --- | --- | --- | --- |
| gold | 85 | 0 | gold |
| on_policy | 85 | 0 | on_policy |
| mixed | 170 | 0 | mixed |

Synthetic corpus: 200 rows across 40 problems (5 states each), with ~15% of
problems held out.  Train rows split evenly between gold and on-policy sources;
the mixed corpus samples both sources without replacement up to the train pool.

### Honesty invariants exercised

- `UNKNOWN` verdict rows are preserved, not relabeled.
- Held-out split rows are excluded from the train mixes.
- Cross-split leakage guard is enforced: any `split_group_id` appearing in both a
train split and a held-out split would be rejected.

## Honest caveats

- **Wiring-only / no checkpoint loaded.** No solver trace replay, model decode, or
  real search cost measurement was performed.
- **Synthetic rows.** The fixture uses deterministic synthetic states, not actual
  solver traces; see `slm_training.harnesses.distill.solver_supervision` for the
  replay-verified builder that produces real gold/on-policy rows.
- **No quality claim.** This fixture proves the schema and mixer work; it does not
  claim that any mix improves meaningful-parse or ship gates.
- **Not wired into training loops.** The mixes are emitted as artifacts; no SFT/RL
  run consumes them yet.

## Verification checklist

- [x] `pytest tests/test_evals/test_solver_state_supervision.py` — 8 passed.
- [x] `python -m scripts.run_solver_state_supervision_fixture --run-id iter-efs3-01-20260719` — bundle written.
- [x] `python -m scripts.repo_policy` — ok.
- [x] `.githooks/check-changed` — 420 passed, 12 deselected.
- [x] `python -m scripts.verify_version_stamps --check` — ok.
- [x] `git diff --check` — clean.
