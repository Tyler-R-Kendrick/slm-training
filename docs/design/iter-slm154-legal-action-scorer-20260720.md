# SLM-154 (SPV3-01): Capacity-matched autoregressive legal-action scorer fixture

**Status:** fixture / wiring only.  
**Claim class:** `wiring`.  
**Honest verdict:** `fixture_wiring`.

This change implements a minimal, fixture-only legal-action scorer baseline. It is **not** a ship-ready training pipeline and does not integrate with the live compiler-choice decode loop. Real compiler-owned exact-state scoring is deferred to later SPV3 work.

## What this exercises

- `LegalActionScorerConfig` and a shared `LegalActionScorer` interface.
- Three capacity-matched variants: `global_head`, `mlp`, and `cross_attention`.
- Soft `PlanActionFeatures` fusion without changing legal membership.
- Forced-singleton skip, unsupported-pack abstention, and legal-set-only softmax.
- Schema-versioned checkpoint save/load with config-mismatch fail-closed behavior.
- `train_fixture_scorer` and `evaluate_fixture_scorer` for synthetic compiler-decision data.

## Scorer contract

- The scorer receives inference-available context/state features and the complete live legal action set.
- It returns one score per supplied candidate in the same order; it cannot add, remove, or reorder candidates.
- Soft plan features are fused via `PlanActionFeatures`; they never modify `A_G(s)`.
- Singleton decisions bypass the model and are recorded as `forced`.

## Fixture recipe

| Key | Value |
| --- | --- |
| `n_train` | 128 |
| `n_test` | 32 |
| `fixture_steps` | 40 |
| `fixture_lr` | 0.05 |
| `backend` | cpu |
| `scorer_id` | legal-action-scorer-v1 |

## Fixture result table

| Variant | Params | Initial loss | Final loss | Test accuracy | Forced | Abstained |
| --- | --- | --- | --- | --- | --- | --- |
| `global_head` | 24976 | 695.1060 | 168.9194 | 0.312 | 0 | 0 |
| `mlp` | 29073 | 196.9271 | 6.3339 | 1.000 | 0 | 0 |
| `cross_attention` | 40785 | 194.8232 | 168.8575 | 0.844 | 0 | 0 |

## Caveats

- Synthetic fixture decisions are used in place of real compiler `CompletionForest` states.
- No live TwoTower or grammar-diffusion decode loop is exercised.
- No ship gate is evaluated or weakened.
- The external scorer ceiling from SLM-108 is not wired in this baseline.

## Verification commands

```bash
python -m pytest tests/test_models/test_legal_action_scorer.py -q
python -m scripts.verify_version_stamps --check
```

Both commands passed on this branch at the time of writing.