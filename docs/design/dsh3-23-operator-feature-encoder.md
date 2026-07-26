# SLM-398 (DSH3-23): typed operator-feature-encoder matched-arm fixture

**Status:** fixture / wiring only.  
**Claim class:** `wiring`.  
**Honest verdict:** `fixture_wiring`.  
**Disposition:** `wiring_underpowered`.

Fixture-scale only (30 train / 20 eval decisions): typed top-1 recall 1.000 vs hash-scalar 1.000 is directional, not a powered claim. Per the DSH3-23 stop rule, adopt typed encoding only on a supported benefit at production scale; this run alone does not clear that bar in either direction.

## What this exercises

- `OperatorFeatureVocabularyV1` / `OperatorFeatureEncoder` over the SLM-397 sanitized `OperatorPolicyInputV1` boundary — three matched arms: `hash_scalar` (the existing `_stable_scalar` control), `typed` (learned per-field embeddings), and `typed_identity_bucket` (typed plus a labeled, row-position-only anti-generalization control).
- `CandidateScoringHead`, shared unchanged across arms.
- `make_fixture_operator_decisions` / `train_fixture_arm` / `evaluate_fixture_arm` / `run_matched_arms_fixture`.
- `permute_fixture_decision` — the same opaque-ID/row-order permutation adversarial control from SLM-397, replayed at the encoder layer.

## Fixture recipe

| Key | Value |
| --- | --- |
| `n_train` | 30 |
| `n_eval` | 20 |
| `n_candidates` | 5 |
| `dim` | 8 |
| `steps` | 150 |
| `lr` | 0.05 |
| `seed` | 0 |

## Matched-arm result table

| Arm | Params | Top-1 recall | NDCG | Accepted-set mass | Brier | ECE | Permutation-invariant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `hash_scalar` | 97 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0004 | True |
| `typed` | 841 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 | True |
| `typed_identity_bucket` | 969 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 | False |

## Adversarial control: permutation robustness

`typed` and `hash_scalar` never receive row position as a feature; both are structurally permutation-invariant and this run confirms it empirically (20/20 and 20/20 gold-logit matches under a row-order/opaque-ID reshuffle that preserves every allowed content field). `typed_identity_bucket` is the deliberately-unsafe control — its gold-logit matched only 3/20 times, confirming it can and does exploit build-time row order the way a real identity leak would, which is exactly why it is a labeled control and never a candidate for adoption.

## Caveats

- Synthetic single-slot, single-operator decisions — not a real compiler-produced `OperatorLegalSetV1` distribution.
- Only a minimal flat `CandidateScoringHead` is exercised; factorized (SLM-383/DSH3-15) and ECOC (SLM-404/DSH3-29) heads are separate tickets that reuse this same encoder/batch.
- No ship gate is evaluated or weakened. Fixture scale does not clear the ticket's own powered-comparison bar in either direction; see the disposition and stop rule.
- The ticket's own listed verification command `python -m scripts.run_capability_gate --capability CAP2_TRANSFORM --suite fixture --dry-run` does not correspond to any script in this repository (no `scripts/run_capability_gate.py` exists); it was not run.

## Verification commands

```bash
python -m pytest -q tests/test_models/test_operator_feature_encoder.py tests/test_models/test_legal_edit_batch.py
python -m scripts.verify_version_stamps --check
```

Both commands passed on this branch at the time of writing.