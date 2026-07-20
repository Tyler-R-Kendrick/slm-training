# SLM-185 (FFE0-03): judge resolution audit fixture (slm185-judge-resolution-20260720)

Matrix set: `slm185_judge_resolution`

Version: `quality-v3`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no trainable weights were updated, and no ship-gate claim is made.

## Hypothesis

Deterministic fixture judges can expose test-retest reliability, canonical-equivalence invariance, and semantic perturbation detection at a per-endpoint resolution floor.

## Falsifier

The reliability metrics collapse (NaN/undefined) or the equivalence invariance error rate is non-zero for canonical-equivalent pairs.

## Endpoints

| endpoint | flip_rate | equiv_invariance_error | perturbation_detection | min_delta | margin |
| --- | --- | --- | --- | --- | --- |
| fixture_binding_aware_v2 | 0.0 | 0.0 | 1.0 | 0.05 | 0.01 |
| fixture_canonical_equal | 0.0 | 0.0 | 1.0 | 0.02 | 0.0 |
| fixture_seeded_hash_scorer | 0.7083333333333334 | 1.0 | 1.0 | 0.3 | 0.1 |

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The test-retest reliability, canonical-equivalence invariance, and semantic-resolution metrics are wired and exercised on deterministic synthetic judges, but no live external judge or production eval suite was used. The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_eval`` until it is validated with independent judges and real suite results.

## Honest caveats

- Fixture judges are deterministic and synthetic; real judges have sampling   variance, cost, latency, and provenance constraints not modeled here.
- The `fixture_seeded_hash_scorer` endpoint deliberately varies by repeat   seed to exercise flip-rate computation; it is not a semantic judge.
- Canonical-equivalent pairs are verified with `canonical_equal` using the   current D2 canonicalizer. Not every named surface transformation   (e.g., dead-binding prune) is normalized by the current canonicalizer;   only pairs that canonicalize equally are included.
- No ship-gate claim is made; this is wiring evidence only.

## Reproducibility

```bash
python -m scripts.run_judge_resolution_audit --mode describe
python -m scripts.run_judge_resolution_audit --mode build-corpus
python -m scripts.run_judge_resolution_audit --mode run --write-design-docs
python -m scripts.run_judge_resolution_audit --mode analyze-history \
  --history <scoreboard-or-manifest.json>
```
