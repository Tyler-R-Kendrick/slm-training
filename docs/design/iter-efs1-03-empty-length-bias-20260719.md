# EFS1-03: Resolve valid-but-empty length bias on durable frontier checkpoints (SLM-110)

**Linear issue:** SLM-110
**Branch:** `agent/slm-110-efs1-03-empty-length-bias`
**Date:** 2026-07-19
**Status:** wiring fixture / score-policy plumbing; SLM-110 acceptance incomplete

Evidence: [iter-efs1-03-empty-length-bias-20260719.json](iter-efs1-03-empty-length-bias-20260719.json).
Harness: [`src/slm_training/evals/score_policy.py`](../../src/slm_training/evals/score_policy.py),
probe: [`src/slm_training/evals/emptiness_probe.py`](../../src/slm_training/evals/emptiness_probe.py),
fixture runner: [`scripts/run_empty_bias_fixture.py`](../../scripts/run_empty_bias_fixture.py).
Tests: [`tests/test_evals/test_score_policy.py`](../../tests/test_evals/test_score_policy.py),
[`tests/test_evals/test_emptiness_probe.py`](../../tests/test_evals/test_emptiness_probe.py).

## What changed

Added eval-only score-policy plumbing for the A1 / EFS1-03 empty-length-bias diagnostic:

- `src/slm_training/evals/score_policy.py`
  - `CandidatePath` dataclass holding token-level log-probs, optional removed mass, and optional semantic-decision mask.
  - Five score policies behind disabled flags:
    - `RawCumulativePolicy` — current greedy constrained-decoder ranking;
    - `SemanticLengthNormPolicy` — normalize by non-forced semantic decisions with tunable `alpha`;
    - `GrammarAlignedMassPolicy` — ASAp-style correction using per-step removed mass;
    - `MinimumMassRemaskPolicy` — soft content-pressure bonus/penalty on semantic positions;
    - `ContentFloorPolicy` — hard floor rejecting empty/minimal-shell candidates (diagnostic only).
  - `rank_candidates()` and `compare_policies()` helpers that report rank changes across policies.
- `src/slm_training/evals/emptiness_probe.py`
  - `EmptinessProbeConfig` now accepts optional `score_policies`.
  - `evaluate_emptiness()` builds `CandidatePath` objects for the populated and minimal-valid programs and reports per-policy scores, rankings, and the fraction of records where each policy still prefers the empty candidate.
- `scripts/run_empty_bias_fixture.py`
  - Synthetic fixture that demonstrates raw-cumulative preference for the empty candidate and a flip to populated under length-normalization / grammar-aligned-mass / content-floor policies.
- `tests/test_evals/test_score_policy.py` and `tests/test_evals/test_emptiness_probe.py`
  - Regression tests for policy math, rank changes, config plumbing, and emptiness-probe integration.
- `src/slm_training/resources/versions.json`
  - Bumped `evals.scoring` to `v2` and `evals.loss_suite` to `v2`.

## Fixture run

Command:

```bash
python -m scripts.run_empty_bias_fixture --run-id fixture-20260719
```

Recipe: CPU; synthetic candidates; no checkpoint load.

### Score-policy rankings on the fixture

| policy | first-ranked candidate | note |
| --- | --- | --- |
| raw_cumulative | empty | length bias: shorter candidate wins despite equal per-token score |
| semantic_length_norm (alpha=1.0) | populated | per-semantic-decision average removes length advantage |
| grammar_aligned_mass (beta=1.0) | populated | populated candidate retains more mass per step |
| minimum_mass_remask (gamma=0.5) | empty | wiring-level penalty is too weak in this hand-set fixture |
| content_floor (min=1) | populated | empty candidate has zero semantic decisions |

This is synthetic evidence only; the fixture proves the policy interface and rank-change plumbing.

## Honest caveats

- **Wiring-only / no checkpoint loaded.** No durable frontier checkpoint (E228, E396, etc.) was probed.
- **No live compiler decode.** Removed mass and semantic masks are hand-set; real runs must derive them from the compiler's live legal action sets.
- **Not a campaign.** SLM-110 remains incomplete until at least three durable checkpoints are probed on all five arms with candidate-level attribution.
- **No model card update.** Nothing is promoted.

## Verification checklist

- [x] `pytest tests/test_evals/test_score_policy.py` — 10 passed.
- [x] `pytest tests/test_evals/test_emptiness_probe.py` — 5 passed.
- [x] `python -m scripts.run_empty_bias_fixture --run-id fixture-20260719` — bundle written.
- [x] `python -m scripts.repo_policy` — ok.
- [x] `.githooks/check-changed` — 413 passed, 12 deselected.
- [x] `python -m scripts.verify_version_stamps --check` — ok.
- [x] `git diff --check` — clean.
