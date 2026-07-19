# SLM-127 / EFS3-04: Contract-grounded candidate selector with calibrated abstention

**Claim class:** wiring / fixture only  
**Run date:** 2026-07-19  
**Machine-readable result:** [`iter-slm127-candidate-selector-20260719.json`](iter-slm127-candidate-selector-20260719.json)

This iteration implements the SLM-127 candidate-selector harness. No checkpoint
was trained, no GPU was used, and no ship-gate claim is made.

## What landed

- `src/slm_training/harnesses/experiments/candidate_selector.py`
  - Frozen dataclasses: `SelectionCandidate`, `SelectionDecision`,
    `CandidateSelectionGroupV1`, `ThresholdManifestV1`.
  - `CandidateSelector` protocol with `selector_id` and keyword-only `select(...)`.
  - Baseline selectors: `ModelScoreSelector`, `ValueScoreSelector`,
    `EnergyScoreSelector`, `HardThenSimpleSelector`.
  - Learned selector: `ContractSelectorScorer` (tiny MLP) +
    `LearnedCandidateSelector`.
  - Calibration utilities: `select_threshold_on_validation`,
    `risk_coverage_curve`, `brier_score`, `expected_calibration_error`.
  - Evaluation metrics: `evaluate_selector`.
  - Training fixture: `train_selector_fixture`.
  - JSONL I/O helpers and synthetic fixture generator.
- `scripts/run_candidate_selector.py`
  - CLI arms: `--fixture`, `--groups`, `--selector`, `--calibrate-target-risk`,
    `--out`.
- Tests under `tests/test_harnesses/experiments/test_candidate_selector.py` and
  `tests/test_scripts/test_run_candidate_selector.py`.
- Registry entries: `harness.experiments` bumped to v21 and a new
  `harness.experiments.candidate_selector` v1 component.

## Selector interface

A selector implements the `CandidateSelector` protocol:

```python
@runtime_checkable
class CandidateSelector(Protocol):
    @property
    def selector_id(self) -> str: ...
    def select(
        self,
        *,
        prompt_context: Mapping[str, Any],
        structured_contract: Mapping[str, Any],
        candidates: Sequence[SelectionCandidate],
    ) -> SelectionDecision: ...
```

`SelectionCandidate` carries `generator_score`, `value_score`, `energy_score`,
`semantic_success`, `acceptable_set`, and an `available_features` map.
`SelectionDecision` records the selected candidate, abstention status, predicted
success probability when available, utility scores, and a reason code.

## Baselines

- `ModelScoreSelector`: highest `generator_score`.
- `ValueScoreSelector`: highest `value_score`.
- `EnergyScoreSelector`: highest `energy_score`.
- `HardThenSimpleSelector`: candidates with `semantic_success=True` are preferred;
  within that pool, highest `generator_score`; tie-break by `candidate_id`.

All baselines include deterministic tie-breaking and fall back to the first
`candidate_id` when scores are missing or NaN.

## Learned head

`ContractSelectorScorer` is a 2-layer MLP with ReLU activations. Input features
per candidate are:

1. `generator_score` (imputed to `0.0` when missing/NaN)
2. `value_score`
3. `energy_score`
4. candidate-set size `k`
5. feature count from `available_features`

Outputs:

- `utility_logit` — used for pairwise ranking over acceptable vs unacceptable.
- `contract_success_logit` — used for predicted acceptability.
- optional `set_has_success_logit` (not trained in the fixture).

`LearnedCandidateSelector` picks the candidate with the highest
`sigmoid(contract_success_logit)`. If a `ThresholdManifestV1` is supplied and the
maximum probability does not exceed the calibrated threshold, the selector
abstains.

## Calibration

`select_threshold_on_validation` sweeps candidate max-success thresholds on the
validation split only. It chooses the lowest threshold whose selected-error risk
(`selected non-acceptable / total selected`) is at most `--calibrate-target-risk`
(default `0.05`). Coverage is the fraction of validation groups in which the
selector makes a selection.

## Fixture results

The built-in fixture generates 8 groups (4 train, 2 validation, 2 test), 2
generators per group, `k=4`, with known acceptable sets and a few no-positive
groups. The tiny learned head overfits the fixture; with abstention enabled the
threshold is driven very high and the selector abstains on uncertain groups.

Key numbers from [`iter-slm127-candidate-selector-20260719.json`](iter-slm127-candidate-selector-20260719.json):

| Arm | pass@K | selected-pass@K | regret | abstention_rate | invalid_over_valid |
| --- | --- | --- | --- | --- | --- |
| model_score | 0.75 | 0.75 | 0.00 | 0.00 | 0 |
| value_score | 0.75 | 0.75 | 0.00 | 0.00 | 0 |
| energy_score | 0.75 | 0.75 | 0.00 | 0.00 | 0 |
| hard_then_simple | 0.75 | 0.75 | 0.00 | 0.00 | 0 |
| learned_no_abstain | 0.75 | 0.75 | 0.00 | 0.00 | 0 |
| learned_abstain | 0.75 | 1.00 | 0.50 | 0.625 | 0 |

The fixture is separable, so the baselines already perform perfectly. The
`learned_abstain` arm demonstrates the abstention mechanism: it refuses to pick
when uncertain, trades recall for precision, and keeps
`invalid_over_valid_count` at zero.

## Honest verdict

**`no_safe_direction` / wiring-only.** The harness compiles, the selector
protocol is stable, and calibration behaves as designed on a toy corpus. The
fixture is too small and too artificial to tell whether the learned head will
generalize to real OpenUI candidates. A production claim would require:

- A trained checkpoint producing diverse candidates,
- Independent validation labels (not overlapping with training prompts or
  checkpoints),
- A matched comparison against the baselines on honest ship-gate suites,
- Measured calibration on held-out groups, and
- An explicit audit that the selector does not use hidden gold channels.

Until then this is wiring and a reusable harness, not a ship result.
