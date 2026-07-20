# SDE0-02 metric-gaming stress suite

Date: 2026-07-19
Issue: SLM-162
Branch: `agent/slm-162-sde0-02-metric-gaming-suite`

## What this is

The SDE0-02 fixture is a deterministic adversarial suite for the existing
`binding_aware_meaningful_v2` judge. It does **not** train or evaluate a model.
It builds 101 hard-valid OpenUI programs that are designed to game surface-level
metrics while failing semantic meaning, then scores them with the production
meaningful-program metric.

## Slices

| slice | n | intent |
| --- | --- | --- |
| `minimal_valid` | 35 | Valid programs that strip requested content (empty shells, partial content, type swaps, nested empties) |
| `rare_omission` | 12 | Omit/substitute historically rare components (`Tabs`, `Slider`, `SwitchItem`, `Form`) |
| `inventory_free` | 26 | Pairs with and without explicit prompt slot inventory, plus partial inventories |
| `retry_sensitive` | 28 | First-attempt / selected-attempt / oracle-best comparisons across 2 and 3 attempts |

## Fixture results

```json
{
  "schema_version": "metric_gaming/v1",
  "metric_name": "binding_aware_meaningful_v2",
  "metric_version": "2.0.0",
  "n_cases": 101,
  "strict_rate": 1.0,
  "coverage_conditioned_rate": 1.0,
  "false_positive_count": 0,
  "false_negative_count": 0
}
```

Machine-readable results:

- `outputs/runs/sde0-02-metric-gaming/iter-sde0-02-20260719/summary.json`
- `outputs/runs/sde0-02-metric-gaming/iter-sde0-02-20260719/manifest.json`
- `docs/design/iter-sde0-02-metric-gaming-20260719.json`

## Honest caveats

- **Fixture only.** This run exercises the judge on hand-authored transforms;
  it is not evidence that a trained model passes ship gates.
- **No model, tokenizer, or GPU was used.** The suite is pure deterministic
  OpenUI source manipulation.
- **No checkpoint was written or promoted.** `docs/MODEL_CARD.md` was not
  updated because no model artifact was produced.
- The `strict_rate == 1.0` here means the current judge *rejects* these
  adversarial negatives and *accepts* the positives; it does not mean a model
  would generate them correctly.
- Retry-sensitive rows show that `first_attempt_pass`, `selected_attempt_pass`,
  and `oracle_best_pass` can differ; production sampling should be evaluated on
  the same axes, not on single-sample syntax parse.

## Files changed

- `src/slm_training/evals/metric_gaming.py` — suite implementation
- `scripts/run_sde0_02_metric_gaming_fixture.py` — fixture runner
- `tests/test_evals/test_metric_gaming.py` — regression tests
- `src/slm_training/resources/versions.json` — bumped `evals.scoring` to `v2`
- `docs/design/iter-sde0-02-metric-gaming-20260719.json` — machine-readable result
- `docs/design/iter-sde0-02-metric-gaming-20260719.md` — this memo
