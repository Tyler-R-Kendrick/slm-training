# SLM-312 (LAR2-04): seedward + on-policy state sources vs near-gold

**Verdict: `rejected`** (fixture-scale matched arms; not a ship claim)

## Preregistered (locked before results)

- arms/weights: `{"gold_only": {"gold_only": 1.0}, "seedward": {"seedward": 1.0}, "on_policy": {"on_policy": 1.0}, "mixed": {"gold_only": 0.34, "seedward": 0.33, "on_policy": 0.33}}`
- per-arm budget: 24 rows; identical model/steps/optimizer/seeds/evaluator
- beam regret improvement >= 0.05
- rank correlation degradation <= 0.05
- rule: distribution_gap_closed iff best non-gold arm improves beam regret >= 0.05 AND degrades rank correlation by <= 0.05, else rejected

## Headline (held-out seed trajectories)

| arm | rows | rank corr | concordance | Brier | beam regret | valid-final |
| --- | --- | --- | --- | --- | --- | --- |
| gold_only | 17 | -0.2582 | 0.3333 | 0.0340 | 0.0000 | 1.0000 |
| mixed | 24 | 0.2582 | 0.6667 | 0.0068 | 0.0000 | 1.0000 |
| on_policy | 18 | 0.2582 | 0.6667 | 0.0144 | 0.0000 | 1.0000 |
| seedward | 8 | 0.2582 | 0.6667 | 0.0301 | n/a | 1.0000 |

Best non-gold arm: **on_policy** — beam regret improvement 0.0000, rank correlation degradation -0.5164.

## State-space coverage by source (full snapshot)

| source | rows | unique fingerprints |
| --- | --- | --- |
| gold_only | 17 | 17 |
| on_policy | 18 | 18 |
| seedward | 8 | 8 |

Snapshot duplicate rate 0.140; 37/43 unique states. Manifest `1c5fc80b721b12f4…` at `outputs/slm312/state_snapshot.jsonl`.

## Coverage + oracle cost

- UNKNOWN coverage 0.333 (2/6), measurable 4, oracle 1397.1 ms/state

## Honesty

Fixture-scale matched arms over immutable content-addressed state snapshots; evaluation uses held-out seed trajectories only (fail-closed leak guard ran before training); UNKNOWN/unbounded oracle labels are excluded and counted, never coerced. A fixture verdict is wiring/state-source evidence, not a production ship claim.
