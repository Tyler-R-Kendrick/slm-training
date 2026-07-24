# SLM-308 (LAR2-02): bounded-distance value labels vs mutation-count

**Verdict: `rejected`** (fixture-scale matched arms; not a ship claim)

## Preregistered thresholds (locked before results)

- beam regret improvement (A−B) >= 0.05
- rank correlation improvement (B−A) >= 0.1
- rule: adopted iff both thresholds are met, else rejected

## Headline

| metric | arm A (mutation_count) | arm B (bounded_distance) | improvement |
| --- | --- | --- | --- |
| rank correlation | 0.4330 | 0.5052 | 0.0722 |
| concordance | 0.7500 | 0.7917 | 0.0417 |
| Brier | 0.0308 | 0.0043 | 0.0265 |
| beam regret (mean) | 0.8889 | 0.8889 | 0.0000 |

## Coverage + oracle cost

- UNKNOWN coverage 0.175 (7/40), measurable 17, oracle 661.7 ms/state, 64 nodes expanded

## Calibration (arm B, distance bins d=0..8+)

| d | n | mean predicted value | mean target |
| --- | --- | --- | --- |
| 1 | 8 | 0.787 | 0.875 |
| 2 | 9 | 0.785 | 0.750 |

## Honesty

Fixture-scale matched arms; UNKNOWN/unbounded states are excluded from distance-referenced metrics and counted, never coerced. A fixture verdict is wiring/label evidence, not a production ship claim.
