# SLM-317 (LAR2-06): do-no-harm AR→repair hybrid screen

**Disposition: `inconclusive`** — LAR3 (learned repair integration) remains OPEN but NOT advanced: the screen neither cleared the value gate nor ruled the effect out at fixture power; a powered rerun is required before LAR3 can open or close.

Fixture-scale mechanism screen; **not a ship claim**.

## Preregistered (locked before results)

- safety: invalid-over-valid count of ar_repair_improved vs ar_only on the frozen safety set (all eval examples) must be exactly 0
- value: paired hard-valid improvement of ar_repair_improved over ar_only: Wilson 95% lower bound of improvements/n_paired must exceed 0.05 with zero damages
- disposition: repair_positive iff safety AND value AND reachability gates pass; repair_negative iff safety fails OR the value Wilson UPPER bound is below the minimum useful effect (effect ruled out); else inconclusive
- declared deviation: pre-SLM-305 4-action edit space is not reproducible on this branch (edit space extended in place, no legacy-subset knob); the historical arm keeps mutation_count value labels + legacy STOP-slot accounting only

## Arms (matched examples + seeds + budgets)

| arm | n | hard-valid rate | mean hard rank | invalid-over-valid vs AR |
| --- | --- | --- | --- | --- |
| ar_only | 16 | 1.000 | 3.000 | — |
| repair_only | 16 | 1.000 | 3.000 | 0 |
| ar_repair_historical | 16 | 1.000 | 3.000 | 0 |
| ar_repair_improved | 16 | 1.000 | 3.000 | 0 |
| oracle_commit | 16 | 1.000 | 3.000 | 0 |

## Paired outcomes vs `ar_only` (per-example, nothing aggregated away)

| arm | both valid | AR valid, arm invalid | AR invalid, arm valid | both invalid | unpaired |
| --- | --- | --- | --- | --- | --- |
| repair_only | 16 | 0 | 0 | 0 | 0 |
| ar_repair_historical | 16 | 0 | 0 | 0 | 0 |
| ar_repair_improved | 16 | 0 | 0 | 0 | 0 |
| oracle_commit | 16 | 0 | 0 | 0 | 0 |

## Advancement gates

- **Safety**: invalid-over-valid = 0 (must be 0) → PASS
- **Value**: improvements 0/16, Wilson 95% [0.0, 0.19360768053443644] vs minimum useful effect 0.05 → FAIL
- **Reachability/provenance**: slm299/291 artifacts present → PASS

## Commit reasons

| arm | reason | count |
| --- | --- | --- |
| ar_repair_historical | hard_regression | 16 |
| ar_repair_improved | no_improvement | 16 |
| oracle_commit | oracle_no_gain | 16 |

Oracle commit upper bound hard-valid rate: 1.000 (sanity ≥ improved hybrid: True).

Durable per-example commit decisions: `outputs/slm317/commit_decisions.jsonl`.

## Honesty

Fixture-scale wiring screen: the frozen corpus is the SLM-155 synthetic decision fixture (n eval decisions below any ship-gate prompt count), models are tiny CPU fixtures trained for a handful of steps, and no ship-gate claim is made. The commit rule is deterministic and metamorphism-invariant by construction; the screen measures the mechanism, not production quality. The historical arm's pre-SLM-305 4-action space is declared non-reproducible on this branch (see preregistered deviation).
