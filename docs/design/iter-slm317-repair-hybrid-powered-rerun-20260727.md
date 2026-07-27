# SLM-317 (LAR2-06): do-no-harm AR→repair hybrid screen

**Disposition: `repair_negative`** — LAR3 (learned repair integration) stays CLOSED: the powered rerun damaged valid AR programs or ruled out the preregistered minimum per-seed pass rate; the second reopening condition is not met.

Fixture-scale mechanism screen; **not a ship claim**.

## Preregistered (locked before results)

- safety: invalid-over-valid count of ar_repair_improved vs ar_only on the frozen safety set (all eval examples) must be exactly 0
- value: paired hard-valid improvement of ar_repair_improved over ar_only: Wilson 95% lower bound of improvements/n_paired must exceed 0.05 with zero damages
- disposition: repair_positive iff safety AND value AND reachability gates pass; repair_negative iff safety fails OR the value Wilson UPPER bound is below the minimum useful effect (effect ruled out); else inconclusive
- declared deviation: pre-SLM-305 4-action edit space is not reproducible on this branch (edit space extended in place, no legacy-subset knob); the historical arm keeps mutation_count value labels + legacy STOP-slot accounting only

## Arms (matched examples + seeds + budgets)

| arm | n | hard-valid rate | mean hard rank | invalid-over-valid vs AR |
| --- | --- | --- | --- | --- |
| ar_only | 40 | 1.000 | 3.000 | — |
| repair_only | 40 | 0.000 | 1.000 | 40 |
| ar_repair_historical | 40 | 1.000 | 3.000 | 0 |
| ar_repair_improved | 40 | 1.000 | 3.000 | 0 |
| oracle_commit | 40 | 1.000 | 3.000 | 0 |

## Paired outcomes vs `ar_only` (per-example, nothing aggregated away)

| arm | both valid | AR valid, arm invalid | AR invalid, arm valid | both invalid | unpaired |
| --- | --- | --- | --- | --- | --- |
| repair_only | 0 | 40 | 0 | 0 | 0 |
| ar_repair_historical | 40 | 0 | 0 | 0 | 0 |
| ar_repair_improved | 40 | 0 | 0 | 0 | 0 |
| oracle_commit | 40 | 0 | 0 | 0 | 0 |

## Advancement gates

- **Safety**: invalid-over-valid = 0 (must be 0) → PASS
- **Value**: improvements 0/40, Wilson 95% [6.938893903907228e-18, 0.0876216011972866] vs minimum useful effect 0.05 → FAIL
- **Reachability/provenance**: slm299/291 artifacts present → PASS

## Preregistration (powered rerun, locked before any new seed was read)

- seed pass rule: a seed passes iff ar_repair_improved shows >=1 paired improvement over ar_only (AR invalid, arm valid) with 0 damages (AR valid, arm invalid) on that seed's frozen SLM-155 eval decisions
- disposition rule: repair_positive iff safety AND reachability gates pass AND the Wilson 95% lower bound of the per-seed pass rate clears min_pass_rate; repair_negative iff safety fails OR the Wilson UPPER bound falls below min_pass_rate (pass rate ruled out); else inconclusive_underpowered
- seed count: `5` (disjoint from SLM-317's original [0, 1])
- declared deviation (SLM-434): 5 seeds (2..6) instead of SLM-421's mirrored
  20 — a 20-seed run measurably exceeds `MAX_RUN_MINUTES=3`
  (`src/slm_training/levers.py`) at ~16 s/seed decode cost on this shared CPU
  host (measured wall 158.5 s for 5 seeds; 8 seeds exceeded 179 s and was
  aborted, never used as evidence). The disposition still resolves: with 0/5
  passing seeds the Wilson 95% upper bound (0.4345) falls below the
  preregistered 0.5, so the pass rate is ruled out rather than left
  underpowered. The negative is structural, not a power artifact: `ar_only`
  is hard-valid on all 40 paired outcomes (no headroom), so no seed can pass
  the >=1-improvement rule on this frozen fixture corpus.
- observed passing seeds: `0`
- observed pass rate: `0.0000` (Wilson 95% CI [`0.0000`, `0.4345`], n=`5`) vs preregistered min pass rate `0.5`

| seed | improvements | damages | pass |
| ---: | ---: | ---: | --- |
| 2 | 0 | 0 | FAIL |
| 3 | 0 | 0 | FAIL |
| 4 | 0 | 0 | FAIL |
| 5 | 0 | 0 | FAIL |
| 6 | 0 | 0 | FAIL |

## Commit reasons

| arm | reason | count |
| --- | --- | --- |
| ar_repair_historical | hard_regression | 40 |
| ar_repair_improved | hard_regression | 40 |
| oracle_commit | oracle_no_gain | 40 |

Oracle commit upper bound hard-valid rate: 1.000 (sanity ≥ improved hybrid: True).

Durable per-example commit decisions: `outputs/slm317/commit_decisions_powered_20260727.jsonl`.

## Honesty

Fixture-scale wiring screen: the frozen corpus is the SLM-155 synthetic decision fixture (n eval decisions below any ship-gate prompt count), models are tiny CPU fixtures trained for a handful of steps, and no ship-gate claim is made. The commit rule is deterministic and metamorphism-invariant by construction; the screen measures the mechanism, not production quality. The historical arm's pre-SLM-305 4-action space is declared non-reproducible on this branch (see preregistered deviation). SLM-431 powered rerun: only the seed count and the value decision rule changed (per-seed pass/fail + Wilson interval, preregistered and locked before any new seed was read); the corpus, matched arms, do-no-harm commit rule, safety gate, and reachability gate are byte-identical to SLM-317. New seeds are disjoint from SLM-317's original [0, 1].
