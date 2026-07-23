# SLM-279 recursive depth-supervision arithmetic correction

Verdict: **corrected; correction-only fixture evidence**. This is not a quality, readiness, or ship-gate result.

## Recipe

- Device/backend: `cpu` / `scratch`
- Optimizer/steps: `AdamW` / `1`
- Data/suite n: `synthetic_fixture` / `2`
- Honesty mode: `wiring_only_correction`

## Historical arithmetic correction

- Raw intermediate/final losses: `[28.09830665588379, 27.8798885345459]`
- Historical weights: `[0.5, 1.0]`
- Old buggy `sum(L_d) / sum(w_d)`: `37.31879679361979`
- Corrected `sum(w_d * L_d) / sum(w_d)`: `27.952694574991863`

## Canonical objective

- Mode: `intermediate_only`
- Auxiliary coefficient: `1.0`
- Primary final reconstruction: `27.8798885345459`
- Intermediate auxiliary: `28.09830665588379`
- Final-depth auxiliary contribution: `0.0`
- Combined loss: `55.97819519042969`

The final recursion supplies the primary reconstruction term and is structurally excluded from the auxiliary loop. Only depths `0..R-2` receive the normalized auxiliary weighting.

## Compatibility

Persisted configs that predate `recursive_depth_aux_mode` migrate to `legacy_all_depths`, preserving their old corrected all-depth behavior. New weighted configs must name their semantics explicitly.
