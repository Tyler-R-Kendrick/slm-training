# SLM-137 / LDI4-03: Intervention unification fixture (ldi4_03_fixture)

Matrix set: `ldi4-03-intervention-unification`
Version: `ldi4-03-v1`
Status: **wiring_only**

## What this exercises

A common manifest, registry, evaluation bundle, and promotion state machine for four intervention kinds: `causal_peft`, `twotower_delta`, `reft`, and `sae_diagnostic`. No model is loaded; this is wiring-only evidence.

## Kinds

`causal_peft`, `twotower_delta`, `reft`, `sae_diagnostic`

## Promotion transitions

| Intervention | From | To | OK | Failures |
| --- | --- | --- | --- | --- |
| peft-1 | wiring | diagnostic | True | — |
| peft-1 | diagnostic | eligible | True | — |
| peft-1 | eligible | promoted | True | — |
| delta-1 | diagnostic | eligible | False | protected ship gate failed; cannot become eligible |
| reft-1 | diagnostic | eligible | True | — |
| sae-1 | wiring | diagnostic | True | — |
| sae-1 | diagnostic | promoted | False | illegal transition diagnostic -> promoted |

## Closeout index

Total artifacts: 4
Best deployable: `peft-1`
Statement: best deployable intervention: peft-1

By status:
- wiring: []
- diagnostic: ['delta-1', 'sae-1']
- rejected: []
- eligible: ['reft-1']
- promoted: ['peft-1']

## Fixture caveat

Wiring-only evidence. Real model loading, merge/export parity, dashboard integration, and bucket upload are deferred to the integration run.
