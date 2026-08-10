# SLM-483 (SIE-006): Evaluator construct-validity and human calibration campaign (EXP-SR-3)

**Claim class:** `diagnostic` (matches the locked exp-sr-3 catalogue entry)

**Catalogue:** `exp-sr-3`

**Primary metric (`evaluator_human_agreement_kappa`):** 0.0

**minimum_effect:** `0.1` -- kill_threshold=`0.0` (le)

## Acceptance snapshot

| Check | Value |
| --- | --- |
| meets_minimum_effect | False |
| killed_by_catalogue_gate | True |
| independence_ok | True |
| authorized | False |
| pair_n | 12 |
| vce010_disposition | complete |
| external_human_calibration_error | 0.22775000000000004 |
| admission_divergence_rate | 0.0 |
| promotion | False |

## Scope

Diagnostic-scale evidence scoped to the locked exp-sr-3 catalogue identity, computed entirely from VCE-010's own evaluator-calibration protocol (no forked mechanism). The external judge and its transport are a labeled mock (no real external-model credentials or network call); human raters are synthetic, seeded labels standing in for a real blinded pair study, per SLM-478's (SIE-005) frozen packet design. No real external-family judge or human calibration was performed in this environment -- SLM-483's own escape hatch: this is honest fixture/diagnostic-scale wiring evidence, not a construct-validity claim. No downstream experiment may cite 'authorized' here as calibrated evidence for a promotion/risk claim; a real external-judge/blinded-human run remains required before any such claim.

Command: `python -m scripts.run_sie006_calibration_campaign --mode diagnostic`

Full detail: `docs/design/iter-slm483-sie-006-calibration-campaign-20260810.json`.
