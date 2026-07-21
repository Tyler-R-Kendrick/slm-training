# SLM-231 (SPV0-05): RoleSlot cardinality dead-field consumption probe (slm231-spv0-05-role-cardinality-dead-field-20260721)

**Matrix set:** `slm231_role_cardinality_dead_field`
**Version:** `spv0-05-v1`
**Status:** fixture
**Claim class:** wiring
**Corpus:** SLM-144 fixture, n=19, seed=0
**Harness-local derivation populated cardinality:** True
**Gate hash:** `76e1c590ea5712f9...`
**Disposition:** cardinality_confirmed_unconsumed — Every fixture-corpus record produced byte-identical seed text, ok flag, and reason whether or not RoleSlot.min_cardinality/max_cardinality were populated on the roles+topology oracle arm. Combined with SLM-230's producer-side finding (extractor never populates these fields), the cardinality fields are confirmed to have neither a producer nor a consumer in the current pipeline: a fully dead field, not merely an unpopulated one.

## Hypothesis

Populating RoleSlot.min_cardinality/max_cardinality with a deterministic harness-local candidate derivation (observed same-component_family sibling count per parent), then oracle-substituting the `roles`+`topology` factors as in SLM-230's C4 arm, produces byte-identical PlanSeedBuilder/OpenUISemanticPlanCompiler output (seed text, ok flag, reason) to the same arm with cardinality left at None (today's production extractor output), on every record of the SLM-144 fixture corpus -- showing the cardinality fields are not just unpopulated (SLM-230) but also fully unconsumed by the current PlanSeedBuilder mechanism: a field with neither a producer nor a consumer.

## Falsifier

Any fixture-corpus record produces a different seed text, `ok` flag, or `reason` between the cardinality-populated arm and the cardinality-None arm, holding every other factor identical -- showing something in the current PlanSeedBuilder/OpenUISemanticPlanCompiler mechanism already reads RoleSlot cardinality.

## Honest caveats

- Fixture/wiring evidence only: the deterministic SLM-144 fixture corpus (`build_fixture_plan_corpus`), not a real or held-out completion corpus. No checkpoint, learned predictor, GPU run, or ship-gate claim is made or implied.
- The cardinality values injected here are a harness-local *candidate* derivation (`_derive_sibling_cardinality`), not a proposal for `OpenUISemanticPlanExtractor`'s real extraction policy. It exists only to get non-None values through the real pipeline for this consumption probe; no accuracy, calibration, or policy claim is made about it.
- This harness does not modify `OpenUISemanticPlanExtractor`, `PlanOracleSubstitutor`, `PlanSeedBuilder`, or `OpenUISemanticPlanCompiler` -- it only exercises them, matching the SLM-230 (SPV0-04) convention.
- A negative (unconsumed) result here does not mean cardinality is worthless -- it means the *current* PlanSeedBuilder mechanism (which renders one child per topology edge with no repetition/expansion logic) has no code path that would act on a cardinality bound even if one were supplied. A future PlanSeedBuilder that synthesizes repeated children from a cardinality count would need new code, not just populated data.
- Both arms here use `honesty_mode=oracle_diagnostic` and `plan_source=gold`; neither arm is promotable per the SemanticPlanV1 contract, matching SLM-230.

## Per-arm results

| arm | cardinality populated | n | seed valid rate | mean component coverage |
| --- | --- | --- | --- | --- |
| F0_no_cardinality | False | 19 | 1.000 | 1.000 |
| F1_with_cardinality | True | 19 | 1.000 | 1.000 |

## Mismatches

- (none — every record matched exactly)

## No-go for promotion

This report is wiring/fixture evidence only. It does not change `OpenUISemanticPlanExtractor`, `PlanOracleSubstitutor`, `PlanSeedBuilder`, or `OpenUISemanticPlanCompiler`, does not train or evaluate a learned head, and does not reopen SLM-145. It supplies consumer-side evidence complementing SLM-230's producer-side finding, for a human maintainer to weigh together when deciding whether cardinality extraction is worth building.

## Reproducibility

```bash
python -m scripts.run_slm231_role_cardinality_dead_field --mode plan-only
python -m scripts.run_slm231_role_cardinality_dead_field --mode fixture
```
