# SLM-482 (SIE-004): Prompt-requirement predictor campaign (EXP-SR-2)

**Claim class:** `fixture` only (catalogue identity claim_class=`screening`; execution is fixture — no promotion)

**Catalogue:** `exp-sr-2`

**Primary metric (`requirement_predictor_f1`, learned arm mean):** 0.0

**Effect vs frequency:** `-0.448389` (minimum_effect=`0.05`)

**Ceiling recovery vs gold oracle:** `0.0`

## Acceptance snapshot

| Check | Value |
| --- | --- |
| legal_support_parity_exact | True |
| falsifier_holds | True |
| clears_frequency_baseline | False |
| clears_none_baseline | False |
| dominated_by_simpler_control | True |
| paired Δ vs frequency (mean ± SE) | -0.448389 ± 0.032079 (n=3) |
| SIE-002 authorized_factors | `['roles', 'bindings']` |
| mapped program families | `['binder_reference', 'property_role_value']` |
| seeds | [0, 1, 2] |
| promotion | False |

## Arms (mean F1 across seeds)

| Arm | requirement_predictor_f1_mean | per-seed |
| --- | ---: | --- |
| none | 0.0 | [0.0, 0.0, 0.0] |
| deterministic_extractor | 0.0 | [0.0, 0.0, 0.0] |
| frequency | 0.448389 | [0.508197, 0.398374, 0.438596] |
| retrieval | 0.637172 | [0.590164, 0.707317, 0.614035] |
| learned_predictor | 0.0 | [0.0, 0.0, 0.0] |
| gold_oracle | 1.0 | [1.0, 1.0, 1.0] |

## Falsifier / authorization notes

- Catalogue falsifier: predictor F1 does not exceed the majority-class/no-prediction baseline on the held-out prompt slice.
- SIE-002 auth note: SIE-002 authorized ['roles', 'bindings']; mapped program families ['binder_reference', 'property_role_value']. SIE-003 seam targets 'style_layout' (NOT authorized).

## Legal support parity

- exact: `True`
- Advisory RequirementFacts never unlock hard prune; byte/set legal support unchanged with predictor on vs off (I6).

## Scope

Fixture-scale EXP-SR-2 prompt-requirement predictor campaign over openui_hard_valid_v1 positive-role rows. Arms reuse SIE-003 predictor primitives; retrieval is train-only 1-NN (leakage-safe). Primary metric is requirement_predictor_f1 on archetype/style_layout labels. SIE-002 authorized_factors=['roles', 'bindings'] (program families ['binder_reference', 'property_role_value']); the seam target 'style_layout' is not in that set, so advisory fact emission under the SIE-002 gate is zero. Legal support parity is exact (advisory facts never hard-prune). claim_class=fixture; no promotion.

Command: `python -m scripts.run_sie004_predictor_campaign --mode fixture`

Full detail: `docs/design/iter-slm482-sie-004-predictor-campaign-20260810.json`.
