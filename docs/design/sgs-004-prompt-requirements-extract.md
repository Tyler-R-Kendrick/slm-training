# SGS-004 — Conservative deterministic request→requirement extraction

**Issue:** [SLM-442](https://linear.app/quickdeploy-ai/issue/SLM-442/sgs-004-implement-conservative-deterministic-request-to-requirement)
**Status:** implemented (code + unit fixtures)

## Outcome

`extract_prompt_requirements(GenerationRequest)` in
`src/slm_training/data/semantic_plan/requirements_extract.py` emits
`PromptSemanticRequirementsV1` (SGS-003) from authored request fields only.

## Rules (stable ids)

| Rule | Source | Disposition |
| --- | --- | --- |
| `sgs004.slot_contract.v1` | `slot_contract` opaque markers | required binder refs |
| `sgs004.runtime_symbol.v1` | `runtime_symbols` | required binder/role |
| `sgs004.output_kind.v1` | non-default `output_kind` / `output_category` | required inventory |
| `sgs004.prompt_component_required.v1` | prompt prose via existing `_prompt_component_requirements` | required inventory/cardinality |
| `sgs004.prompt_component_forbidden.v1` | explicit negation around component names | forbidden inventory |
| `sgs004.prompt_either_or.v1` | explicit “either A or B” | optional + ambiguity group (no pick) |

Vague style-only prompts abstain (empty facts). Gold/target kwargs raise
`GoldTargetAccessError`. Authority is always `advisory-learned`; facts must not
shrink legal decode support (integration owned by SGS-006).

## Validation

`tests/test_data/test_semantic_plan_extraction/test_requirements_extract.py`
covers whitespace metamorphic fingerprinting, abstention, gold guards,
structured fields, inventory preservation, forbidden/either-or, and span
round-trip.
