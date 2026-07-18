# CAP1-05: Template-abstraction sufficiency audit

**Date:** 2026-07-18T19:02:26.204705+00:00
**Status:** wiring harness / fixture-only evidence. No training run, no checkpoint, no ship claim.

## Summary

- Records audited: 16
- Value classes in inventory: 3
- Paired variants generated: 16
- Violations (value change altered choice stream): 0

## Value-class inventory

### `string_value` (string)

- Slot representation: `STRING_SLOT`
- Retained: token_kind
- Discarded: exact_text, length, line_breaks, locale, role
- Structural decisions: property_assignment
- Pack constraints: finite_literal
- Late-realization owner: semantic_decoder
- Example fingerprints: ['06271baf49532c87', '06271baf49532c87', '82244417f956ac7c', '093e7d5fdbaacfa9', '82244417f956ac7c']

### `number_value` (number)

- Slot representation: `NUMBER_SLOT`
- Retained: token_kind
- Discarded: sign, magnitude, range_bin, unit, sentinel
- Structural decisions: property_assignment
- Pack constraints: finite_literal
- Late-realization owner: semantic_decoder
- Example fingerprints: ['5feceb66ffc86f38', 'ad57366865126e55', '6b86b273ff34fce1', 'd59eced1ded07f84']

### `boolean_value` (boolean)

- Slot representation: `BOOLEAN_SLOT`
- Retained: token_kind
- Discarded: polarity
- Structural decisions: property_assignment
- Pack constraints: finite_literal
- Late-realization owner: semantic_decoder
- Example fingerprints: ['b5bea41b6c623f7c']

## Refinement candidates

- None proposed.

## Honesty caveats

- Audit covers literal values only (strings, numbers, booleans).
- Identifier/component-reference abstraction is inherited from structural fingerprinting.
- Estimated added bits are coarse wiring placeholders, not measured state counts.

## Next steps

- Feed the violation JSON and aligned-action records into `scripts.analyze_task_quotient` and `scripts.analyze_conditional_rate` to measure before/after state and rate costs.
- Validate candidate refinements against held-out examples and update the template contract version if adopted.
