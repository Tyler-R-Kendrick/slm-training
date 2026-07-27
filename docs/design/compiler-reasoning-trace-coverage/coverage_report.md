# CompilerReasoningTraceV1 bounded K/c coverage probe (LOT0-02 / SLM-249)

**Evidence class: `bounded_probe`.** Sample size n=16 (`src/slm_training/resources/test_seeds.jsonl`), not a production corpus statistic. See the LOT0-01 authorization (`docs/design/lotus-openui-fidelity-contract-v1.md`) for why corpus-scale K/c derivation is out of scope for this issue.

- proposed K: **6** (fixed by stage-set construction)
- proposed c (provisional, chars): **479**
- fraction of probe records requiring all 6 stages: 1.000
- fraction truncated: 0.000

## Truncation/fallback policy

fixed K=6 by stage-set construction (see compiler_reasoning_trace.py module docstring); c is provisional (max per-stage p95 char length over this bounded probe) and any step whose canonical_target would exceed c must set is_truncated=True and a non-null ambiguity_reason rather than silently drop content -- no truncation occurred in this probe (fraction_truncated=0.0 over n=16)

## Per-stage canonical_target length (chars)

| stage | n | min | p50 | p95 | max |
| --- | --- | --- | --- | --- | --- |
| intent_contract | 16 | 5 | 12.0 | 12.0 | 12 |
| component_inventory | 16 | 56 | 118.0 | 313.0 | 313 |
| topology_skeleton | 16 | 27 | 79.0 | 235.0 | 235 |
| semantic_edges_roles | 16 | 68 | 179.0 | 479.0 | 479 |
| binding_scope_references | 16 | 28 | 176.0 | 266.5 | 283 |
| serialization_plan | 16 | 19 | 41.0 | 85.8 | 97 |

Sample record IDs: smoke_hero_01, smoke_button_01, smoke_callout_01, held_out_form_01, held_out_dual_card_01, held_out_input_01, held_out_tabs_01, held_out_settings_01, adv_empty_prompt_01, adv_dual_card_01, adv_deep_nest_01, adv_many_buttons_01, ood_dashboard_01, ood_gallery_01, ood_modal_01, ood_auth_01
