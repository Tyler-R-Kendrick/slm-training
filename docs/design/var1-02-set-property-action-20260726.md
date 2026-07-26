# SLM-299 (LAR1-03): X22 edit-space reachability audit

- generated_at: `2026-07-26T19:32:41Z`
- seed: `root = Stack([], "column")`
- mode: `extended`
- max_edits: 8, node_budget: 15
- verdict policy: reachable_fraction is computed over decided cases only; UNKNOWN_BUDGET cases are reported separately and are never counted as unreachable; suites without a corpus are corpus_unavailable, never zero-reachable. Reachability is a space-coverage proof, never a model-quality claim.

> Reachability is space coverage, not model quality: no quality claim
> follows from these proofs alone.

## SLM-425 production delta

`ACTION_SET_PROPERTY` is now a real format-3 action. It mutates only the
target container's finite, pack-owned `rest` domain and fragment-validates the
structured rebuilt statement. The broader VAR1-01 `row` control remains
hypothetical; it is not production authority.

| suite | SLM-305 baseline | VAR1-01 arm B | SLM-425 production | delta / caveat |
| --- | --- | --- | --- | --- |
| train | 0/6 | unavailable | unavailable | Current frozen train corpus is unavailable; no comparison claimed. |
| smoke | 0/3 | 0/3 | 0/3 | No verdict flips. |
| held_out | 0/5 | 0/5 | 0/5 | No verdict flips. |
| adversarial | 0/4 | 1/2 decided (0.5) | 1/3 decided (0.333333) | `adv_empty_prompt_01` flips `needs_direction_change` → reachable in 2 edits (`ADD`, `SET_PROPERTY`); production is at or below arm B. |
| ood | 0/4 | 0/4 | 0/4 | No verdict flips. |
| rico | 0/6 historical | 35 unknown | 0/1 decided; 34 unknown | No favorable comparison: historical corpus size differs and UNKNOWN_BUDGET is not unreachability evidence. |

This bounded local run used `max_edits=8`, `node_budget=15`, and the full
currently available smoke/held_out/adversarial/ood/rico inputs. It is a
capability-space coverage result only: no quality, ship, promotion, checkpoint,
or meaningful-parse claim follows from it.

## Suite summary

| suite | n | decided | unknown | reachable_fraction | min/med/max edits |
| --- | --- | --- | --- | --- | --- |
| train | corpus_unavailable | — | — | — | — |
| smoke | 3 | 3 | 0 | 0.0 | — |
| held_out | 5 | 5 | 0 | 0.0 | — |
| adversarial | 4 | 3 | 1 | 0.333333 | 2/2/2 |
| ood | 4 | 4 | 0 | 0.0 | — |
| rico | 35 | 1 | 34 | 0.0 | — |

## Reason-code histograms

- **smoke**: `{"unsupported_component": 3}`
- **held_out**: `{"unsupported_component": 5}`
- **adversarial**: `{"budget": 1, "needs_direction_change": 1, "reached": 1, "unsupported_component": 1}`
- **ood**: `{"unsupported_component": 4}`
- **rico**: `{"budget": 34, "needs_direction_change": 1}`

## Action / component coverage (reachable paths)

- **smoke**: actions `{}` components `{}`
- **held_out**: actions `{}` components `{}`
- **adversarial**: actions `{"ADD": 1, "SET_PROPERTY": 1}` components `{"TextContent": 1}`
- **ood**: actions `{}` components `{}`
- **rico**: actions `{}` components `{}`

## Per-case verdicts

- **train**: corpus unavailable
- **smoke**:
  - `smoke_hero_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `smoke_button_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `smoke_callout_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
- **held_out**:
  - `held_out_form_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `held_out_dual_card_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `held_out_input_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `held_out_tabs_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `held_out_settings_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
- **adversarial**:
  - `adv_empty_prompt_01` PROVEN_REACHABLE (reached, lower_bound=2)
  - `adv_dual_card_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `adv_deep_nest_01` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `adv_many_buttons_01` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
- **ood**:
  - `ood_dashboard_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `ood_gallery_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `ood_modal_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `ood_auth_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
- **rico**:
  - `rico_eval_test_0` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_1` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_2` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_4` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_8` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_9` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_12` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_17` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_20` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_25` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_34` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_35` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_38` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_40` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_41` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_42` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_47` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_48` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_51` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_53` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_55` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_56` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_57` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_58` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_59` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_60` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_68` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_69` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_77` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_81` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_91` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_95` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_97` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_99` UNKNOWN_BUDGET (budget, lower_bound=None)
  - `rico_eval_test_104` UNKNOWN_BUDGET (budget, lower_bound=None)

## X22 evidence annotations (append-only)

- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [adversarial]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.333333 over 3 decided adversarial cases (1 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on adversarial in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [adversarial]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.333333 over 3 decided adversarial cases (1 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on adversarial in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [held_out]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 5 decided held_out cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on held_out in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [held_out]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 5 decided held_out cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on held_out in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [ood]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 4 decided ood cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on ood in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [ood]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 4 decided ood cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on ood in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [rico]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 1 decided rico cases (34 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on rico in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [rico]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 1 decided rico cases (34 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on rico in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [smoke]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 3 decided smoke cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on smoke in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [smoke]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 3 decided smoke cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on smoke in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
