# SLM-299 (LAR1-03): X22 edit-space reachability audit

- generated_at: `2026-07-24T15:22:54Z`
- seed: `root = Stack([], "column")`
- max_edits: 8, node_budget: 800
- verdict policy: reachable_fraction is computed over decided cases only; UNKNOWN_BUDGET cases are reported separately and are never counted as unreachable; suites without a corpus are corpus_unavailable, never zero-reachable.

## Suite summary

| suite | n | decided | unknown | reachable_fraction | min/med/max edits |
| --- | --- | --- | --- | --- | --- |
| train | 97 | 97 | 0 | 0.0 | — |
| smoke | 3 | 3 | 0 | 0.0 | — |
| held_out | 5 | 5 | 0 | 0.0 | — |
| adversarial | 4 | 4 | 0 | 0.0 | — |
| ood | 4 | 4 | 0 | 0.0 | — |
| rico | 35 | 35 | 0 | 0.0 | — |

## Reason-code histograms

- **train**: `{"needs_container_add": 49, "needs_direction_change": 15, "unsupported_component": 33}`
- **smoke**: `{"needs_container_add": 2, "unsupported_component": 1}`
- **held_out**: `{"needs_container_add": 3, "unsupported_component": 2}`
- **adversarial**: `{"needs_container_add": 2, "needs_direction_change": 2}`
- **ood**: `{"needs_container_add": 2, "unsupported_component": 2}`
- **rico**: `{"needs_container_add": 35}`

## Action / component coverage (reachable paths)

- **train**: actions `{}` components `{}`
- **smoke**: actions `{}` components `{}`
- **held_out**: actions `{}` components `{}`
- **adversarial**: actions `{}` components `{}`
- **ood**: actions `{}` components `{}`
- **rico**: actions `{}` components `{}`

## Per-case verdicts

- **train**:
  - `train_auth_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_auth_01_aug_dir` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_auth_01_syn_2` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_button_row_01` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `train_button_row_01_aug_dir` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `train_button_row_01_syn_0` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `train_button_row_01_syn_1` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `train_button_row_01_syn_2` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `train_callout_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_callout_01_aug_cta` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_callout_01_aug_dir` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_callout_01_syn_0` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_callout_01_syn_1` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_callout_01_syn_2` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_card_header_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_card_header_01_aug_cta` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_card_header_01_aug_dir` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_card_header_01_syn_0` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_card_header_01_syn_1` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_card_header_01_syn_2` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_card_stack_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_card_stack_01_aug_dir` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_card_stack_01_syn_0` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_card_stack_01_syn_1` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_cta_01` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `train_cta_01_aug_dir` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `train_cta_01_syn_0` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `train_cta_01_syn_1` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `train_cta_01_syn_2` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `train_form_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_form_01_aug_dir` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_form_01_syn_0` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_form_control_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_form_control_01_aug_dir` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_form_control_01_syn_0` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_form_control_01_syn_2` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_gallery_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_gallery_01_aug_dir` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_gallery_01_syn_0` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_gallery_01_syn_1` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_gallery_01_syn_2` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_header_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_header_01_aug_cta` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_header_01_aug_dir` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_header_01_syn_0` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_header_01_syn_1` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_header_01_syn_2` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_hero_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_hero_01_aug_cta` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_hero_01_aug_dir` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_hero_01_fd_0` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_hero_01_fd_1` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_hero_01_fd_2` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_image_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_image_01_aug_cta` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_image_01_aug_dir` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_image_01_syn_0` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_image_01_syn_1` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_image_01_syn_2` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_metrics_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_metrics_01_aug_dir` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_metrics_01_syn_0` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_metrics_01_syn_1` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_metrics_01_syn_2` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_modal_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_modal_01_aug_cta` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_modal_01_aug_dir` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_modal_01_syn_0` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_modal_01_syn_1` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_modal_01_syn_2` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_nested_01` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `train_nested_01_aug_dir` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `train_nested_01_syn_0` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `train_nested_01_syn_1` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `train_nested_01_syn_2` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `train_pricing_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_pricing_01_aug_dir` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_pricing_01_syn_0` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_pricing_01_syn_1` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_pricing_01_syn_2` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_separator_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_separator_01_aug_cta` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_separator_01_aug_dir` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_separator_01_syn_0` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_separator_01_syn_1` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_separator_01_syn_2` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_settings_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_settings_01_aug_cta` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_settings_01_aug_dir` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_settings_01_syn_0` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_settings_01_syn_1` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_settings_01_syn_2` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `train_tabs_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_tabs_01_aug_cta` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_tabs_01_aug_dir` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_tabs_01_syn_0` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `train_tabs_01_syn_2` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
- **smoke**:
  - `smoke_hero_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `smoke_button_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `smoke_callout_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
- **held_out**:
  - `held_out_form_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `held_out_dual_card_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `held_out_input_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `held_out_tabs_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `held_out_settings_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
- **adversarial**:
  - `adv_empty_prompt_01` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `adv_dual_card_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `adv_deep_nest_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `adv_many_buttons_01` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
- **ood**:
  - `ood_dashboard_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `ood_gallery_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `ood_modal_01` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `ood_auth_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
- **rico**:
  - `rico_eval_test_0` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_1` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_2` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_4` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_8` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_9` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_12` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_17` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_20` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_25` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_34` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_35` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_38` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_40` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_41` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_42` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_47` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_48` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_51` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_53` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_55` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_56` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_57` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_58` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_59` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_60` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_68` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_69` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_77` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_81` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_91` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_95` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_97` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_99` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)
  - `rico_eval_test_104` PROVEN_UNREACHABLE (needs_container_add, lower_bound=None)

## X22 evidence annotations (append-only)

- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [adversarial]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 4 decided adversarial cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on adversarial in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [adversarial]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 4 decided adversarial cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on adversarial in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [held_out]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 5 decided held_out cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on held_out in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [held_out]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 5 decided held_out cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on held_out in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [ood]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 4 decided ood cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on ood in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [ood]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 4 decided ood cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on ood in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [rico]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 35 decided rico cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on rico in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [rico]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 35 decided rico cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on rico in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [smoke]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 3 decided smoke cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on smoke in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [smoke]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 3 decided smoke cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on smoke in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [train]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 97 decided train cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on train in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [train]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 97 decided train cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on train in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
