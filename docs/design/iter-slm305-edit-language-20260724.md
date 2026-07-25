# SLM-299 (LAR1-03): X22 edit-space reachability audit

- generated_at: `2026-07-24T20:22:57Z`
- seed: `root = Stack([], "column")`
- mode: `extended`
- max_edits: 8, node_budget: 120
- verdict policy: reachable_fraction is computed over decided cases only; UNKNOWN_BUDGET cases are reported separately and are never counted as unreachable; suites without a corpus are corpus_unavailable, never zero-reachable. Reachability is a space-coverage proof, never a model-quality claim.

> Reachability is space coverage, not model quality: no quality claim
> follows from these proofs alone.

## Suite summary

| suite | n | decided | unknown | reachable_fraction | min/med/max edits |
| --- | --- | --- | --- | --- | --- |
| train | 6 | 6 | 0 | 0.0 | — |
| smoke | 3 | 3 | 0 | 0.0 | — |
| held_out | 5 | 5 | 0 | 0.0 | — |
| adversarial | 4 | 4 | 0 | 0.0 | — |
| ood | 4 | 4 | 0 | 0.0 | — |
| rico | 6 | 6 | 0 | 0.0 | — |

## Reason-code histograms

- **train**: `{"needs_direction_change": 3, "unsupported_component": 3}`
- **smoke**: `{"unsupported_component": 3}`
- **held_out**: `{"unsupported_component": 5}`
- **adversarial**: `{"needs_direction_change": 3, "unsupported_component": 1}`
- **ood**: `{"unsupported_component": 4}`
- **rico**: `{"needs_direction_change": 6}`

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
  - `adv_empty_prompt_01` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `adv_dual_card_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `adv_deep_nest_01` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `adv_many_buttons_01` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
- **ood**:
  - `ood_dashboard_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `ood_gallery_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `ood_modal_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `ood_auth_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
- **rico**:
  - `rico_eval_test_0` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_1` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_2` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_4` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_8` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_9` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)

## X22 evidence annotations (append-only)

- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [adversarial]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 4 decided adversarial cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on adversarial in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [adversarial]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 4 decided adversarial cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on adversarial in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [held_out]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 5 decided held_out cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on held_out in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [held_out]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 5 decided held_out cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on held_out in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [ood]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 4 decided ood cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on ood in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [ood]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 4 decided ood cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on ood in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [rico]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 6 decided rico cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on rico in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [rico]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 6 decided rico cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on rico in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [smoke]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 3 decided smoke cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on smoke in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [smoke]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 3 decided smoke cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on smoke in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [train]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 6 decided train cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on train in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [train]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 6 decided train cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on train in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.

## Old (v1) vs extended (SLM-305) reachability

| suite | reachable v1 | reachable extended | verdict flips | action-cost delta min/med/max |
| --- | --- | --- | --- | --- |
| train | 0.0 | 0.0 | 0 | — |
| smoke | 0.0 | 0.0 | 0 | — |
| held_out | 0.0 | 0.0 | 0 | — |
| adversarial | 0.0 | 0.0 | 0 | — |
| ood | 0.0 | 0.0 | 0 | — |
| rico | 0.0 | 0.0 | 0 | — |
