# VAR0-03 (SLM-426): pack-derived tree-edit inventory

**Claim class:** refactor. `TreeEditSpace` and the SLM-299 analyzer now resolve
their alphabet from the canonical `DslPack`; this is not a model-quality,
training, checkpoint, or ship result.

## Pack coverage and replay disposition

The OpenUI pack retains the prior seven-component tree-edit inventory
(`TextContent`, `Button`, `Image`, `TextInput`, `Stack`, `Card`, `Form`) and
the two pre-existing container rest forms. The current suite references
additional names (`dashboard`, `gallery`, `modal`, `tabs`, `settings`, and
`dual_card`); those remain outside the pack and are reported below as
`unsupported_component`, not silently added. `toy-layout` supplies its own
`text`/`button` leaves and `row`/`col` containers; the arithmetic pack has no
tree-edit component surface.

The frozen 2026-07-24 report cannot be replayed bit-for-bit from today's
inputs: its train artifact is unavailable and RICO grew from 6 to 35 records.
On the shared smoke/held-out/ood cases the refactor preserves the 0.0
reachable fraction; the changed RICO denominator and the adversarial budget
case are current-corpus observations, not a regression claim.

## Canonical bounded audit

- generated_at: `2026-07-26T19:13:06Z`
- seed: `root = Stack([], "column")`
- mode: `extended`
- max_edits: 8, node_budget: 120
- verdict policy: reachable_fraction is computed over decided cases only; UNKNOWN_BUDGET cases are reported separately and are never counted as unreachable; suites without a corpus are corpus_unavailable, never zero-reachable. Reachability is a space-coverage proof, never a model-quality claim.

> Reachability is space coverage, not model quality: no quality claim
> follows from these proofs alone.

## Suite summary

| suite | n | decided | unknown | reachable_fraction | min/med/max edits |
| --- | --- | --- | --- | --- | --- |
| train | corpus_unavailable | — | — | — | — |
| smoke | 3 | 3 | 0 | 0.0 | — |
| held_out | 5 | 5 | 0 | 0.0 | — |
| adversarial | 4 | 3 | 1 | 0.0 | — |
| ood | 4 | 4 | 0 | 0.0 | — |
| rico | 35 | 35 | 0 | 0.0 | — |

## Reason-code histograms

- **smoke**: `{"unsupported_component": 3}`
- **held_out**: `{"unsupported_component": 5}`
- **adversarial**: `{"budget": 1, "needs_direction_change": 2, "unsupported_component": 1}`
- **ood**: `{"unsupported_component": 4}`
- **rico**: `{"needs_direction_change": 35}`

## Action / component coverage (reachable paths)

- **smoke**: actions `{}` components `{}`
- **held_out**: actions `{}` components `{}`
- **adversarial**: actions `{}` components `{}`
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
  - `adv_empty_prompt_01` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `adv_dual_card_01` PROVEN_UNREACHABLE (unsupported_component, lower_bound=None)
  - `adv_deep_nest_01` UNKNOWN_BUDGET (budget, lower_bound=None)
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
  - `rico_eval_test_12` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_17` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_20` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_25` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_34` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_35` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_38` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_40` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_41` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_42` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_47` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_48` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_51` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_53` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_55` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_56` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_57` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_58` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_59` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_60` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_68` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_69` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_77` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_81` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_91` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_95` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_97` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_99` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)
  - `rico_eval_test_104` PROVEN_UNREACHABLE (needs_direction_change, lower_bound=None)

## X22 evidence annotations (append-only)

- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [adversarial]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 3 decided adversarial cases (1 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on adversarial in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [adversarial]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 3 decided adversarial cases (1 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on adversarial in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [held_out]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 5 decided held_out cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on held_out in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [held_out]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 5 decided held_out cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on held_out in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [ood]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 4 decided ood cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on ood in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [ood]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 4 decided ood cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on ood in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [rico]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 35 decided rico cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on rico in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [rico]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 35 decided rico cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on rico in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-x22-d3-kapur-tree-edit-20260717.md` [smoke]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 3 decided smoke cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on smoke in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
- `docs/design/iter-efs0-04-x22-reproduction-20260717.md` [smoke]: SLM-299 edit-reachability audit: from the standard X22 minimal seed, reachable_fraction=0.0 over 3 decided smoke cases (0 UNKNOWN_BUDGET, never counted unreachable). Suite-level quality readings of the X22 tree-edit decode on smoke in this document are bounded by that fraction: the unreachable share of gold programs cannot be produced by the decode space at all, so measured quality on those cases reflects space coverage, not model quality.
