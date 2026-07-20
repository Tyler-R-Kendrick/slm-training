# SLM-174 (SDE2-07): action-alias generalization fixture (slm174-action-alias-generalization-20260720)

Matrix set: `slm174_action_alias_generalization`

Version: `sde2-07-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no trainable weights were updated, and no ship-gate claim is made.

## Hypothesis

Action descriptions remain semantically clusterable even when canonical action names are replaced by opaque aliases.

## Falsifier

Aliased descriptions collapse into a single cluster or nearest-neighbor geometry no longer reflects sibling families.

## Alias-generalization arms

| arm_id | arm_name | seed |
| --- | --- | --- |
| canonical_name_plus_description__s0 | canonical_name_plus_description | 0 |
| canonical_name_description_without_name__s0 | canonical_name_description_without_name | 0 |
| fixed_alias_description_without_name__s0 | fixed_alias_description_without_name | 0 |
| multiple_alias_augmentation_held_out__s0 | multiple_alias_augmentation_held_out | 0 |
| multiple_alias_shuffled_descriptions__s0 | multiple_alias_shuffled_descriptions | 0 |
| alias_signature_only__s0 | alias_signature_only | 0 |
| canonical_evaluated_under_unseen_alias__s0 | canonical_evaluated_under_unseen_alias | 0 |

## Results

| arm_id | arm_name | seed | n_actions | mean_nearest_cosine | family_purity | held_out_transfer | canonical_unseen | leakage | wall_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canonical_name_plus_description__s0 | canonical_name_plus_description | 0 | 79 | 0.396 | 0.127 | - | - | no | 0.019 |
| canonical_name_description_without_name__s0 | canonical_name_description_without_name | 0 | 79 | 0.549 | 0.278 | - | - | no | 0.013 |
| fixed_alias_description_without_name__s0 | fixed_alias_description_without_name | 0 | 79 | 0.430 | 0.063 | - | - | no | 0.015 |
| multiple_alias_augmentation_held_out__s0 | multiple_alias_augmentation_held_out | 0 | 79 | 0.422 | 0.089 | 0.580 | - | no | 0.085 |
| multiple_alias_shuffled_descriptions__s0 | multiple_alias_shuffled_descriptions | 0 | 79 | 0.413 | 0.051 | - | - | no | 0.034 |
| alias_signature_only__s0 | alias_signature_only | 0 | 79 | 0.430 | 0.089 | - | - | no | 0.021 |
| canonical_evaluated_under_unseen_alias__s0 | canonical_evaluated_under_unseen_alias | 0 | 79 | 0.422 | 0.089 | - | 0.570 | no | 0.089 |

## Per-arm means

| arm_name | mean_nearest_cosine | family_purity | held_out_transfer | canonical_unseen |
| --- | --- | --- | --- | --- |
| canonical_name_plus_description | 0.396 | 0.127 | - | - |
| canonical_name_description_without_name | 0.549 | 0.278 | - | - |
| fixed_alias_description_without_name | 0.430 | 0.063 | - | - |
| multiple_alias_augmentation_held_out | 0.422 | 0.089 | 0.580 | - |
| multiple_alias_shuffled_descriptions | 0.413 | 0.051 | - | - |
| alias_signature_only | 0.430 | 0.089 | - | - |
| canonical_evaluated_under_unseen_alias | 0.422 | 0.089 | - | 0.570 |

## Disposition

**baseline_unreliable**

Canonical-name-plus-description baseline does not cluster by family; the fixture encoder or catalog is not representative.

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The alias map, description sources, and clustering telemetry are exercised over a deterministic synthetic encoder, but no real model was trained or evaluated. The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` until a trained model and AgentV evaluation are available.

## Honest caveats

- The FixtureDescriptionEncoder is a deterministic hash surrogate, not a trained   language model; geometry may differ with real text encoders.
- Sibling-family grouping is a coarse semantic proxy.
- No ship-gate claim is made; this is wiring evidence only.

## Reproducibility

```bash
python -m scripts.run_slm174_action_alias_generalization_fixture --mode plan-only
python -m scripts.run_slm174_action_alias_generalization_fixture --mode fixture
```
