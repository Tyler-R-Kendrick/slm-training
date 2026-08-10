# SLM-476 (SIE-002): Factor-wise semantic oracle localization (EXP-SR-1)

**Claim class:** `fixture` only

**Catalogue:** `exp-sr-1`

**Primary metric (`oracle_factor_localization_precision`):** 1.0

**Authorized factors for later learned-predictor work:** `['roles', 'bindings']`

## Acceptance snapshot

| Check | Value |
| --- | --- |
| leakage_audit_passed | True |
| no_op_control_holds | True |
| declared_factor_fidelity_rate | 1.0 |
| n_active_baselines (seed-aggregated) | 15 |
| seeds | [0, 1, 2] |
| promotion | False |

## Per-factor

| Factor | change_rate | ceiling_recovery_rate | fidelity | authorized |
| --- | ---: | ---: | ---: | --- |
| archetype | 0.0 | 0.0 | 1.0 | no |
| roles | 0.466667 | 1.0 | 1.0 | yes |
| topology | 0.0 | 0.0 | 1.0 | no |
| bindings | 0.266667 | 1.0 | 1.0 | yes |

## Authorization notes

- falsifier_holds: `False`
- One or more factors cleared the fixture authorization bar; still claim_class=fixture — not a promotion.

Per-factor reasons:

- `archetype`: one_factor_change_rate=0.0 < minimum_effect=0.03
- `roles`: authorized_fixture_scale_oracle_localization
- `topology`: one_factor_change_rate=0.0 < minimum_effect=0.03
- `bindings`: authorized_fixture_scale_oracle_localization

## Scope

Fixture-scale factor-wise oracle localization over a VCE-008-cleaned openui_hard_valid_v1 pool. Arms generated exclusively via VCE-009's generate_baseline_arms (no fork). No model, verifier search, or GPU cost is measured — plan-hash / changed-factor proxies only. claim_class=fixture; authorized_factors gate later predictor work and never promote a checkpoint.

Command: `python -m scripts.run_sie002_oracle_localization --mode fixture`

Full detail: `docs/design/iter-slm476-sie-002-oracle-localization-20260810.json`.
