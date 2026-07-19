# pretrained_denoiser_activation/v1

**Hypothesis:** H18
**Activation status:** blocked
**Activation verdict:** activation_blocked
**Campaign verdict:** unrun
**Primary metric:** binding_aware_meaningful_program_rate
**Manifest hash:** `b615fe24a949ddfa`

## Candidate

- **ID:** unset
- **Provider:** unset
- **Repository:** 
- **Model:** unset
- **Revision:** main
- **Architecture:** unset
- **Parameters:** 0
- **License:** unset
- **Local offline available:** False

## Budget

- **Model acquisition:** None
- **GPU hours:** None
- **Storage:** None
- **Conversion:** None
- **Eval:** None
- **Total:** 0.0

## Activation gates

| Gate | Depends on | Required | Available |
|------|------------|----------|-----------|
| slm161_data_contract_closed | SLM-161 | closed | False |
| slm24_evaluation_ready | SLM-24 | closed | False |
| slm175_connector_spec_closed | SLM-175 | closed | False |
| small_baseline_stable | SDE4-04 | passed | False |
| budget_approved | SDE4-04 | approved | False |
| license_compatible | SDE4-04 | approved | False |

## Arms

- **current_small_controller_baseline** (`current_small_controller_baseline`) — eligible
- **b4_pilot_reference** (`b4_pilot_reference`) — eligible
- **pretrained_denoiser_plus_adapters** (`pretrained_denoiser_plus_adapters`) — eligible
- **frozen_backbone_connector_only** (`frozen_backbone_connector_only`) — eligible
- **adapters_disabled_diagnostic** (`adapters_disabled_diagnostic`) — omitted (zero-training diagnostic only)
- **equal_compute_small_controller_control** (`equal_compute_small_controller_control`) — eligible
- **random_init_short_budget_diagnostic** (`random_init_short_budget_diagnostic`) — omitted (optional short-budget diagnostic only when financially justified)

## Note

SDE4-04 pretrained-denoiser activation manifest (wiring slice).
