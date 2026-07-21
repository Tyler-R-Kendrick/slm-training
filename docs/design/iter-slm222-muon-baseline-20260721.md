# SLM-222 (NCS2-03): Muon/AdamW hybrid baseline fixture (slm222-muon-baseline-20260721)

**Matrix set:** `slm222_muon_baseline`
**Version:** `ncs2-03-v1`
**Status:** fixture
**Claim class:** wiring

## Honest caveats

- Fixture/wiring evidence only: no trained model, checkpoint promotion, GPU run, or ship-gate claim.
- The full O0-O4 matched AdamW-vs-Muon campaign (capacity- and data-matched, with spectral LR control) requires local E224+ checkpoints and dedicated GPU time and is documented as future work.
- The fixture uses a tiny scratch-context model and a single synthetic record; no meaningful-parse or generalization conclusion can be drawn.
- AdamW and Muon arms start from the same random seed but the comparison is limited to optimizer wiring, not convergence, final loss, or downstream eval metrics.

## Summary

- Hybrid partition valid: True
- Muon group parameters: 28672
- AdamW group parameters: 19106

## AdamW arm

- run_id: `slm222-muon-baseline-adamw-0`
- steps_completed: 2
- last_loss: 29.259197235107422
- finite_parameters: True
- optimizer_fingerprint: `{}`

## Muon arm

- run_id: `slm222-muon-baseline-muon-0`
- steps_completed: 2
- last_loss: 29.590923309326172
- finite_parameters: True
- optimizer_fingerprint: `{'optimizer': 'muon_hybrid', 'muon_ns_steps': 5, 'muon_nesterov': False}`

## Recipe fields (Muon arm)

```json
{
  "adamw_lr": 0.0003,
  "batch_size": 1,
  "binder_arity_decode_weight": null,
  "binder_arity_loss_weight": 0.0,
  "binder_component_plan_decode_weight": null,
  "binder_component_plan_loss_weight": 0.0,
  "binder_topology_decode_weight": null,
  "binder_topology_loss_weight": 0.0,
  "compiler_alignment_loss_weight": 0.0,
  "compiler_alignment_margin": 0.0,
  "compiler_alignment_semantic_exhaustive": false,
  "compiler_alignment_stratified": false,
  "component_edge_alignment_loss_weight": 0.0,
  "component_edge_decode_weight": null,
  "component_edge_loss_weight": 0.0,
  "component_inventory_decode_weight": null,
  "component_inventory_loss_weight": 0.0,
  "component_plan_decode_weight": null,
  "component_plan_loss_weight": 0.0,
  "design_md_dropout": 0.0,
  "fastpath_aux_weight": 0.0,
  "fidelity_loss_weight": 0.5,
  "fuse_ltr_loss": true,
  "grammar_constrained": false,
  "honest_slot_contract": false,
  "honesty_mode": "design-md-context",
  "initialization_weight_retention": 0.0,
  "learning_rate": 0.0003,
  "ltr_loss_weight": 0.5,
  "muon_lr": 0.0003,
  "muon_momentum": 0.9,
  "muon_nesterov": false,
  "muon_ns_steps": 5,
  "optimizer_name": "muon_hybrid",
  "replay_fraction": 0.0,
  "retrieval_k": 0,
  "root_reference_arity_decode_weight": null,
  "root_reference_arity_loss_weight": 0.0,
  "root_reference_identity_decode_weight": null,
  "root_reference_identity_loss_weight": 0.0,
  "root_reference_identity_negative_weight": 1.0,
  "root_reference_identity_sampling_records": 1,
  "root_reference_identity_strict_subset_multiplier": 1,
  "root_reference_identity_strict_subset_records": 0,
  "schema_in_context": false,
  "seed": 0,
  "slot_component_class_balance_power": 0.0,
  "slot_component_decode_weight": null,
  "slot_component_lexeme_prior_weight": 0.0,
  "slot_component_loss_weight": 0.0,
  "slot_component_next_context": false,
  "slot_component_owner_counts": {},
  "slot_component_owner_rare_classes": [],
  "slot_component_owner_rare_multiplier": 1,
  "slot_component_owner_rare_records": 0,
  "slot_component_owner_rare_threshold": 0,
  "slot_component_owner_sampling_records": 1,
  "slot_component_pair_interaction": false,
  "slot_component_prompt_context": true,
  "slot_component_span_prior_weight": 0.0,
  "slot_contract_in_context": false,
  "steps_requested": 2,
  "weight_decay": 0.0
}
```

## No-go for promotion

This report is wiring/fixture evidence only. No checkpoint, GPU train, or ship gate is claimed.

