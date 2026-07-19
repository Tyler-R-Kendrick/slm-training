# SDE3-03 proxy-metric calibration activation/budget manifest

- **manifest_id**: sde3-03-v1
- **schema_version**: proxy_metric_calibration/v1
- **hypothesis_id**: H23
- **activation_status**: ready
- **activation_verdict**: ready_to_spend
- **campaign_verdict**: unrun
- **primary_metric**: binding_aware_meaningful_program_rate
- **proxy_eval_mode**: shadow
- **conservative_floor**: 0.7
- **risk_budget**: 0.05
- **manifest_hash**: c6c96a00b741972f

## Activation gates

- **slm105_binding_aware_metrics** (SLM-105 -> Done): ✅ available
  - evidence: Binding-aware deterministic metrics stable and versioned.
- **slm169_canonical_ast_binding** (SLM-169 -> Done): ✅ available
  - evidence: Canonical AST, codec round-trip, binding integrity gates.
- **slm175_eval_cache** (SLM-175 -> Done): ✅ available
  - evidence: Content-addressed evaluation cache artifacts available.
- **proxy_feature_contract_reviewed** (SLM-177 -> Done): ✅ available
  - evidence: Feature contract reviewed: no forbidden features included.
- **budget_approved** (SLM-177 -> approved): ✅ available
  - evidence: Calibration budget approved.

## Feature contract

- schema_version: proxy_features/v1
- target_primary: binding_aware_meaningful_program_rate
- target_gate: full_gate_pass
- features: parser_valid, schema_valid, binding_aware_meaningful_rate, component_recall, role_recall, minimality_flag, empty_output_flag, first_attempt_action_count, legal_action_margin, entropy, termination_confidence, ast_node_count, binding_graph_edges, latency_ms, output_length, tree_depth, component_count

## Budget cap

- max_historical_rows: 10000
- max_dollars: 0.0
- gpu_hours: 0.0
- eval_dollars: 0.0
- total_dollars: 500.0

## Arms

- **rule_baseline** (rule_baseline) — eligible
- **regularized_linear** (regularized_linear) — eligible
- **bounded_tree** (bounded_tree) — omitted
  - omission_reason: reserved for ablation if linear models are insufficient
- **shadow_only** (shadow_only) — eligible

## Honest caveats

This manifest is a wiring-only artifact. No proxy model has been trained, no full-suite invocation behavior has changed, and no promotion or ship claim has occurred. The default `proxy_eval_mode` is `off`; triage mode must not be enabled until the activation criteria are met.

