# SDE4-03 teacher-paraphrase activation/budget manifest

- **manifest_id**: sde4-03-v1
- **schema_version**: teacher_paraphrase_activation/v1
- **hypothesis_id**: H19
- **activation_status**: blocked
- **activation_verdict**: activation_blocked
- **campaign_verdict**: unrun
- **primary_metric**: binding_aware_meaningful_program_rate
- **max_derivatives_per_root**: 5
- **manifest_hash**: fc5908312d84e467

## Activation gates

- **canonical_ast_codec_binding** (SLM-169 -> Done): ❌ not available
  - evidence: canonical AST, codec round-trip, binding integrity, lineage, split-leakage gate
- **roottype_diversity_economics** (SLM-171 -> Done): ❌ not available
  - evidence: prompt/template diversity identified as a plausible bottleneck at fixed roots
- **independent_judge_path** (SLM-106 -> Done): ❌ not available
  - evidence: cross-family judge or blinded human rubric available and disjoint from teacher generator

## Provider

- provider: unset
- model: unset
- revision: unset

## Budget cap

- max_dollars: 0.0
- max_input_tokens: 0
- max_output_tokens: 0

## Arms

- **canonical_only** (canonical_only) — eligible
- **deterministic_templates** (deterministic_templates) — eligible
- **teacher_paraphrases** (teacher_paraphrases) — eligible
  - styles: concise, detailed, business_user_story, imperative, multi_constraint
- **mixed_50_50** (mixed_50_50) — eligible
- **teacher_shuffled_target** (teacher_shuffled_target) — omitted
  - omission_reason: diagnostic control only; never an eligible training corpus beyond a bounded specificity test
- **teacher_low_diversity** (teacher_low_diversity) — eligible
  - styles: concise

## Honest caveats

This manifest is a wiring-only artifact. No teacher API calls, no model training, and no ship claim have occurred. Teacher spend must not begin until `activation_verdict` is `ready_to_spend`. The default output is intentionally blocked/budgeted to avoid accidental spend.

