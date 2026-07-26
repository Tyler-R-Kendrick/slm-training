# VAR3-01 (SLM-429): real teacher behind OperatorTeacherAdapter

- **Status:** wiring only (no real LLM available)
- **Claim class:** `experiment`
- real_llm_used: `False`
- teacher_model: `gpt-5.6-sol`

## Honesty scoping (read first)

This script reuses `compare_operator_teacher_ceiling` unchanged from DSH4-02, swapping only the `teacher=` argument for `PromptedTeacherV1`. Whether that satisfies the issue's three preconditions depends entirely on `real_llm_used` above -- see `preconditions_met` in the JSON twin for the per-precondition breakdown.

## Preconditions

- `precondition_1_real_teacher_implemented`: True
- `precondition_2_real_output_re_run`: False
- `precondition_3_honest_stop_rule_read`: False

## Stop rule

- go: `False`
- reasons: ["teacher does not significantly beat required baselines on 'mrr' (paired CI lower bound <= 0): ['baseline:compiler_order', 'baseline:current_scorer', 'baseline:frequency']", 'teacher scores do not positively correlate with verifier-backed (accepted) outcomes (correlation=None)']

## Perturbation robustness

- within_bound: `True`
- kind_distances: {'candidate_order': 0.0, 'opaque_id_relabel': 0.0, 'description_length': 0.0, 'prompt_template': 0.0}

## Decision this run licenses

WIRING ONLY: no real LLM client was available in this environment (no openai package/credentials configured), so this run used the explicitly-labelled deterministic _FakeWiringClient stand-in. Precondition 1 (PromptedTeacherV1 implemented and harness-compatible) is met; preconditions 2 and 3 (a real teacher's output, and an honest stop-rule read of it) are NOT met by this run. Re-run with a provisioned OPENAI_API_KEY / installed openai package to satisfy the issue's acceptance criteria.
