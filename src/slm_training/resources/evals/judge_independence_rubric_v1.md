# OpenUI independent pair rubric v1

You are comparing two anonymized OpenUI programs for one user prompt. Do not
infer or guess their model, checkpoint, training source, or prior automatic
scores. Judge only the prompt and the two programs shown.

For each candidate, determine whether it independently satisfies the requested
semantic contract. Check all of the following:

1. the root and component inventory implement the requested UI roles;
2. declared bindings are defined, reachable, and used in valid component roles;
3. placeholders cover the requested user-facing slots without unrelated or
   silently substituted content;
4. required children and arguments are populated rather than merely syntactic;
5. the structure is a plausible layout for the prompt.

Prefer the candidate with better semantic contract satisfaction. Use `tie` only
when both are equivalently acceptable or equivalently unacceptable. Do not
reward syntax alone. Do not use formatting, identifier spelling, or verbosity
as a quality signal.

Return exactly one JSON object matching `JudgePairResponseV1`:

```json
{
  "verdict": "left|right|tie|refusal",
  "left_acceptable": true,
  "right_acceptable": false,
  "score": 0.0,
  "confidence": 0.0,
  "reason_codes": ["component_inventory"],
  "cost_usd": 0.0
}
```

`score` is preference strength from 0 (no preference) to 1 (decisive). Allowed
reason codes are `component_inventory`, `binding_failure`,
`placeholder_failure`, `schema_role_failure`, `empty_or_trivial`,
`structural_mismatch`, `multiple_valid_modes`, and `ambiguous_prompt`.
