# SLM-313 AbstractPlan functional evidence

This locked functional-evidence preflight makes no model or promotion claim.

- Verdict: unavailable; promotion eligible: false.
- Reason: No trained learned-plan checkpoint was supplied; AP-023's side-channel head and AP-024's zero-gate connector are not evidence of learned plan function.
- Locked manifest: b4ad49cf1b73ad50528709daaad53dbf4846036c9dea787f1c2017c16e0a2d48.
- Meaningful-parse and meaning-v2 were not measured.
- AgentEvals/AgentV records the fail-closed non-promotion assertion.

## Local training attempts (non-evidence)

On 2026-07-26, two capped CPU attempts were made before any locked evaluation:

- `slm230_symbol_only_v1` failed before its first optimizer step because its
  persisted semantic placeholder names violate the current opaque
  `:slot_<ordinal>` contract. The contract was not weakened.
- `slm313_local_plan_v1` used the current eight-record
  `e1281_text_form_input_required_direct_natural_v1` snapshot with the
  learned connector enabled. It reached step 18 / 2,016 target tokens before
  the 170-second interrupt, so it produced no terminal checkpoint and is not
  usable evidence.

Neither attempt measured meaning-v2, binder/reference F1, tokens, latency, or
the locked matrix. The verdict remains `unavailable` and promotion remains
false.

The replacement `slm313_local_plan_1k_v1` completed locally: CPU scratch,
8 canonical direct-natural records, 9 steps, 1,006 target tokens, and 18.47s.
It wrote a local, explicit-no-sync checkpoint solely to execute the planned
locked interventions. It has no evaluation result yet and is neither reusable
nor promotable.
