# E389–E395 effective choice canvas — 2026-07-18

E389 evaluated frozen RICO rows 336–384 under a policy that reported a
320-token LTR canvas. One 12-slot row retained only 2 slots, reducing shard
fidelity to 0.9688. E390 merged the then-current shards into 384 rows.
Diagnosis showed that choice-codec generation ignored
`grammar_ltr_max_tokens` and silently used the checkpoint's `gen_len=58`,
while evaluation telemetry reported 320.

The shared generation path now uses
`min(grammar_ltr_max_tokens, max_target_len)` when no explicit `max_len` is
provided. For this checkpoint the honest effective canvas is 256 tokens.
Evaluation telemetry reports that same limit. E391 re-evaluates the 12-slot
outlier: fidelity improves from 0.1667 to 1.0, structure from 0.1277 to 0.3808,
component recall from 0.5 to 1.0, and reward from 0 to 1.0. E392 repeats all
48 rows 336–384: fidelity improves from 0.9688 to 1.0, structure from 0.6395
to 0.6514, recall from 0.9826 to 0.9931, and reward from 0.9892 to 0.9994,
with zero failures. E392 completes in approximately 128 seconds under the hard
290-second command cap.

E393 reruns the four complete bounded suites under the corrected 256-token
canvas. Smoke, adversarial, and OOD pass, but held-out component recall is
0.2333 against the unchanged 0.30 gate; AgentV is therefore 3/4. Fidelity is
1.0 on all four suites. E394 adds component-inventory decode weight 1.0 and
E395 doubles slot-component weight from 8 to 16; both are exactly identical to
E393 and are rejected as ineffective.

Because the old choice path actually used 58 tokens, E376–E390 results are
historical diagnostics only and invalid for current-policy selection despite
their recorded `grammar_ltr_max_tokens=320`. They must not be merged with
E391+ evidence.

**Verdict:** retain the effective-canvas correctness fix. Do not promote the
checkpoint: corrected evidence has only one 48-row RICO shard and bounded
AgentV 3/4. Improve held-out component-type selection before restarting
contiguous full-RICO accumulation.
