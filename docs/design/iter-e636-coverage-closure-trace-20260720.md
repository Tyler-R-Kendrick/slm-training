# E636 — coverage-closure intervention trace

Date: 2026-07-20
Status: completed diagnostic; no behavior change; not ship

E636 adds bounded score traces and aggregate counters to the E635
`slot_coverage_close_decode_weight` policy. It records the open owner/frame,
missing visible slots, choice before and immediately after the policy, and the
final choice after downstream decoder policies. Decode scores are unchanged.

## Reused checkpoint and recipe

No training ran and no checkpoint was created. The clean CPU eval reused
E634's rejected local-only checkpoint, SHA-256
`3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`.
It used the exact E635 r2 OOD `n=4` recipe: honest slot contract, slot contract
in context and constrained decode, public-schema role candidates, coverage
weight 2, role-slot weight 8, the retained semantic-plan weights, and a
160-token canvas. The run completed under the three-minute cap with no timeout
or fallback and emitted AgentEvals JSONL plus an AgentV SDK bundle without
execution errors.

## Reproduction check

All semantic outputs and metrics exactly reproduce E635 r2. Timing is reported
as observational noise, not a performance claim.

| OOD `n=4` | E635 r2 | E636 trace |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 | 0.7500 | 0.7500 |
| strict meaning v2 | 0.0000 | 0.0000 |
| v2 judgment coverage | 1.0000 | 1.0000 |
| placeholder fidelity | 0.5917 | 0.5917 |
| placeholder validity | 0.7550 | 0.7550 |
| structural similarity | 0.4029 | 0.4029 |
| component recall | 0.5000 | 0.5000 |
| reward | 0.8175 | 0.8175 |
| AST node / edge F1 | 0.4690 / 0.2625 | 0.4690 / 0.2625 |
| latency p50 / p95 | 3346.44 / 6198.48 ms | 3067.47 / 6277.99 ms |
| AgentV | 0/1 | 0/1 |

## Intervention evidence

The policy applied 11 times and changed the immediate argmax 8 times. Nine
interventions survived all later policies. Two Gallery interventions selected
`Button`, but downstream policies replaced them with `&0` and `]`; these are
the only observed late overrides.

- Dashboard: two interventions fired inside already-wrong owners (`Button`,
  then `Card`). They selected `AccordionItem` and `TextContent`; the model still
  ended with one of five slots. The missing Dashboard inventory is therefore
  upstream of this closure policy, not a root-level termination that this
  policy can see.
- Gallery: the policy correctly opened the image object and selected `alt` and
  `details`. Later, two attempted child `Button` continuations were overridden,
  leaving the hint and CTA absent.
- Modal: the policy preserved `TextContent`, changed premature `]` to the
  required `Button`, then rewarded `]` after complete coverage. This is the
  clean success case.
- Auth: while inside `Button` argument 1 with name and email missing, the
  policy changed closure `-` to `SwitchGroup`. Nothing downstream reversed it.
  Public-schema role expansion treats the generic `name` role as compatible
  with `SwitchGroup`, even though the authored prompt explicitly requires two
  Input components.

## Decision

Keep the telemetry and retain E635 default-off. Do not promote or claim ship
readiness: strict v2 remains zero, AgentV fails, and no checkpoint was created.
The next lever should make closure continuation ranking prefer explicit
prompt-owned component mentions (for example, `name input` and `email input`)
over broad schema-only role matches. Dashboard needs a separate upstream
inventory/owner correction; increasing the closure weight cannot create a
missing root inventory.

Evidence: [JSON](iter-e636-coverage-closure-trace-20260720.json).
