# E496 current-main E396 honest smoke — 2026-07-18

E496 loads the exact durable E396 checkpoint from
`hf://buckets/TKendrick/OpenUI/checkpoints/e396-balanced-type-head-continuation-r1`
and evaluates it with code revision
`2351c1f10bbda16fee0a00707434a1e057f8abde` from clean `main`.
The CPU run completed normally in 18.4 seconds under the hard three-minute
policy. It used local HF files, honest constrained slot contracts, a 320-token
LTR canvas, eight generation steps, three attempts, no unconstrained fallback,
and the complete three-record smoke suite.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.0 | 0.5556 | 0.1131 | 0.0 | 0.3023 |

AgentV passes 0/5 with zero execution errors. There are no decode timeouts or
fallbacks. One earlier setup attempt is excluded because an online HF lookup
retried before offline mode was set; it produced no model or evaluation row.

The result contradicts any interpretation of E490 as reproducible from current
`main`. E490 used a long-lived experimental decoder branch containing visible
prompt-role, semantic-slot, schema-enum, array-item, JSON-number, and typed-any
constraints that are not present on `main`. Its five-suite numbers remain valid
branch-only diagnostic evidence for the same checkpoint, not deployable-code
evidence.

**Verdict:** checkpoint persistence passes; deployable decoder provenance fails.
Keep E396 diagnostic-only. Selectively reconcile the generalized decoder
invariants with current `main`, then rerun bounded honest suites before restoring
any current-policy champion wording.
