# E630 — prompt-owned closure filtering

Date: 2026-07-20
Status: completed negative; implementation reverted; not ship

E622 traced Auth's wrong `SwitchGroup` to broad public-schema role matching
inside `Button`. E630 tested the smallest apparent correction: when an
uncovered slot has a compatible component explicitly mentioned in the prompt,
restrict closure-continuation components to that prompt-mentioned subset and
otherwise retain broad schema fallback.

## Reused checkpoint and recipe

No training ran and no checkpoint was created. The clean CPU eval reused
E620's rejected local-only checkpoint, SHA-256
`3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`,
with the exact E622/E621 r2 OOD `n=4` recipe. The run completed under the
three-minute cap with no timeout or fallback and emitted AgentEvals JSONL plus
an AgentV SDK bundle without execution errors.

## Measured result

| OOD `n=4` | E622 baseline | E630 treatment |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 | 0.7500 | 0.5000 |
| strict meaning v2 | 0.0000 | 0.0000 |
| v2 judgment coverage | 1.0000 | 1.0000 |
| placeholder fidelity | 0.5917 | 0.5083 |
| placeholder validity | 0.7550 | 0.7050 |
| structural similarity | 0.4029 | 0.3379 |
| component recall | 0.5000 | 0.3750 |
| reward | 0.8175 | 0.7850 |
| AST node / edge F1 | 0.4690 / 0.2625 | 0.3857 / 0.2625 |
| latency p50 / p95 | 3067.47 / 6277.99 ms | 3449.90 / 14871.38 ms |
| closure applications / changes | 11 / 8 | 14 / 9 |
| AgentV | 0/1 | 0/1 |

Dashboard, Gallery, and Modal predictions were unchanged. Auth regressed from
`Button(create, SwitchGroup(name, ...))` to `TextContent(email)`. The trace
shows the intended local substitution happened: at `Button` argument 1 the
policy chose `Input` instead of closure for missing name/email. That nests the
Input under the already-wrong Button owner, lengthens the intermediate decode,
and destabilizes later binding/root selection. Selecting the right family at
the wrong structural depth is worse than abstaining.

## Decision

Reject and revert the prompt-owned filter. Model v64 restores v62 behavior;
v63 remains in version history as the failed evaluated arm. Do not promote or
claim ship readiness. The next correction must be frame-aware: when the
prompt-owned family differs from the active component owner and the current
position is a scalar/component argument rather than a structural child list,
close the wrong owner and let root-level inventory planning place the required
family as a sibling. Do not increase the closure weight.

Evidence: [JSON](iter-e630-prompt-owned-closure-20260720.json).
