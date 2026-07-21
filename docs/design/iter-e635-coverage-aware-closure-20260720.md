# E635 — role-compatible coverage before frame closure

Date: 2026-07-20
Status: completed mixed; default-off scratch policy retained; not ship

E634 showed that ten times more scratch training reduced train loss without
fixing OOD slot coverage. E635 instead extends the existing
`slot_coverage_close_decode_weight` policy. When a component, object, or array
can legally close while a visible slot remains unused, the decoder floors the
best compiler-legal role-compatible continuation above closure. It abstains
when the public schema and prompt-derived semantic-role map cannot prove a
compatible continuation. The prior behavior—reward typed-array closure after
all visible slots are covered—remains intact.

## Reused checkpoint and recipe

No training ran and no checkpoint was created. Both E635 evals reused E634's
local-only rejected checkpoint, SHA-256
`3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`.
The OOD `n=4` recipe is otherwise identical to E634 treatment: CPU, honest slot
contract, slot contract in context and constrained decode, public-schema role
candidates, the retained semantic-plan weights, coverage weight 2, role-slot
weight 8, and a 160-token canvas. Every run emitted AgentEvals JSONL and an
AgentV SDK bundle without execution errors.

## r1 failure and correction

The clean v60 r1 proved schema reachability but did not require a direct slot's
active owner to match its public semantic role. Gallery improved by emitting
`details=:gallery.caption`, but Button and TextContent could absorb email,
name, and confirm slots. That implementation was rejected immediately.

Model v61 requires owner-role compatibility before a direct slot can outrank
closure. Unit coverage includes typed-array completion, compatible component
continuation, compatible object-property continuation, and wrong-owner direct
slot abstention.

## Measured result

The baseline is E634 treatment on the byte-identical checkpoint and recipe.

| OOD `n=4` | E634 baseline | E635 r1 | E635 r2 |
| --- | ---: | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 | 1.0000 |
| meaningful v1 | 0.5000 | 0.5000 | 0.7500 |
| strict meaning v2 | 0.0000 | 0.0000 | 0.0000 |
| v2 judgment coverage | 1.0000 | 1.0000 | 1.0000 |
| placeholder fidelity | 0.5500 | 0.6250 | 0.5917 |
| placeholder validity | 0.7300 | 0.6750 | 0.7550 |
| structural similarity | 0.4886 | 0.4223 | 0.4029 |
| component recall | 0.4792 | 0.4375 | 0.5000 |
| reward | 0.8140 | 0.6668 | 0.8175 |
| AST node F1 | 0.5437 | 0.4770 | 0.4690 |
| AST edge F1 | 0.3750 | 0.1667 | 0.2625 |
| latency p50 | 3116.67 ms | 2609.49 ms | 3346.44 ms |
| latency p95 | 13704.44 ms | 13445.24 ms | 6198.48 ms |
| AgentV | 0/1 | 0/1 | 0/1 |

No decode timed out or fell back. Corrected r2 improves meaningful v1 by 0.25,
fidelity by 0.0417, validity by 0.0250, component recall by 0.0208, reward by
0.0035, and p95 latency by 7.51 seconds. It also gives Gallery its caption and
Modal its confirmation Button. Those gains are not a ship result: strict v2
remains zero, structural similarity falls by 0.0856, and both AST metrics
regress. Dashboard still emits only one of five slots; Auth chooses Button and
SwitchGroup instead of the two required Input components and misses email.

## Decision

Retain v61 as a default-off scratch policy and reject r1. Do not sync, promote,
or make a readiness claim. The next experiment should instrument and constrain
root-level inventory termination and Auth owner selection; increasing the
closure weight would amplify the same mixed trade-off without addressing the
remaining failure class.

Concurrent `main` merges rewrote the local feature commits before publication.
Both arms were replayed from clean detached worktrees at the reachable v60 and
v61 model commits; semantic outputs and all non-timing headline metrics were
identical. The JSON records retain the earlier clean-run SHAs and designate the
reachable replays as authoritative provenance.

Evidence: [r2 JSON](iter-e635-coverage-aware-closure-20260720.json) and
[rejected r1 JSON](iter-e635-coverage-aware-closure-r1-20260720.json).
