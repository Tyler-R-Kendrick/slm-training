# E650 — concrete role-obligation margin

Date: 2026-07-20
Status: completed positive scratch result; retained; not ship

E650 converted each visible-role-inferred family into a concrete
`(component family, visible slot)` obligation. At a compatible schema position,
only the assigned unused slot receives a margin floor. Prompt-planned families
and ordinary schema-role scoring retain their existing behavior.

The capped run was originally launched as E644. It was renumbered E650 when
the rebase incorporated the independently landed E639–E644 sequence; the
committed JSON retains the original run ID explicitly.

No training ran and no checkpoint was created. One capped CPU OOD `n=4` run
reused E620's rejected local-only checkpoint with the exact E637 policy. It
completed in 23.4 seconds without timeout or fallback and emitted AgentEvals
JSONL plus an AgentV SDK bundle without execution errors.

| OOD `n=4` | E637 r2 baseline | E650 r1 |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 / strict v2 | 0.7500 / 0.5000 | 0.7500 / 0.7500 |
| fidelity / validity | 0.6750 / 0.8050 | 0.8500 / 0.9100 |
| structure / component recall | 0.5817 / 0.6250 | 0.6056 / 0.6875 |
| reward | 0.8545 | 0.9115 |
| AST node / edge F1 | 0.6437 / 0.4554 | 0.6778 / 0.4464 |
| latency p50 / p95 | 2303.87 / 6831.74 ms | 2963.90 / 13254.53 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Gallery now emits each inferred role exactly once: CTA binds to Button and both
hint roles bind to distinct TextContent instances, eliminating E647/E649's
nested Button spam. Dashboard also binds distinct title/body roles. Retain the
treatment stamped v88 as the next scratch baseline. After rebasing onto E649
restoration v91, the append-only lineage records the retained E650 behavior as
v92. Every headline quality metric except edge F1 improves, but do not claim
ship readiness: this is a diagnostic `n=4`
subset, AgentV fails the evidence-size gate, edge F1 slips slightly, and p95
latency nearly doubles. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e650-role-obligation-margin-20260720.json).
