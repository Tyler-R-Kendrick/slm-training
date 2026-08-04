# Autotrain c1799: slot augmentation rejected

**Verdict:** reject request-local slot permutation and alpha-renaming
augmentation at this recipe. Both size-matched arms completed 24 CPU scratch
steps and the full 3-record smoke screen. The candidate reduced structural
similarity from 0.13750 to 0.05750 and binder-reference F1 from 0.82222 to
0.63333.

| Arm | Params | Loss | Structure | Binder F1 | Fidelity | p50 ms | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,608,962 | 16.15545 | 0.13750 | 0.82222 | 0.72222 | 5086.88 | retain baseline only |
| slot augmentation | 1,608,962 | 18.68529 | 0.05750 | 0.63333 | 0.52778 | 1004.31 | reject approach |

Both arms have parse rate 1.0, meaningful-program rate 0, component recall 0,
reward 0, three completed documents, and zero decode timeouts. AgentV completed
both bundles without execution errors. The candidate's large latency reduction
does not override the preregistered quality and binder non-regressions. This is
fixture evidence only; honest ship gates fail.

The checkpoints are local scratch artifacts under the c1799 campaign, with
explicit no-sync policy. Control SHA-256 is `51f71d3d...730f2`; candidate SHA-256
is `108bfc6b...73a3a`. Neither is reusable, promotable, synced, or ship evidence.
Lean is `not_applicable:screening`; no confirmation or promotion preflight was
authorized.

Two reporting gaps were repaired after the result. Screening actually evaluated
`smoke.structural_similarity`, but the handoff and terminal matrix retained the
CLI-requested held-out suite and rendered `Primary —`. Campaign harness v99
binds the primary to the evaluated policy suite and the matrix follows the
handoff's effective metric. The locked manifest, train command, and feature-flag
snapshot prove `symbol_slot_augmentation=true`, but the train summary recipe
omitted that field. Train harness v31 now records the corruption pattern,
runtime-symbol mode, slot augmentation, semantic masks, and constraint-graph
mode in every summary. These repairs change provenance, not the measured result.

Next priority: test `mask_pattern=mixed` against explicit `random` at identical
size, seed policy, and smoke evaluation. It changes the corruption curriculum,
not deterministic legality or model capacity.

Machine evidence:
[`autotrain-cycle-1799-slot-augmentation-rejected.json`](autotrain-cycle-1799-slot-augmentation-rejected.json).
