# E648 — dynamic-literal-aware semantic-root probe

Date: 2026-07-20
Status: completed positive scratch result; retained; not ship

E648 routes semantic-root verification through the tokenizer's normal decode
path. That path expands dynamic `LIT_STR` frames before parsing, so the verifier
now judges the same program representation that generation will emit rather
than rejecting the internal marker as an unknown grammar choice.

No training ran and no checkpoint was created. One capped CPU OOD `n=4` run
reused E620's rejected local-only checkpoint with the exact E644 policy. It
completed without timeout or fallback and emitted AgentEvals JSONL plus an
AgentV SDK bundle without execution errors.

| OOD `n=4` | E644/E647 baseline | E648 r1 |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 / strict v2 | 0.7500 / 0.7500 | 1.0000 / 0.7500 |
| fidelity / validity | 0.8500 / 0.9100 | 0.9500 / 0.9700 |
| structure / component recall | 0.6056 / 0.6875 | 0.7355 / 0.8750 |
| reward | 0.9115 | 0.9640 |
| AST node / edge F1 | 0.6778 / 0.4464 | 0.7987 / 0.5798 |
| latency p50 / p95 | 2771.18 / 13031.29 ms | 2750.02 / 7705.93 ms |
| root applications / choice changes | — | 38 / 26 |
| root abstentions | 1 | 0 |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Dashboard now closes as a valid five-section `Stack` containing the generated
status, callout, and metric sections. Gallery, Modal, and Auth are byte-identical
to E644, so the repair is localized to the previously unreachable root. Retain
the treatment stamped v93. After rebasing onto E647 deduplicated telemetry v95,
the append-only lineage records the retained E648 behavior as v96: all
non-strict quality measures improve, strict v2 does not regress, and
p95 latency falls substantially. This remains non-ship evidence because the
diagnostic subset has only four records and AgentV fails its evidence-size gate.
No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e648-dynamic-literal-root-probe-20260720.json).
