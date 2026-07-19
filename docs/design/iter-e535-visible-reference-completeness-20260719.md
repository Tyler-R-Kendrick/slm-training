# E535 — visible generated-reference completeness

E535 tests a choice-codec topology intervention after E534 repaired shallow
slot ownership but left reference-graph validity at 0/4. While a root
expression is open, the new opt-in bias prefers each unused legal reference to
an already-generated bound element once. It uses only the decoder's generated
declarations, never gold components or the gold graph, and fails closed outside
honest slot-constrained choice decoding.

The E531 checkpoint and complete E534 OOD n=4 recipe are held fixed. E535 adds
only `--visible-reference-decode-weight 4`. The CPU run completed under the
170-second process cap from clean commit `3874815` and emitted AgentEvals JSONL
plus the pinned AgentV SDK bundle.

| Metric | E534 control | E535 reference bias | Delta |
| --- | ---: | ---: | ---: |
| Syntax parse rate | 1.0000 | 1.0000 | 0.0000 |
| Meaningful program rate v1 | 0.2500 | 0.2500 | 0.0000 |
| Placeholder fidelity | 1.0000 | 1.0000 | 0.0000 |
| Structural similarity | 0.1959 | 0.1959 | 0.0000 |
| Component type recall | 0.5417 | 0.5417 | 0.0000 |
| Reward | 0.7402 | 0.7402 | 0.0000 |
| AST node F1 | 0.1627 | 0.1627 | 0.0000 |
| AST edge F1 | 0.0417 | 0.0417 | 0.0000 |
| Reference-graph exact | 0.0000 | 0.0000 | 0.0000 |
| Strict binding-aware meaning | 0.0000 | 0.0000 | 0.0000 |
| AgentV | 0 / 1 | 0 / 1 | unchanged |

Telemetry is decisive: the reference bias had zero applications and changed
zero choices. On these generated trajectories, the choice grammar did not
expose an unused-reference alternative at a multi-candidate token decision.
Therefore token-level root reranking is unreachable for the observed failure.

Reject E535 as a quality lever and do not train it. Retain the fail-closed
instrumentation for future reachability diagnostics, but the next intervention
must operate at completion-path or declaration/reference-plan scope and first
prove non-zero causal reach. Machine-readable evidence is in
[the E535 JSON](iter-e535-visible-reference-completeness-20260719.json).
