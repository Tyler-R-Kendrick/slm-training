# Autotrain c1795: component-edge alignment null

**Verdict:** reject the train-only component-edge alignment objective. The
size-matched treatment changed checkpoint bytes and raised training loss and
wall time, but produced an exact smoke quality tie. Its 3.71% p50 improvement
is below the preregistered 5% efficiency floor.

| Arm | Params | Loss / train wall | Smoke quality | p50 | Decision |
| --- | ---: | --- | --- | ---: | --- |
| edge alignment 1.0 | 1,766,987 | 21.2387 / 6.85 s | parse 1; meaning 0; structure .05750; binder F1 .6333; recall 0; fidelity .5278; reward .76533 | 1,302.79 ms | rejected null |
| matched control 0.0 | 1,766,987 | 17.9337 / 3.16 s | parse 1; meaning 0; structure .05750; binder F1 .6333; recall 0; fidelity .5278; reward .76533 | 1,352.96 ms | matched baseline |

Both CPU scratch arms used 21 steps, batch 2, seed 101795, the same
`component-edge` auxiliary head, and exactly 1,766,987 trainable parameters.
The treatment changed only `component_edge_alignment_loss_weight: 0→1`.
Both completed all 3 smoke documents without a timeout. The pinned AgentV
bundles are complete with zero execution errors.

This is fixture evidence only. Both arms fail ship gates for evidence volume,
meaningful-program rate, component recall, AST BEQ, canonical BEQ, and missing
held-out/adversarial/OOD/RICO suites. Neither checkpoint is reusable,
promoted, synced, or ship-ready. Lean is `not_applicable:screening`; there is
no champion or promotion optimum to prove.

The null exhausted the registered quality-arm bank. Campaign harness v93 now
removes every stale executable priority when no distinct arm remains and emits
a prerequisite `repair_harness(model_build)` action instead of proposing the
matched control as an experiment. Historical confirmation rows also project
the authoritative champion-queue disposition, so c1793 renders as rejected.

Next priority: expand the canonical model-build harness with a genuinely new,
size-matched objective. The existing versioned semantic-contrast corpus and
SLM-292 loss are the strongest candidate because they supervise semantic
separability rather than retuning another exhausted structural weight. The
next run must remain a bounded fixture screen and must not open promotion or
Lean preflight unless fresh quality evidence later confirms.

Machine evidence:
[`autotrain-cycle-1795-edge-alignment-null.json`](autotrain-cycle-1795-edge-alignment-null.json).
