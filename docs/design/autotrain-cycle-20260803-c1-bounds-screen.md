# Autotrain cycle 1 — bounds screen

The corrected continuous loop completed a paired, size-matched CPU screen under
the canonical wall cap. Both arms exited normally and the Phase A handoff was
complete; this is a fixture result, not a production-learning claim.

| Arm | parse | meaningful | structure | binder F1 | p50 latency | gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Control | 1.000 | 0.000 | 0.0575 | 0.6333 | 971.56 ms | fail |
| Bounds candidate | 1.000 | 0.000 | 0.0575 | 0.6333 | 981.72 ms | fail |

Recipe: CPU scratch TwoTower, 20 steps, strict grammar-constrained compiler-tree
decode, smoke `n=3`; held-out, adversarial, OOD, and `rico_held` suites were not
run. AgentV completed with no execution errors. The candidate is a quality null
and 1.05% slower, so it is rejected and neither checkpoint is promotable,
reusable, synced, or ship-eligible.

What prevents a learning claim is capability/evidence, not Lean execution:
meaningful-program rate and exact AST/canonical agreement are zero, and the
smoke sample is below the required `n>=20`. The next hypothesis is the distinct
size-matched component-plan quality arm, retaining the unchanged control. The
semantic-contrast family must be rerun only with the new CE-mask parity telemetry
(`semantic_contrast_ce_mask_sha256` and scored-token counts).

JSON twin: [autotrain-cycle-20260803-c1-bounds-screen.json](autotrain-cycle-20260803-c1-bounds-screen.json)
