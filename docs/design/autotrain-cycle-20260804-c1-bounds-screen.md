# Autotrain cycle 1 — bounds screen (2026-08-04)

Scheduled continuous-loop invocation (`continuous-openui-local`) ran one
supervised, size-matched CPU screen under the canonical wall cap. Both arms
exited normally and the Phase A handoff was complete; this is a fixture
result, not a production-learning claim.

| Arm | parse | meaningful | structure | binder F1 | placeholder fidelity | p50 latency | gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Control | 1.000 | 0.000 | 0.0575 | 0.6333 | 0.5278 | 4726.59 ms | fail |
| Bounds candidate | 1.000 | 0.000 | 0.0575 | 0.6333 | 0.5278 | 4614.58 ms | fail |

Recipe: CPU scratch TwoTower, `wf_smoke_v2` train data, 20 steps, strict
grammar-constrained compiler-tree decode, smoke `n=3`; held-out, adversarial,
OOD, and `rico_held` suites were not run (fixture-scale bounded cycle).
AgentV completed with no execution errors. The candidate is a quality null
(identical structural similarity to control) so it is rejected; neither
checkpoint is promotable, reusable, synced, or ship-eligible.

What prevents a learning claim is capability/evidence, not Lean execution:
meaningful-program rate and exact AST/canonical agreement are zero, and the
smoke sample is below the required `n>=20`. SDLC Phase A classifies this cycle
**NON_POSITIVE** (`fixture_insufficient_n` on both arms,
`primary_metric_null_or_worse`) — per `autotrain-iteration-delivery`, no
stacked PR opens for this cycle; only this local commit + docs. The next
hypothesis is the distinct size-matched component-plan quality arm, retaining
the unchanged control.

Both `last.pt` checkpoints under
`outputs/autoresearch/continuous-loop-20260804-continuous-openui-local-8c0b60dd-c1/runs/`
are local-only fixture-screen artifacts (gitignored, never PR'd) — see
[`MODEL_CARD.md`](MODEL_CARD.md#current-checkpoint-roster) for the roster row.

JSON twin: [autotrain-cycle-20260804-c1-bounds-screen.json](autotrain-cycle-20260804-c1-bounds-screen.json)
Prior comparable cycle: [autotrain-cycle-20260803-c1-bounds-screen.md](autotrain-cycle-20260803-c1-bounds-screen.md)
