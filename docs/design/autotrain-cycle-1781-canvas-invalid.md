# Autotrain c1781: canvas comparison is invalid

**Verdict:** non-scoreable. The experiment plan and training snapshot correctly
set `compact_active_canvas=false` for the control and `true` for the candidate,
but the evaluation compiler omitted the runtime-symbol flags. Evaluation then
recorded `true` for both arms, so the apparent 3.76% candidate slowdown cannot
be attributed to canvas compaction.

| Arm | Params / train | Reported smoke | Runtime flag integrity | Decision |
| --- | --- | --- | --- | --- |
| canvas | 1,608,962; 22 steps; loss 20.37200; 2.89 s | n=3; parse 1; meaning .3333; structure .17417; binder .6333; fidelity .5278; reward .76533; p50 1,086.17 ms | train `true`; eval `true` | invalid |
| nominal control | 1,608,962; 22 steps; loss 20.37200; 3.17 s | identical quality; p50 1,046.84 ms | train `false`; eval `true` | invalid control |

Both AgentV bundles completed with zero execution errors, but post-run
provenance establishes a measurement-integrity failure. Gates fail independently
and this CPU fixture is not ship evidence. Both explicit no-sync checkpoints
are provenance-only and must not be reused, promoted, synced, or shipped. Lean
is `not_applicable:invalid_measurement`; there is no champion or proof target.

Harness repair `harness.autoresearch.experiment_campaign/v84` now serializes
all declared TwoTower runtime-symbol levers into the evaluation command through
the typed feature-flag surface. The regression test verifies the exact mapping,
including an explicit false control. The next `both` diagnostic may run only
after this repair and must establish its result against its own corrected
matched control.

Machine evidence:
[`autotrain-cycle-1781-canvas-invalid.json`](autotrain-cycle-1781-canvas-invalid.json).
