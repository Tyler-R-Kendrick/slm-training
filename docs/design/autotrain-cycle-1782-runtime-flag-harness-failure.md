# Autotrain c1782: runtime-flag evaluation failed before measurement

**Verdict:** incomplete and non-scoreable. Both size-matched arms trained, but
evaluation failed before producing a scoreboard because the first c1781 repair
routed non-registered model config fields through the curated OpenFeature
registry. The registry rejected `compact_active_canvas` explicitly; no model or
runtime comparison is available.

| Arm | Params / train | Evaluation | Decision |
| --- | --- | --- | --- |
| component plan | 1,755,764; 20 steps; loss 17.88950; 5.85 s | failed before scoreboard: unknown experiment lever `compact_active_canvas` | incomplete |
| matched control | 1,755,764; 20 steps; loss 14.16453; 2.12 s | same typed failure | incomplete |

The failure is harness evidence, not a training failure. Neither arm has
quality, latency, AgentV, or gate evidence. Both explicit no-sync checkpoints
are restricted to the exact frozen replay and must not otherwise be reused,
promoted, synced, or shipped. Lean is `not_applicable:screening`; no scoreable
champion or proof target exists.

The canonical repair now derives a checkpoint runtime-override allowlist from
explicit CLI options, preserves every other checkpoint-declared runtime field,
and snapshots the loaded effective model config. Components are
`harness.autoresearch.experiment_campaign/v85`, `harness.flags/v2`, and
`harness.model_build.eval/v76`. The next run must replay the frozen c1782
control and candidate manifests before trying another hypothesis.

Machine evidence:
[`autotrain-cycle-1782-runtime-flag-harness-failure.json`](autotrain-cycle-1782-runtime-flag-harness-failure.json).
