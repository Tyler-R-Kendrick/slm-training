# Autotrain c1 (continuous-openui-local, 2026-08-04): AgentV SDK missing, not a model result

**Verdict:** infrastructure failure, not scoreable. Both the `control` and
`-bounds` `wf_smoke_v2` arms trained and decoded normally -- `parse_rate=1.0`,
`structural_similarity=0.0575`, `binder_reference_f1=0.633` on both arms, no
attributable delta -- then crashed inside `evaluate_model.py --ship-gates` at
`publish_agentv_evaluation()` with:

```
RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER
```

This fresh scheduled-task sandbox had never run the Node half of
`scripts/setup_dev_env.sh` (`env -u NODE_OPTIONS npm ci`), so
`node_modules/@agentv/core` did not exist yet. Neither arm produced a
`scoreboard.json`; this is not evidence about the model, the compiler-tree
grammar objective, or any lever.

The continuous driver's `SDLC_PHASE_A` classification flagged this correctly
as `NON_POSITIVE` / `harness_failure` and routed a `repair_harness` action
against `model_build` before permitting any new model hypothesis.

Root cause and disposition: this is the **same class of gap** already fixed
in `autotrain-cycle-c1-torch-missing-infra-failure` (commit `72fdffaf`) --
`src/slm_training/evals/agentv.py::_agentv_runtime()` already fails closed
with an actionable message, and `scripts/setup_dev_env.sh` already runs
`npm ci`. No source fix was needed here; the agent ran
`env -u NODE_OPTIONS npm ci` in this session (`node_modules/@agentv/core` is
now present) rather than author a redundant patch. The `repair_harness`
handoff action for this cycle is therefore **not** acknowledged with a
receipt: the evidence contract requires a commit dated after this campaign's
`integration_commit`, and manufacturing a no-op commit against an
already-fixed, already-documented gap would be process theater, not repair.

No checkpoint was created; nothing here is reusable, promotable, or ship
evidence. Lean is `not_applicable:screening`. AgentV bundles never ran to
completion (both arms failed at publish, after decode).

Next: replay the identical frozen `-bounds` arm (`retry_measurement`) now
that `node_modules/@agentv/core` is installed in this sandbox. A future
scheduled-task session will hit the identical gap again unless the sandbox
bootstrap (`scripts/setup_dev_env.sh`) runs automatically at session start --
worth a follow-up `SessionStart` hook, tracked separately rather than bundled
into this non-positive cycle's evidence.

Machine evidence:
[`autotrain-cycle-20260804-c1-agentv-missing-infra-failure.json`](autotrain-cycle-20260804-c1-agentv-missing-infra-failure.json).
