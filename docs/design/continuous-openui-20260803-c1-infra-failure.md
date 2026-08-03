# Autotrain c1 (continuous-openui-20260803): cold sandbox, AgentV preflight added

**Verdict:** infrastructure failure, not scoreable. Training completed for
both the control and `-bounds` `wf_smoke_v2` arms (1,608,962 params, 21
steps, losses `22.6219` for both, checkpoints `f2fe8f5a...b81b2` control /
`2c1749c6...41742f64` bounds — bit-identical to the `2026-08-03` c2 checkpoints
already on `main`, confirming this is a deterministic fixture replay, not a
new model), but `evaluate_model.py --ship-gates` crashed before producing a
scoreboard for either arm: `RuntimeError: AgentV SDK is unavailable; run npm
ci in the checkout or set AGENTV_RUNNER`. Neither arm has smoke metrics; this
is not evidence about the model.

This is a fresh remote-execution sandbox for this session, distinct from the
one that landed the `2026-08-03` c1/c2 torch and NODE_OPTIONS fixes on
`main` — it simply hadn't been bootstrapped yet:

1. `python -m scripts.run_autotrain_continuous ...` failed identically to the
   prior c2 cycle: AgentV SDK unavailable because `npm ci` had never run in
   this checkout.
2. Running `npm ci` directly failed with `node: --import tsx is not allowed
   in NODE_OPTIONS` — the host shell exports a `NODE_OPTIONS` this Node build
   rejects for the `npm` process itself. This is a *different* exposure than
   the already-fixed `bridge_utils.sanitized_node_env()`, which only
   sanitizes the environment for node subprocesses spawned *by Python*, not
   for `npm` invoked directly from the shell. `env -u NODE_OPTIONS npm ci` —
   the exact invocation already codified in `scripts/setup_dev_env.sh` —
   succeeded and installed `node_modules/@agentv/core`.
3. With the SDK installed, both arms had already spent their full training
   wall (`4.12s` / `3.72s`) before the eval-stage crash surfaced the missing
   SDK, and the driver attributed it as a per-arm `harness_failure` rather
   than a single fast, correctly-labeled infra gap.

**Repair landed this cycle** (commit `41dfe6d`): added
`slm_training.evals.agentv.ensure_agentv_available(repo_root)`, a thin
wrapper around the existing `_agentv_runtime()` resolution, and called it at
the top of `scripts.run_autotrain_continuous.run_cycle()` before git sync.
A cold checkout missing the AgentV SDK now fails in milliseconds with the
same actionable "run npm ci" message instead of after a full wasted train
step. No change to `_agentv_runtime()`'s resolution logic or the
NODE_OPTIONS sanitization that already landed; this is a preflight-timing
fix only, covered by two new regression tests in
`tests/test_evals/test_agentv.py`.

No scoreboard, no smoke metrics, no ship-gate result exists for this cycle;
the checkpoints are local, explicit no-sync, and not reusable, promotable, or
ship evidence. Lean is `not_applicable:screening`.

Next: replay the identical frozen `-bounds` arm (`retry_measurement`) now
that the AgentV SDK is installed in this sandbox and the driver preflights it
before spending training wall.

Machine evidence:
[`continuous-openui-20260803-c1-infra-failure.json`](continuous-openui-20260803-c1-infra-failure.json).
