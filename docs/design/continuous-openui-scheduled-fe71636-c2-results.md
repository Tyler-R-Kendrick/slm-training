# Continuous autotrain: 2026-08-04 (scheduled loop `fe71636`) cycle 2 — AgentV harness incomplete, repaired

**Loop:** `continuous-openui-scheduled-fe71636`
**Campaign:** `continuous-loop-20260804-continuous-openui-schedu-3d42338c-c2`
**Integration commit:** `fe716367` (`origin/main` tip at cycle start)

**Verdict:** measurement incomplete, not a model result. Both the matched
`control` and size-matched `canvas` arm trained cleanly (`1,608,962`
trainable params each) but `evaluate_model --ship-gates` exited 2 on both,
because this was a fresh checkout that had never run `npm ci` —
`_agentv_runtime` hard-fails model-ship-gate publication whenever
`node_modules/@agentv/core` is missing, even when a `package-lock.json` is
present to install it from.

```
RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER
```

This is a known recurring gap on fresh scheduled-session containers (see
prior sessions' PRs #1406, #1410, #1420, #1423 for the same root cause).

## Harness repair (`model_build` family)

Commit `2aedf3b` makes `_agentv_runtime` self-heal: when the pinned SDK is
missing but a `package-lock.json` exists, it runs `npm ci` once — via
`bridge_utils.sanitized_node_env`, so a host shell's rejected `NODE_OPTIONS`
(`--import tsx`, observed in this session too) can't make npm's own `node`
invocation exit 9 — before raising. Regression coverage:
`tests/test_evals/test_agentv.py::test_agentv_runtime_bootstraps_missing_sdk_via_npm_ci`
and `..._still_raises_when_bootstrap_fails`. Component `evals.agentv` bumped
`v7 → v8` in `versions.json`.

## SDLC Phase A

**Non-positive** (`measurement_incomplete`, `harness_failure` on both arms —
the primary metric never resolved). Per `sdlc` autotrain-iteration-delivery,
no new stack layer opens for the training cycle itself; the harness repair is
tracked code delivered as an independent unblocking commit (proven executable
unblock, pending replay confirmation next cycle).

## Next priorities

1. Replay the identical frozen `control`/`canvas` arm now that the repair is
   committed, before testing any new hypothesis (rank 1, confidence 0.95,
   `retry_measurement`).
