# Autotrain `continuous-openui-local` c1: AgentV SDK bootstrap gap (infra, not model)

**Verdict:** fixture/scratch measurement **incomplete**, not a model result. Both
matched 1,608,962-parameter arms (`control`, `bounds`) trained cleanly for 21 CPU
steps, but `scripts.evaluate_model --ship-gates` crashed before writing any
scoreboard:

```
RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER
```

## Result matrix

| Arm | Training | Eval | Disposition |
| --- | --- | --- | --- |
| control | 21 steps, 1,608,962 params, exit 0 | crashed pre-scoreboard, exit 2 | measurement_incomplete |
| bounds | 21 steps, 1,608,962 params, exit 0 | crashed pre-scoreboard, exit 2 | measurement_incomplete |

`SDLC_PHASE_A` classified the cycle **`NON_POSITIVE`**
(`harness_failure` + `measurement_incomplete` on both arms;
`primary_metric_null_or_worse` is a side effect of both scoreboards being
absent, not a real comparison). No stack layer was opened for this cycle,
per the autotrain-iteration-delivery gate.

## Harness diagnosis

Every fresh continuous-autotrain container is a clean checkout — `node_modules/`
is never present until someone runs `npm ci`. This is the same infra-gap family
that already hit this loop on 2026-08-02 (`c1`/`c2`: missing torch, then a
rejected host `NODE_OPTIONS`; see the merged fix in commit `72fdffa`). This
cycle hit the AgentV-SDK sibling of that gap: the `@agentv/core` package was
never installed, so `evals.agentv._agentv_runtime` raised its (correct,
fail-closed, I6-compliant) `RuntimeError` before evaluation could start.

**Repair (commit `0a7a7219a1059a288594473cdae50c1c77e22122`):**
`src/slm_training/evals/agentv.py` `_agentv_runtime` now attempts one bounded,
`NODE_OPTIONS`-sanitized `npm ci` in the checkout when
`node_modules/@agentv/core/package.json` is missing, before falling back to the
original actionable error. This is an environment self-heal, not a decode-path
or model change: the fail-closed behavior when bootstrap genuinely cannot
resolve the SDK (no network, no `package-lock.json`, etc.) is unchanged.
Regression tests: `test_agentv_runtime_bootstraps_missing_sdk_via_npm_ci`,
`test_agentv_runtime_raises_actionable_error_when_bootstrap_cannot_help`
(`tests/test_evals/test_agentv.py`). Component bump: `evals.agentv` v7 → v8
(`src/slm_training/resources/versions.json`).

The frozen arm manifest digest
(`c4c7bc4837d5a4f076eaa574ad119c9a9709376e9923d8becf7360215f813528`) is
preserved; the queued `retry_measurement` action replays the identical
`control`/`bounds` pair rather than authoring a new hypothesis.

## Next-run priorities

1. Replay the identical frozen `control`/`bounds` arms now that the AgentV SDK
   self-heal is committed; require a complete scoreboard before any model
   disposition.
2. Keep the repair scoped to environment bootstrap only — no model, decode-path,
   or grammar-constraint change accompanies it.

No checkpoint was created or promoted, so no `MODEL_CARD.md` / README update is
required. Machine-readable values are in
[`autotrain-cycle-continuous-openui-local-c1-agentv-bootstrap.json`](autotrain-cycle-continuous-openui-local-c1-agentv-bootstrap.json).
