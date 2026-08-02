# Autotrain c1741: fresh-checkout AgentV SDK setup blocks screening

**Verdict:** infrastructure-only screening cycle, non-positive. A fresh
container checkout of `continuous-openui-local` (loop cycle 1) had never run
`npm ci`, so `node_modules/@agentv/core` was absent. Both matched-control and
bounds arms trained and produced complete `smoke` suite metrics (parse,
binder F1, structural similarity), but the AgentV ship-gates publish step
crashed for both arms, so neither run has a scoreable `scoreboard.json`. This
is not model evidence; it is an unrepaired checkout defect.

## Result matrix

| Arm | Params | Steps | parse | binder F1 | meaningful | structure | p50 (ms) | Ship gates | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| control | 1,608,962 | 21 | 1.0 | 0.6333 | 0.0 | 0.0575 | 1,467.53 | missing_scoreboard | incomplete |
| bounds (candidate) | 1,608,962 | 21 | 1.0 | 0.6333 | 0.0 | 0.0575 | 1,536.93 | missing_scoreboard | incomplete |

`smoke.structural_similarity` (the declared primary endpoint) shows
control=candidate=0.0575 — an artifact of an incomplete comparison, not a
measured null delta. Handoff: `climb_state=inconclusive`,
`ship_state=blocked`, frozen manifest
`3c9fc6142c72c5e4ec83428f52fef128cf557f922c0ae31d50aee47497b66633`.

## Root cause

`scripts.evaluate_model --ship-gates` calls
`slm_training.evals.agentv.publish_agentv_evaluation`, which requires
`node_modules/@agentv/core` (resolved by
`src/slm_training/evals/agentv.py::_agentv_runtime`). It raised:

```
RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER
```

A second, independent environment defect compounded this: the container's
default `NODE_OPTIONS="--import tsx"` breaks `node` invocation outright
(`node: --import tsx is not allowed in NODE_OPTIONS`, exit 9), so a naive
`npm ci` also failed until `NODE_OPTIONS` was unset for the shell running
node/npm/python subprocesses.

## Repair (no code change)

1. `env -u NODE_OPTIONS npm ci` — installs `node_modules/@agentv/core` and
   `scripts/run_agentv_eval.mjs`'s runtime dependencies.
2. `export NODE_OPTIONS=""` for the shell driving `run_autotrain_continuous`
   / `evaluate_model` subprocesses, so `node` invocations spawned by the
   Python harness inherit a clean environment.

No harness family is implicated (`harness_signals: []`); this is checkout
bootstrap, not a canonical-code defect, so no `improve-openui-harnesses`
lane and no `versions.json` bump apply.

## Next-run priorities

1. Replay the identical frozen control/bounds arms
   (`c20260802-continuous-openui-local-8c0b60dd-c1-*`) now that the AgentV
   SDK is installed and `NODE_OPTIONS` is clean; require a complete
   `scoreboard.json` with `gates` before any model disposition.
2. Do not treat this cycle's identical 0.0575/0.0575 structural-similarity
   pair as a null-effect model result — it is pre-publish-crash partial data.
3. No checkpoint was promoted; `docs/MODEL_CARD.md` / README are unchanged.

Machine-readable evidence is in
[`autotrain-cycle-1741-agentv-sdk-setup.json`](autotrain-cycle-1741-agentv-sdk-setup.json).
