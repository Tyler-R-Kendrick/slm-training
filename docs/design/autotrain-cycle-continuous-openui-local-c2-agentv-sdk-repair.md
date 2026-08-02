# Autotrain continuous-openui-local c2: AgentV SDK checkout bootstrap gap (infrastructure, not a model result)

**Verdict:** cycle 2 of loop `continuous-openui-local` (campaign
`continuous-loop-20260802-continuous-openui-local-8c0b60dd-c2`, frozen replay
of cycle 1's control/bounds arms) is an infrastructure failure, not a
quality-null model result. Training completed on both frozen arms
(1,608,962 trainable params each); evaluation failed identically with
`RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set
AGENTV_RUNNER` before any AgentV assertion ran.

**Root cause:** this scheduled session's checkout never ran `npm ci`, so
`node_modules/@agentv/core` did not exist. `evaluate_model.py --ship-gates`
calls `publish_model_evaluation` → `publish_agentv_evaluation` →
`slm_training.evals.agentv._agentv_runtime`, which raises unless
`AGENTV_RUNNER` is set or the checked-out SDK marker
(`node_modules/@agentv/core/package.json`) is present. This is a distinct,
earlier bootstrap gap from the already-fixed `NODE_OPTIONS` crash (commit
`14ddf5e2`, already present at this cycle's `integration_commit e97e69c9`):
that fix strips the host's `NODE_OPTIONS` from the AgentV subprocess env, but
only once the SDK is actually installed.

**Fix:** ran `npm ci` at the repo root, stripping the ambient
`NODE_OPTIONS="--import tsx" --max-old-space-size=8192` from the invocation
itself (`env -u NODE_OPTIONS npm ci`) since that same host env var blocks
bare `node`/`npm` from starting at all. Added 267 packages including
`node_modules/@agentv/core` in ~18s. Verified directly:
`slm_training.evals.agentv._agentv_runtime(repo_root)` resolves
`scripts/run_agentv_eval.mjs` + `node_modules/@agentv/core/package.json`
with `AGENTV_RUNNER` unset. `git status` confirmed the tree stayed clean
(`node_modules/` is gitignored; no `package.json`/`package-lock.json`
change). No repository code changed for this cycle.

**Next step:** replay the identical frozen control/bounds arms
(`retry_measurement`) now that the AgentV SDK checkout is present — no knob
changes, same `train_version=wf_smoke_v2`, `steps=20`, seed, and manifest.

Lean is `not_applicable:screening`; no ship or climb claim is made here.
Neither fixture checkpoint is promoted or synced.

Machine evidence:
[`autotrain-cycle-continuous-openui-local-c2-agentv-sdk-repair.json`](autotrain-cycle-continuous-openui-local-c2-agentv-sdk-repair.json).
