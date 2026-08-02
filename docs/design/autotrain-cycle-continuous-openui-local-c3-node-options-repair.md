# Autotrain continuous-openui-local c3: AgentV runner NODE_OPTIONS defect (harness repair, code changed)

**Verdict:** cycle 3 of loop `continuous-openui-local` (campaign
`continuous-loop-20260802-continuous-openui-local-8c0b60dd-c3`, frozen replay
of cycle 2's control/bounds arms) is a **genuine canonical harness defect**,
repaired in this cycle. Both frozen arms (checkpoints reused, 1,608,962
trainable params each) failed evaluation identically with
`RuntimeError: AgentV SDK evaluation failed: node: --import tsx is not
allowed in NODE_OPTIONS`.

**Root cause:** `publish_agentv_evaluation`
(`src/slm_training/evals/agentv.py`) spawned `node scripts/run_agentv_eval.mjs`
inheriting the full parent environment. This session's host `NODE_OPTIONS`
(`--import tsx` plus `--max-old-space-size=8192`) makes this pinned Node
build refuse to start. Cycle 2's `npm ci` repair installed the AgentV SDK
checkout but did not touch this subprocess's environment, so the very next
pipeline stage hit this distinct failure. Unlike cycles 1–2, this is not an
environment-bootstrap gap — the same `NODE_OPTIONS`-sanitization pattern
already exists elsewhere in this repo
(`src/slm_training/dsl/grammar/backends/graphql_js.py:_sanitized_env`,
already merged on `main`) but had never been mirrored onto the AgentV
runner's subprocess call.

**Fix:** added `_sanitized_node_env()` to `agentv.py` (mirrors
`graphql_js._sanitized_env()`) and passed it as the AgentV subprocess `env=`.
Added a regression test,
`test_agentv_sanitizes_node_options_for_the_runner_subprocess`, that asserts
the spawned subprocess env clears `NODE_OPTIONS` even when the parent process
sets it. Bumped `evals.agentv` to `v7` in `versions.json` with a history
entry. `pytest tests/test_evals/test_agentv.py`: 8/8 passed. The broader
`tests/test_evals/` suite has the same pre-existing failure/error set with
and without this change staged (67 failed on unmodified HEAD vs. 64 failed
with the fix — the fix's own tests moved from failing to passing; the
remainder is unrelated, pre-existing repo test debt).

Fix commit: `f8abb9d794559fc89812d32aeeea288a0ee730b4`.

**Next step:** replay the identical frozen control/bounds arms
(`retry_measurement`) now that the NODE_OPTIONS defect is repaired — no knob
changes, same `train_version=wf_smoke_v2`, `steps=20`, seed, and manifest.

Lean is `not_applicable:screening`; no ship or climb claim is made here.
Neither fixture checkpoint is promoted or synced.

Machine evidence:
[`autotrain-cycle-continuous-openui-local-c3-node-options-repair.json`](autotrain-cycle-continuous-openui-local-c3-node-options-repair.json).
