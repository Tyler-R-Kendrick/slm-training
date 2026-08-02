# Continuous autotrain cycle 1 results (2026-08-02, loop pq269u)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-pq269u` |
| Campaign | `continuous-loop-20260802-continuous-openui-pq269u-2619fa49-c1` |
| Source | `27a8134fbcf5def3ce8463a3059ace8445464d80` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Status |
| --- | --- |
| c1-control | trained; **eval harness_failure** (NODE_OPTIONS exit 9) |
| c1-bounds | trained; **eval harness_failure** (AgentV SDK unavailable) |

No scoreboard was produced for either arm — `primary_metric` (`smoke.structural_similarity`) is
unavailable this cycle.

## Root cause and repair

`scripts.evaluate_model --ship-gates` calls `publish_model_evaluation` →
`publish_agentv_evaluation`, which spawns the pinned AgentV SDK via `node`. Two stacked defects
blocked this in the fresh session checkout:

1. **Missing SDK.** `node_modules/@agentv/core` was absent (fresh checkout, `npm ci` never run),
   so `src/slm_training/evals/agentv.py:_agentv_runtime` raised
   `RuntimeError: AgentV SDK is unavailable`. A concurrent session already diagnosed and fixed
   this exact symptom on `origin/main` (commit `3cb0638`, PR #1327), alongside a stale
   `ast_beq_rate`/`canonical_beq_rate` ship-gate fixture bug in `tests/test_evals/test_agentv.py`
   that would otherwise have caused a second, unrelated failure on replay. This cycle merged that
   fix rather than re-deriving it.
2. **Unsanitized `NODE_OPTIONS`.** After restoring `node_modules`, `node -v` itself failed with
   `node: --import tsx is not allowed in NODE_OPTIONS` (exit 9) — this session's environment sets
   `NODE_OPTIONS="--import tsx" --max-old-space-size=8192`, which the pinned Node build rejects.
   `publish_agentv_evaluation`'s `subprocess.run(["node", ...])` call inherited this broken
   environment. The repo already has a fix for the identical class of failure in
   `src/slm_training/dsl/grammar/backends/graphql_js.py` (`_sanitized_env()`, which zeroes
   `NODE_OPTIONS` before spawning `node`); `agentv.py` did not yet have the equivalent guard. Added
   `_sanitized_node_env()` and passed it as `env=` to the runner subprocess (commit
   `a3ceb3ae9ada5ec40c35d889684d7e600a01633d`).

Reproduced the fix in isolation (`publish_agentv_evaluation` with a broken `NODE_OPTIONS` in the
environment) before and after the patch; `tests/test_evals/test_agentv.py` is 8/8 passing,
including a new regression test asserting the runner subprocess receives a sanitized
`NODE_OPTIONS=""`.

Repair actions (`repair_harness`, family `model_build`, frozen manifest
`0bb366d4e94dad5023ea7cd9177ccbe50c2a8cf873039efb4dda449f1c740683`):

1. Ran `npm ci` to restore the pinned SDK.
2. Added `_sanitized_node_env()` to `src/slm_training/evals/agentv.py` and wired it into the node
   runner `subprocess.run` call.
3. Merged `origin/main` (commit `3cb0638`) to absorb the concurrent `npm ci` / fixture repair
   instead of duplicating it (merge commit `b22185c3f9e9aaedc30b37e22d8c9622dd567a58`).

## Next-run priorities

1. **infrastructure:** replay `c20260802-continuous-openui-pq269u-2619fa49-c1-bounds` /
   `-control` now that both harness defects are repaired (queued).
2. **harness:** audit other `subprocess.run(["node", ...])` call sites for the same unsanitized
   `NODE_OPTIONS` gap `graphql_js.py` and `agentv.py` now guard against.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-pq269u-2619fa49-c1/`
- Runs: `.../runs/c20260802-continuous-openui-pq269u-2619fa49-c1-control/`,
  `.../runs/c20260802-continuous-openui-pq269u-2619fa49-c1-bounds/`
- JSON twin: `continuous-openui-pq269u-c1-results.json`
