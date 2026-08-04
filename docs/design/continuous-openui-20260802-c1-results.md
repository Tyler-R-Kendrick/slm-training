# Continuous autotrain cycle 1 results (2026-08-02, resumed session)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260802` |
| Campaign | `continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c1` |
| Source | `f9bd228807cb05687de485ff159c975a864411fe` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Status |
| --- | --- |
| c1-control | trained; **eval harness_failure** |
| c1-bounds | trained; **eval harness_failure** |

No scoreboard was produced for either arm — `primary_metric` (`smoke.structural_similarity`) is
unavailable this cycle.

## Root cause and repair

This is a fresh scheduled-agent container resuming the `continuous-openui-20260802` loop after the
c2/c3/c4 cycles already documented earlier the same day. `outputs/autoresearch/` (loop state,
`node_modules/`) is gitignored and not carried between sessions, so the driver started a new
cycle-1 campaign and hit the **exact same harness_failure class already diagnosed and repaired in
the c2 cycle**: `scripts.evaluate_model --ship-gates` calls `publish_model_evaluation`, which
requires the pinned AgentV SDK (`node_modules/@agentv/core`). It was missing, so
`src/slm_training/evals/agentv.py:_agentv_runtime` raised `RuntimeError: AgentV SDK is
unavailable`, and both arms failed at evaluation.

Repair actions (`repair_harness`, family `model_build`, frozen manifest
`c229ddf96c4ded4b7b1491650261c1bf16a10c616e7298f40e429e02a9edeed7`):

1. Ran `npm ci` to restore the pinned SDK — this is the complete fix; no code change was needed.
2. Confirmed the v6 ship-gate fixture fix from the c2 cycle (commit
   `3cb06385d95769bbcde2e5b0b0e0f16138b1d3c8`) is already present on this commit, so
   `tests/test_evals/test_agentv.py` did not need re-fixing.
3. Separately found the session's exported `NODE_OPTIONS` (`"--import tsx"
   --max-old-space-size=8192`) is malformed — `node`/`npm` refuse it (`--import tsx is not allowed
   in NODE_OPTIONS`). Worked around with `env -u NODE_OPTIONS` for `npm`/`python` invocations this
   session. This is a session/container environment issue outside the repository, not a harness
   defect — no repo change made for it.

**Process note:** since `npm ci` (and, in this container, `pip install -e ".[dev]"` for a
Python 3.12 venv — the system interpreter was 3.11) must be repeated on every fresh scheduled
session, this bootstrap cost recurs every cycle in this environment. Flagged as a next-priority
process improvement below rather than fixed in-repo (this is a session-provisioning gap, not a
harness bug — `ci.yml` already runs `npm ci` before the AgentV-touching shard).

## Next-run priorities

1. **infrastructure:** replay `c20260802-continuous-openui-202608-39ee9cf7-c1-bounds` /
   `-control` now that `node_modules/@agentv/core` is restored (queued as `retry_measurement`).
2. **model:** resume the ranked matrix from the c4 cycle doc (size-matched `component-plan` lever)
   once the retry completes.
3. **process:** consider a repo-level session-start hook (see the `session-start-hook` skill) that
   runs `npm ci` and provisions a Python 3.12 venv automatically, so fresh scheduled containers
   don't re-pay this bootstrap cost every cycle.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c1/`
- Runs: `.../runs/c20260802-continuous-openui-202608-39ee9cf7-c1-control/`,
  `.../runs/c20260802-continuous-openui-202608-39ee9cf7-c1-bounds/`
- JSON twin: `continuous-openui-20260802-c1-results.json`
