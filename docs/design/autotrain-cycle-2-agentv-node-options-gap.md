# Autotrain c2: AgentV `NODE_OPTIONS` eval-publication gap

**Verdict:** infrastructure/harness failure only — no model measurement.
Cycle 2 of loop `continuous-openui-20260802-local`. Training completed for
both arms (the c1 `torch` repair held), but evaluation failed for both
before producing any scoreboard.

## Result matrix

| Arm | Train | Eval | Disposition |
| --- | --- | --- | --- |
| control | completed, 1,608,962 params | Failed: AgentV SDK unavailable, then (post `npm ci`) `NODE_OPTIONS` rejection | Not scoreable; replay in c3 |
| bounds | completed, 1,608,962 params | Identical failure sequence | Not scoreable; replay in c3 |

## Diagnosis and repair

Two stacked infrastructure gaps, both fixed this cycle:

1. **Missing AgentV SDK.** Fresh container had no `node_modules`. Fixed
   locally with `npm ci` (pinned `@agentv/core@4.42.4` per
   `package-lock.json`); no code change.
2. **`NODE_OPTIONS` rejected by Node.** `publish_agentv_evaluation` in
   `src/slm_training/evals/agentv.py` spawned the AgentV node runner via a
   bare `subprocess.run(command, cwd=runtime_root, ...)` with no `env=`
   override, so it inherited this session's
   `NODE_OPTIONS="--import tsx" --max-old-space-size=8192`. Node rejects
   `--import` inside `NODE_OPTIONS`, so every `evaluate_model --ship-gates`
   run in an environment that sets this variable fails at eval publication
   — regardless of model, knob, or arm. This is the same class of gap
   already fixed once for the GraphQL-JS grammar bridge
   (`src/slm_training/dsl/grammar/backends/graphql_js.py:46-51`,
   `_sanitized_env()`).

   Fixed in commit `ca15f5c` on `claude/great-dirac-ptxx92`
   ([PR #1292](https://github.com/Tyler-R-Kendrick/slm-training/pull/1292)):
   added an equivalent `_sanitized_env()` to `agentv.py` and pass
   `env=_sanitized_env()` to the AgentV `subprocess.run` call.
   `evals.agentv` bumped v6 → v7.

   While re-running the tests touched by that file, two fixtures in
   `tests/test_evals/test_agentv.py` turned out to be stale relative to the
   current ship-gate policy: they predated the BEq metrics
   (`ast_beq_rate`, `canonical_beq_rate`, `meaningful_program_rate`) that
   `openui_ship_gates_v6.json` now requires for the `smoke` suite, so a
   "fully passing smoke suite" fixture was silently missing three required
   metrics. Completed both fixtures and corrected the expected pass/fail
   counts (`test_agentv_model_bundle_cannot_pass_a_smoke_only_run`:
   9 passed / 4 failed, up from a stale 7/4). Confirmed via `git stash` that
   these two failures pre-existed on unmodified `HEAD` and are unrelated to
   the `NODE_OPTIONS` change; the fix is a net improvement
   (75 failed / 663 passed → 73 failed / 681 passed across
   `tests/test_evals` + `tests/test_harnesses/model_build`, no new
   regressions).

## Disposition

Not a positive result on its own — no metrics were produced this cycle. Per
`autotrain-iteration-delivery.md`'s "executable unblocking" category, this
repair becomes positive once cycle 3 replays the **identical frozen** c2
arms and they complete with a usable scoreboard; that replay-proof is
recorded in the c3 doc, not claimed here.

Eval commit: `ca15f5c` (`evals.agentv=v7`).
