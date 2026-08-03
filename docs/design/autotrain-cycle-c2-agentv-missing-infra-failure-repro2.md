# Autotrain c2 (continuous-openui-local): AgentV SDK missing, reproduced in a second sandbox — still not a model result

**Verdict:** infrastructure failure, not scoreable — a second, independent
reproduction of the exact gap already recorded in
[`autotrain-cycle-c2-agentv-missing-infra-failure.md`](autotrain-cycle-c2-agentv-missing-infra-failure.md),
hit in a different fresh sandbox on the same calendar date. The loop's
campaign-ID hash is derived from `(loop_id, date)` only, so this run's
campaign id string is byte-identical to the prior finding's even though the
two are distinct runs with distinct checkpoint digests (`1bc6370f...286e`
control / `9f73b7a8...053b1a4` canvas here, vs `f2fe8f5a...b81b2` /
`2c1749c6...41742f64` there).

Training completed for both the `control` and `canvas` `wf_smoke_v2` arms
(1,608,962 params, 22 steps, loss `14.3902` for both, 101 records), but
`evaluate_model.py --ship-gates` crashed before producing a scoreboard for
either arm: `RuntimeError: AgentV SDK is unavailable; run npm ci in the
checkout or set AGENTV_RUNNER`. A non-ship-gate smoke eval did complete for
both arms and ties exactly (`structural_similarity=0.32666...`,
`meaningful_program_rate=0.0` for both) — not a model result either way.

**No source change was required.** The fix landed previously in commit
`72fdffaf054efee2f3dccc4bab0ca97c111072e1` (#1360): `evaluate_model.py`'s
AgentV runner already goes through `sanitized_node_env()`, and
`scripts/setup_dev_env.sh` already documents `env -u NODE_OPTIONS npm ci` as
the required bootstrap step. This sandbox simply hadn't run that step yet.
Ran `env -u NODE_OPTIONS npm ci` at the repo root, which installed
`node_modules/@agentv/core` and unblocked the runner.

No scoreboard, no ship-gate result exists for this cycle; the checkpoints are
local, explicit no-sync, and not reusable, promotable, or ship evidence. Lean
is `not_applicable:screening`.

Next: replay the identical frozen arm (`retry_measurement`) now that the
AgentV SDK is installed in this sandbox.

Machine evidence:
[`autotrain-cycle-c2-agentv-missing-infra-failure-repro2.json`](autotrain-cycle-c2-agentv-missing-infra-failure-repro2.json).
