# Continuous autotrain loop `continuous-openui-20260801-gd1b0kaj` — cycles c1–c3

JSON: [`continuous-openui-20260801-gd1b0kaj-c1-c3-results.json`](continuous-openui-20260801-gd1b0kaj-c1-c3-results.json)

Honesty: `fixture_or_scratch`. **Not a ship claim.** Recipe: CPU, scratch
context backend, `train_version=wf_smoke_v2`, `eval_version=e938_role_safe_all_targets_v2`,
suite `smoke` (n=3), 20 train steps, `MAX_RUN_MINUTES`-capped stages,
`--ship-gates` on. All three cycles: `positive=False`, `stack_layer=False`
(sdlc autotrain-iteration-delivery — this doc is local-commit-only, no stack
layer opened for any cycle).

## Cycle 1 (`continuous-loop-20260801-c1`) — control vs `grammar_completion_bounds`

| Arm | n | parse_rate | mpr | structural_similarity | latency_ms_p50 |
| --- | --- | --- | --- | --- | --- |
| c1-control | 3 | 1.0 | 0.0 | 0.0575 | 3683.87 |
| c1-bounds | 3 | 1.0 | 0.0 | 0.0575 | 3847.99 |

`smoke.latency_ms_p50` delta = **-164.12ms** (worse, not a win). `parse_rate`
held at 1.0; `meaningful_program_rate` stayed 0.0 for both arms at this
fixture scale — no quality signal to trade against the latency regression.
**Conclusion: `grammar_completion_bounds` did not improve smoke latency on
this size-matched 20-step fixture; non-positive.**

Both arms additionally **hard-crashed** at the AgentV publish step
(`evaluate_model --ship-gates` exit 1, before `evaluate_model` normally exits
2 on a gate fail) — see harness signal below. The metrics above were captured
by the driver before the crash truncated the `evals` block of the scoreboard.

## Cycle 2 (`continuous-loop-20260801-c2`) — control vs `compact_active_canvas`

| Arm | n | decode_timeout_count | parse_rate | mpr |
| --- | --- | --- | --- | --- |
| c2-control | 3 | 3 | — | — |
| c2-canvas | 3 | 3 | — | — |

All 3 smoke-suite decodes hit the 24s decode timeout in **both** arms —
`primary_metric_unavailable`, `fixture_insufficient_n`. **Conclusion:
`compact_active_canvas` produced no usable signal; this is a decode-timeout
wall-clock ceiling at this fixture scale, not evidence for or against the
lever.**

## Cycle 3 (`continuous-loop-20260801-c3`) — control vs `both` (bounds + canvas combined)

| Arm | n | parse_rate | mpr | structural_similarity | latency_ms_p50 |
| --- | --- | --- | --- | --- | --- |
| c3-control | 3 | 1.0 | 0.0 | 0.1725 | 10099.38 |
| c3-both | 3 | 1.0 | 0.0 | 0.1725 | 10328.97 |

`smoke.latency_ms_p50` delta = **-229.59ms** (worse again). Ran clean end to
end with no AgentV crash (the environment fix from cycles 1–2 held).
`meaningful_program_rate` still 0.0 for both arms; `structural_similarity`
moved to 0.1725 for both (vs. 0.0575 in cycle 1) — same for control and
candidate, so this is checkpoint/seed variance across cycles, not a lever
effect. **Conclusion: combining `grammar_completion_bounds` +
`compact_active_canvas` does not improve smoke latency either; non-positive,
consistent with cycles 1–2.**

## Harness signal: AgentV SDK publish crash under inherited `NODE_OPTIONS` (reconfirmed, not landed here)

Cycle 1's crash reproduces a signal a concurrent session already diagnosed
in full on open PR [#1254](https://github.com/Tyler-R-Kendrick/slm-training/pull/1254)
(`docs/design/autotrain-continuous-loop-blockers-20260801.md`, not yet merged
to `main`): `publish_agentv_evaluation` (`src/slm_training/evals/agentv.py`)
spawns `node scripts/run_agentv_eval.mjs` without sanitizing the environment,
and this execution environment's shell carries `NODE_OPTIONS="--import
tsx" --max-old-space-size=8192`, which makes plain `node` refuse to start.
This checkout also had never run `npm ci`, so `node_modules/@agentv/core`
was additionally missing.

This session:

1. Ran `npm ci` once (`NODE_OPTIONS=""` in-shell) — installed
   `node_modules/@agentv/core@4.42.4` cleanly.
2. Exported `NODE_OPTIONS=""` for the remainder of the session's shell before
   re-running cycle 2 — cycle 2's `evaluate_model --ship-gates` runs
   completed the AgentV publish step and produced full `gates.json`
   scoreboards (fixture-scale gate fails, not crashes).
3. **Replayed the identical cycle-1 `c20260801-c1-control` `evaluate_model`
   command** (same args, same `run-root`/`run-id`/checkpoint) with the
   sanitized environment: it now completes and writes
   `outputs/autoresearch/continuous-loop-20260801-c1/runs/c20260801-c1-control/gates.json`
   instead of raising `RuntimeError: AgentV SDK is unavailable`. This
   satisfies "executable unblocking" (identical arm, replay-proven) — but see
   below for why the *code* fix still isn't landed.
4. Applied the exact `env.pop("NODE_OPTIONS", None)` patch text from PR
   #1254's doc to `src/slm_training/evals/agentv.py` locally and reran
   `tests/test_evals/test_agentv.py`: **2 of the same 4 previously-failing
   cases now pass** (`test_publish_agentv_evaluation_uses_sdk_and_jsonl`,
   `test_agentv_contract_checks_fail_even_when_pass_flag_is_true`) — matches
   PR #1254's own finding exactly.
5. Confirmed `python -m scripts.check_changed` maps `evals/agentv.py` changes
   to the full `tests/test_evals` + `tests/test_harnesses/model_build`
   directories, and **9 tests in that scope still fail on this exact commit**
   for two reasons unrelated to `NODE_OPTIONS` (reconfirmed today, list in
   the JSON): a `dsl` grammar-backend serialization gap
   (`LarkBackend.serialize()` doesn't whitespace-normalize) and a
   `model_build` `lever_capability_compatibility` validation error.
6. **Reverted the code edit.** Landing it in this PR would flip
   `check_changed` red for reasons this PR did not introduce and is not
   scoped to fix (core `dsl` is not one of the nine owned harness families;
   the `model_build` failure is a separate pre-existing validation gap).
   This session's own cycles stayed unblocked via the session-local
   `NODE_OPTIONS=""` export + one-time `npm ci` — no tracked code change was
   needed to make cycle 2 produce real scoreboards.

**Action for a future cycle:** land PR #1254's fix once the `dsl` and
`model_build` blockers above are repaired (see `next_priorities` in the
JSON); do not re-diagnose from scratch, the patch text is already correct
and reconfirmed twice now.

## SDLC Phase A

- Cycle 1: `positive=false`, `stack_layer=false`, reasons: empty metrics from
  the truncated crash + `primary_metric_null_or_worse` (latency regressed).
- Cycle 2: `positive=false`, `stack_layer=false`, reasons:
  `fixture_insufficient_n` + `primary_metric_unavailable` (decode timeouts).
- Cycle 3: `positive=false`, `stack_layer=false`, reasons:
  `fixture_insufficient_n` + `primary_metric_null_or_worse` (latency
  regressed again, combined levers).

Non-positive cycles do not open stacked PRs (sdlc autotrain-iteration-delivery).
This PR is docs-only, local commits, no `gh stack` layer.

## Next-run priorities (updated after cycle 3)

Three consecutive cycles at this fixture scale (n=3, 20 steps) show
`meaningful_program_rate` pinned at 0.0 regardless of lever combination
(`bounds`, `canvas`, `both`) — the smoke fixture is not large/capable enough
to discriminate between these levers on quality, only on latency, and every
latency delta so far has been a regression. Recommend either: (a) scale the
next screening cycle's `steps`/checkpoint past this smoke floor so `mpr`
starts moving, or (b) rotate off `grammar_completion_bounds` /
`compact_active_canvas` entirely toward an untested lever from
`allowed_knobs` for the next cycle, since three arms have now agreed these
two are not latency wins at this scale.
