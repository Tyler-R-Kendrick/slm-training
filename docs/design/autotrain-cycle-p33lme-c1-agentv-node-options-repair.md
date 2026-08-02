# Autotrain continuous-openui-p33lme c1: AgentV NODE_OPTIONS repair

**Outcome:** non-positive model measurement (fixture screening, incomplete
scoreboard); positive **harness unblock** — the AgentV SDK eval-publish step
now runs to completion in this environment instead of failing closed on every
suite. This cycle documents both, per the iron law.

## What happened

`python -m scripts.run_autotrain_continuous --loop-id continuous-openui-p33lme
--supervised --max-cycles 1 --train-version wf_smoke_v2 --steps 20` trained a
matched control/candidate pair
(`c20260802-continuous-openui-p33lme-489d3aa7-c1-control` /
`...-c1-bounds`, 1,608,962 trainable params each, CPU, `wf_smoke_v2`, 20
steps) and both decoded their full 3-document `smoke` suite
(`completed_document_n=3/3`, strict compiler-tree policy,
`slot_contract_constrained_decode=True`). Evaluation then raised before
writing `scoreboard.json`:

```
RuntimeError: AgentV SDK evaluation failed: node: --import tsx is not allowed in NODE_OPTIONS
```

## Root cause

`publish_agentv_evaluation` (`src/slm_training/evals/agentv.py`) spawns the
pinned `@agentv/core` SDK via `node scripts/run_agentv_eval.mjs …` with
`subprocess.run(..., env=None)` — i.e. it inherits this environment's
`NODE_OPTIONS`, which this session sets to `--import tsx …
--max-old-space-size=8192`. This Node 22 build refuses that flag inside
`NODE_OPTIONS` and exits 9 on *every* invocation, not just when the SDK is
missing. The identical guard already exists for the GraphQL bridge
(`src/slm_training/dsl/grammar/backends/graphql_js.py::_sanitized_env`) but
had not been applied to the AgentV runner. `node_modules/@agentv/core` was
also absent in this checkout (`npm ci` had not run), which is why the error
surfaced as "AgentV SDK is unavailable" on the first attempt — after `npm ci`
resolved the SDK, the NODE_OPTIONS exit-9 became the live blocker.

## Repair

- `src/slm_training/evals/agentv.py`: added `_sanitized_env()` (same shape as
  the GraphQL bridge's) and pass `env=_sanitized_env()` to the `node` child
  process in `publish_agentv_evaluation`.
- `tests/test_evals/test_agentv.py`: new regression test
  `test_agentv_runner_sanitizes_inherited_node_options` asserts
  `subprocess.run` receives `NODE_OPTIONS=""` even when the ambient
  environment sets `NODE_OPTIONS=--import tsx`.
- Two pre-existing fixtures in the same file
  (`test_model_ship_cases_fail_closed_on_missing_suites`,
  `test_agentv_model_bundle_cannot_pass_a_smoke_only_run`) had drifted from
  the current smoke ship-gate policy, which now also gates `ast_beq_rate` and
  `canonical_beq_rate`. The fixtures were missing those fields, so even a
  fully-passing smoke suite produced spurious `actual=None` assertions. Added
  `exact_match` / `canonical_beq_rate` to the fixture metrics (no gate
  weakened) and corrected the fully-passing-smoke expectation from
  `passed=7` to `passed=9`.
- `src/slm_training/resources/versions.json`: bumped `evals.agentv` v6 → v7.
- Commit: `a82eea405548cea5fcc486c3332823e2e3b008cf`.

## Evidence

- Before the fix: `node scripts/run_agentv_eval.mjs --help` → `node: --import
  tsx is not allowed in NODE_OPTIONS` (exit 9); `pytest -q
  tests/test_evals/test_agentv.py` → 4 failed / 3 passed (2 of those 4 were
  the node-invoking tests dying on exit 9).
- After the fix: `NODE_OPTIONS= node scripts/run_agentv_eval.mjs --help` and
  the sanitized-subprocess path both succeed; `pytest -q
  tests/test_evals/test_agentv.py` → 8 passed / 0 failed.
- `python -m scripts.verify_version_stamps --check` → ok (3 changed files, 1
  component touched).

## Model measurement (still non-positive, unrelated to the repair)

| Arm | latency p50 (ms) | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| control (`...-c1-control`) | 1281.78 | 1.0 | 0.0 | 0.0575 | 0.6333 |
| candidate (`...-c1-bounds`) | 1345.59 | 1.0 | 0.0 | 0.0575 | 0.6333 |

`smoke.structural_similarity` is identical between arms
(improvement = 0.0), and both arms' scoreboards were incomplete at
measurement time because of the AgentV blocker above — SDLC Phase A
classifies this `NON_POSITIVE` (`no_stack_layer_non_positive`):
`measurement_incomplete:...control:missing_scoreboard`,
`measurement_incomplete:...bounds:missing_scoreboard`,
`primary_metric_null_or_worse:smoke.structural_similarity:control=0.0575
candidate=0.0575 improvement=0.0`. No stacked PR is opened for this cycle;
the harness fix is delivered as a plain `fix(autotrain)` commit per
`autotrain-iteration-delivery.md`.

## Next hypotheses

1. Replay the identical frozen `control`/`bounds` arms
   (`frozen_manifest_sha256=99ba319681814a2c4cd5f7597f7220c88f05205ac48e96b18ff8bada38283399`)
   now that AgentV can publish, to get a complete scoreboard before trying a
   new model hypothesis (per `retry_measurement` in `cycle_handoff.json`).
2. `meaningful_program_rate=0.0` on both arms at `wf_smoke_v2`/20 steps is a
   screening-scale artifact, not evidence either arm is broken; do not
   over-interpret until the replay completes with a full scoreboard.
3. Confirm no other harness caller of `node` in this repo relies on an
   unsanitized ambient `NODE_OPTIONS` (only the GraphQL bridge had the guard
   before this cycle; the OpenUI lang-core REPL bridge degrades gracefully
   via `bridge_available()` and was not blocking, but is worth an audit).

Machine-readable evidence is in
[`autotrain-cycle-p33lme-c1-agentv-node-options-repair.json`](autotrain-cycle-p33lme-c1-agentv-node-options-repair.json).
