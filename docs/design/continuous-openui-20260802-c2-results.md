# Continuous autotrain cycle 2 results (2026-08-02)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260802` |
| Campaign | `continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c2` |
| Source | `b2b8bef22a82403a760373ab0eedc6dff9c4bb35` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Status |
| --- | --- |
| c2-control | trained (22 steps observed); **eval harness_failure** |
| c2-canvas | trained (22 steps observed); **eval harness_failure** |

No scoreboard was produced for either arm — `primary_metric` (`smoke.structural_similarity`) is
unavailable this cycle.

## Root cause and repair

`scripts.evaluate_model --ship-gates` calls `publish_model_evaluation`, which requires the pinned
AgentV SDK (`node_modules/@agentv/core`). The session's checkout was fresh and had not run
`npm ci`, so `src/slm_training/evals/agentv.py:_agentv_runtime` raised
`RuntimeError: AgentV SDK is unavailable`, and both arms failed at evaluation. CI's `ci.yml`
already runs `npm ci` ahead of the AgentV-touching shard, so this is an environment bootstrap gap
for a fresh interactive/agent session, not a repository code defect.

Repair actions (`repair_harness`, family `model_build`, frozen manifest
`6d6d01837abf80191cb5e81a786865dbf65f660b68c109c3154911acaaa05acc`):

1. Ran `npm ci` to restore the pinned SDK.
2. While validating the fix, found the AgentV smoke-suite ship-gate fixtures in
   `tests/test_evals/test_agentv.py` predated the v6 gate policy's `ast_beq_rate` /
   `canonical_beq_rate` floors — they would have produced a second, unrelated harness failure on
   replay. Fixed both fixtures (commit `7003208459c56fa6b26d555f103ba725b340fda6`); `pytest
   tests/test_evals/test_agentv.py` now passes 7/7.

## Known debt discovered (out of scope this cycle)

`tests/test_evals/` has **64 failing tests and 5 errors** on `origin/main` (`27a8134`), confirmed
pre-existing (reproduced against a clean checkout before any local edits). CI does not catch this
because `ci.yml`'s Python job only runs a changed-tests-only shard
(`scripts.check_changed --changed-tests-only`), not the full suite, on ordinary pushes. Affected
areas: `test_oracle_scoring_replay.py`, `test_operator_systems_benchmark.py`,
`test_semantic_fidelity.py`, `test_semantic_failure.py`, `test_meaningful_program.py`,
`test_metric_gaming.py`. Sample:

- `test_semantic_fidelity.py::test_ast_beq_true_for_style_normalized_match` — `ast_beq(...)`
  returns `False` for a pure-whitespace reordering that should compare equal.
- `test_oracle_scoring_replay.py::test_variant_rows_count_and_coverage` — `ValueError: persisted
  template markers must use opaque :slot_<ordinal> identities`.
- `test_operator_systems_benchmark.py::test_run_operator_systems_benchmark_end_to_end_small` —
  `OperatorAuthorityError: pack static/schema oracle rejected source`.

This needs its own `improve-openui-harnesses` repair pass; flagged here rather than fixed inline
to keep this cycle's harness repair narrowly scoped and replayable.

## Next-run priorities

1. **infrastructure:** replay `c20260802-continuous-openui-202608-39ee9cf7-c2-canvas` /
   `-control` now that the harness is repaired (queued).
2. **harness:** dedicated repair pass for the 64 pre-existing `tests/test_evals` failures,
   starting with the `ast_beq` whitespace-normalization bug.
3. **process:** consider a full (non-changed-only) scheduled test run so eval-harness regressions
   like this stop hiding between touches.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c2/`
- Runs: `.../runs/c20260802-continuous-openui-202608-39ee9cf7-c2-control/`,
  `.../runs/c20260802-continuous-openui-202608-39ee9cf7-c2-canvas/`
- JSON twin: `continuous-openui-20260802-c2-results.json`
