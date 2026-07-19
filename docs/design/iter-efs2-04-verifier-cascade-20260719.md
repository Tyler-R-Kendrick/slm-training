# EFS2-04: Cached cheap-to-expensive verifier cascade (SLM-115)

**Linear issue:** SLM-115
**Branch:** `agent/slm-115-efs2-04-verifier-cascade`
**Date:** 2026-07-19
**Status:** wiring fixture / cascade scaffolding; SLM-115 acceptance incomplete

Evidence: [iter-efs2-04-verifier-cascade-20260719.json](iter-efs2-04-verifier-cascade-20260719.json).
Harness: [`src/slm_training/evals/verifier_cascade.py`](../../src/slm_training/evals/verifier_cascade.py),
fixture runner: [`scripts/run_verifier_cascade_fixture.py`](../../scripts/run_verifier_cascade_fixture.py).
Tests: [`tests/test_evals/test_verifier_cascade.py`](../../tests/test_evals/test_verifier_cascade.py).

## What changed

Added eval-only verifier-cascade scaffolding for the EFS2-04 hypothesis that
lexical/grammar/schema checks eliminate most invalid candidates cheaply and that
caching + deferred expensive checks preserve sound pruning at lower cost.

- `src/slm_training/evals/verifier_cascade.py`
  - `Verdict` enum: `PASS | FAIL | UNKNOWN | ERROR | NOT_APPLICABLE`.
  - `VerifierStageSpec` — serializable stage contract with stage/version ID,
    dependencies, contract hash, `sound_fail`, cache policy, reason schema, and
    downstream stages to skip on a sound failure.
  - `VerifierResultV1` — per-stage outcome with status, sound flag, reason,
    certificate placeholder, cost, and cache flags.
  - `VerifierCache` — content-addressed in-run cache keyed by source, stage
    version, contract hash, pack version, and sorted context/environment.
  - `VerifierCascade` — runs stages in order, short-circuits only on sound
    `FAIL`, never treats `UNKNOWN`/`ERROR` as rejection, and does not cache
    `ERROR` results as proof. Provides `evaluate()` (cascade) and
    `evaluate_flat()` (authoritative flat-stack baseline) for matched-cost
    comparison.
  - `make_gate_stage()` and `default_openui_cascade()` — wrap the existing
    G0-G12 gate stack (`_lexical`, `_grammar`, `_schema`, `_reference_graph`,
    `_dataflow`, `_canonical`) as cheap-to-medium cascade stages.
- `scripts/run_verifier_cascade_fixture.py`
  - Synthetic fixture using the default gate cascade plus a mock expensive
    semantic stage (`cost_hint=100`). Demonstrates early sound-fail pruning,
    stage skipping, and cache-hit savings.
- `tests/test_evals/test_verifier_cascade.py`
  - Regression tests for sound-fail pruning, `UNKNOWN`/`ERROR` continuation,
    cache hit/miss semantics, version-key invalidation, unsound-fail handling,
    default gate cascade behavior, and JSON serialization.
- `src/slm_training/resources/versions.json`
  - Bumped `evals.scoring` to `v2`.

## Fixture run

Command:

```bash
python -m scripts.run_verifier_cascade_fixture --run-id fixture-20260719
```

Recipe: CPU; synthetic OpenUI programs; no checkpoint load; abstract cost hints.

### Cascade vs flat cost summary

| metric | value |
| --- | --- |
| candidates | 6 |
| cascade total cost | 219.0 |
| flat total cost | 530.0 |
| cost ratio | 0.413 |
| expensive calls | 3 |
| expensive skipped | 3 |
| cache entries | 21 |
| cache hits | 7 |

### Per-candidate cascade status

| candidate | pruned | cost | final status |
| --- | --- | --- | --- |
| empty | true | 1.0 | FAIL |
| unclosed | true | 2.0 | FAIL |
| duplicate_binder | true | 4.0 | FAIL |
| valid_minimal | false | 106.0 | PASS |
| bad_contract_marker | true | 106.0 | FAIL |
| valid_minimal_repeat | false | 0.0 | PASS |

The `valid_minimal_repeat` row costs 0 because every stage hit the cache warmed
by `valid_minimal`. The three cheap-fail candidates never reach the expensive
semantic stage.

## Honest caveats

- **Wiring-only / no checkpoint loaded.** No durable frontier checkpoint or live
  compiler decode was run.
- **Mock expensive stage.** The `semantic` stage is a substring marker, not a
  real structured-contract verifier.
- **Abstract cost.** Cost hints are relative units, not measured wall time; a
  production run must instrument actual verifier latency.
- **No Pareto frontier.** The 95%/30% target is not claimed; the fixture only
  proves that the cascade can skip expensive work and reuse cached results.
- **Not the default stack.** Existing decode/eval paths are unchanged.

## Verification checklist

- [x] `pytest tests/test_evals/test_verifier_cascade.py` — 10 passed.
- [x] `python -m scripts.run_verifier_cascade_fixture --run-id fixture-20260719` — bundle written.
- [x] `python -m scripts.repo_policy` — ok.
- [x] `.githooks/check-changed` — 412 passed, 12 deselected.
- [x] `python -m scripts.verify_version_stamps --check` — ok.
- [x] `git diff --check` — clean.
