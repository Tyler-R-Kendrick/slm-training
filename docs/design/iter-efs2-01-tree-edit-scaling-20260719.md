# EFS2-01: X22 beam-width × edit-depth scaling over valid program states (SLM-111)

**Linear issue:** SLM-111
**Branch:** `agent/slm-111-efs2-01-tree-edit-scaling`
**Date:** 2026-07-19
**Status:** wiring fixture / eval scaffolding; SLM-111 acceptance incomplete

Evidence: [iter-efs2-01-tree-edit-scaling-20260719.json](iter-efs2-01-tree-edit-scaling-20260719.json).
Harness: [`src/slm_training/evals/tree_edit_scaling.py`](../../src/slm_training/evals/tree_edit_scaling.py),
fixture runner: [`scripts/run_efs2_01_tree_edit_scaling_fixture.py`](../../scripts/run_efs2_01_tree_edit_scaling_fixture.py).
Tests: [`tests/test_evals/test_tree_edit_scaling.py`](../../tests/test_evals/test_tree_edit_scaling.py).

## What changed

Added eval-only wiring for the SLM-111 hypothesis that X22 test-time scaling
improves with beam width and edit depth before saturating, while preserving the
all-valid-states invariant.

- `src/slm_training/evals/tree_edit_scaling.py`
  - `TreeEditScalingConfig` — one decode cell with explicit `beam_width`,
    `max_edit_depth`, `expand_per_state`, `max_search_steps`, seed, and mode.
  - `BeamState` — canonical program text/fingerprint, cumulative edit depth,
    parent fingerprint, edit tuple, value score, frozen/STOP status, and valid
    flag.
  - `SearchTelemetry` — visited states, invalid edit attempts, duplicate prunes,
    beam size, steps, frozen count.
  - `EditRanker` protocol plus `RandomValueRanker` and `DeterministicRanker`.
  - `run_tree_edit_scaling_cell()` — value-guided beam search over
    `TreeEditSpace`, with hard edit-depth tracking and parser re-validation.
  - `run_scaling_grid()` — full 3×3 factorial grid over seeds and seed
    programs, version-stamped.
- `scripts/run_efs2_01_tree_edit_scaling_fixture.py`
  - Synthetic fixture running the `{1,4,16} × {1,2,4}` grid over three seeds
    and three small OpenUI seed programs.
- `tests/test_evals/test_tree_edit_scaling.py`
  - Regression tests for valid states, edit-depth ceiling, beam-width limit,
    duplicate removal, grid completeness, and JSON round-trip.
- `src/slm_training/resources/versions.json`
  - Bumped `evals.scoring` to `v5`.

## Fixture run

Command:

```bash
python -m scripts.run_efs2_01_tree_edit_scaling_fixture --run-id iter-efs2-01-20260719
```

Recipe: CPU; three synthetic seed programs; deterministic random value ranker;
`expand_per_state=4`, `max_search_steps=8`.

### Aggregate telemetry

| Cell | Runs | Visited | Invalid | Duplicates | Frozen | Steps |
| --- | --- | --- | --- | --- | --- | --- |
| b1_d1 | 9 | 18 | 0 | 9 | 0 | 18 |
| b1_d2 | 9 | 36 | 0 | 9 | 0 | 27 |
| b1_d4 | 9 | 72 | 0 | 10 | 0 | 45 |
| b4_d1 | 9 | 18 | 0 | 18 | 0 | 18 |
| b4_d2 | 9 | 54 | 0 | 40 | 0 | 27 |
| b4_d4 | 9 | 198 | 0 | 68 | 0 | 45 |
| b16_d1 | 9 | 18 | 0 | 18 | 0 | 18 |
| b16_d2 | 9 | 54 | 0 | 40 | 0 | 27 |
| b16_d4 | 9 | 270 | 0 | 192 | 0 | 45 |

Visited states grow with beam width and depth as expected; invalid edit
attempts are zero because every applied edit is re-verified by the parser and
rejected before entering the beam.

## Honest caveats

- **Wiring-only / no learned model.** The ranker is a deterministic random
  stand-in, not the trained X22 policy/value head.
- **Synthetic seed programs.** Real EFS2-01 needs the X22 training corpus and
  durable checkpoints.
- **No semantic metric.** The fixture reports search telemetry only; it does
  not run binding-aware meaningful-program evaluation or AgentV.
- **Random value ranker.** Width/depth trends here reflect combinatorial growth,
  not learned value quality.

## Verification checklist

- [x] `pytest tests/test_evals/test_tree_edit_scaling.py` — 8 passed.
- [x] `python -m scripts.run_efs2_01_tree_edit_scaling_fixture --run-id iter-efs2-01-20260719` — bundle written.
- [x] `python -m scripts.repo_policy` — ok.
- [x] `.githooks/check-changed` — 436 passed, 12 deselected.
- [x] `python -m scripts.verify_version_stamps --check` — ok.
- [x] `git diff --check` — clean.
