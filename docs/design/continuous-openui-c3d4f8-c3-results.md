# Continuous autotrain: 2026-08-05 (loop `continuous-openui-c3d4f8`) cycles 1-3 — autoresearch harness repair + null-delta screening result

**Loop:** `continuous-openui-c3d4f8`
**Campaign (completed):** `continuous-loop-20260805-continuous-openui-c3d4f8-986c6dc3-c3`
**Blocked campaigns:** `...-c1`, `...-c2`
**Integration commit:** `53812385` (harness repair, this session; `origin/main` tip at cycle start was `34111e6e`)

## Harness repair (cycles 1-2, hard blocker)

Cycles 1 and 2 both hard-failed (exit code 2) before any experiment could be
selected: every screening-role hypothesis in the proposed `HypothesisMatrix`
failed pydantic validation —

```
hypotheses.N.experiment.knobs.generate_batch_size
  Extra inputs are not permitted [type=extra_forbidden]
...
hypotheses
  Tuple should have at least 5 items after validation, not 0
```

Root cause: commit `01767f1` (#1433, "screening batch-size") added
`base["generate_batch_size"] = 1` baking for `role == "screening"` knobs in
`scripts/run_autotrain_continuous.py`, but never declared the field on
`ExperimentKnobs` (a `StrictModel` that forbids extra keys) or in
`DEFAULT_ALLOWED_KNOBS` in `src/slm_training/autoresearch/schemas.py`. Every
screening candidate — the continuous driver's default role — has carried this
knob since that commit landed, so **every** matrix in this family failed
closed with 0/56 hypotheses, well under the `min_hypotheses=5` floor.

This reproduced identically on cycle 2 (`blocker_fingerprint
0a501eb8a241c4ce8f7ca7ef4044a50c29be7b334c228e605c285641e60d424c`,
`blocker_count=2`). Per `continuous.md` rule 3, self-heal (thrash-bank knob
rotation) cannot repair a schema mismatch — it requires a code change, so this
routed to `improve-openui-harnesses` (owner: `src/slm_training/autoresearch/`,
single family `autoresearch`).

**Fix** (commit `5381238`): declare `generate_batch_size: int | None =
Field(default=None, ge=1, le=1024)` on `ExperimentKnobs` and add it to
`DEFAULT_ALLOWED_KNOBS`. Additive-only — no existing knob semantics changed.
Regression test:
[`tests/test_autoresearch/test_harness.py::test_generate_batch_size_knob_is_declared`](../../tests/test_autoresearch/test_harness.py).
Bumped `harness.autoresearch.experiment_campaign` `v180 -> v181`
(`src/slm_training/resources/versions.json`) per the version-stamp contract.
`python -m scripts.verify_version_stamps --check`, `python -m
scripts.refresh_test_cases --check --changed`, `python -m
scripts.repo_policy`, and `pytest -q tests/test_autoresearch` (268 passed) all
green before commit.

**Replay proof:** the identical arm (role=screening, `train_version=wf_smoke_v2`,
`steps=20`, seed-matched control + `both`) completed end-to-end with a usable
scoreboard on the very next supervised invocation (cycle 3, below) —
executable unblocking, replay-proven.

## Cycle 3 model result (screening, expected null delta)

| Arm | structural_similarity | parse_rate | binder_reference_f1 | meaningful_program_rate | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | .23083 | 1.0 | .73333 | .33333 | 13815.51 |
| both | .23083 | 1.0 | .73333 | .33333 | 14385.19 |

Primary (`smoke.structural_similarity`) delta `0.0` — control and the `both`
knob-rotation arm tie exactly. Ship gates fail as expected on fixture scale:
`insufficient_n` (n=3, need ≥20), `meaningful_program_rate` (.333 < .66),
`structural_similarity` (.231 < .35), `component_type_recall` (.167 < .35),
`ast_beq_rate` (0 < .2), `canonical_beq_rate` (0 < .1). `held_out`,
`adversarial`, `ood`, `rico_held` suites are `missing_suite` at this scale
(expected — screening cycles run `smoke` only).

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse` + `fixture_insufficient_n_alone`).
Per `sdlc` autotrain-iteration-delivery, no new stack layer opens for the
model result. The harness repair (commit `5381238`) is committed locally per
Phase A A1 (every green unit commits) but the cycle it unblocked did not
independently clear the positive-result bar (primary metric null, fixture
gate fail) — it is *not* pushed/PR'd as its own layer this session. See
report for the reasoning; a future cycle with a genuine metric win or
ship-quality clear should include this repair in that layer's PR.

## Next priorities

1. Screen the `component-plan` hypothesis next (rank 1, confidence 0.9) —
   distinct from the exhausted null `both` arm.
2. Keep the matched control fixed every cycle (rank 2, confidence 0.7).
3. Rotate thrash recommendation across the full lever bank, not bounds-only
   (rank 3, confidence 0.65).

Machine evidence:
[`continuous-openui-c3d4f8-c3-results.json`](continuous-openui-c3d4f8-c3-results.json).
