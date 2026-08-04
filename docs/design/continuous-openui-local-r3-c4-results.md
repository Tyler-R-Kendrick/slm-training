# Continuous autotrain: 2026-08-04 (loop `continuous-openui-local-r3`) cycle 4 — node_modules staleness blocked measurement, now repaired

**Loop:** `continuous-openui-local-r3`
**Campaign:** `continuous-loop-20260804-continuous-openui-local--c8650581-c4`
**Integration commit:** `78cf357d` (`origin/main` tip `34111e6e`)

**Verdict:** non-positive; measurement incomplete on both arms. A real
harness bug blocked evaluation and has since been repaired; the identical
frozen arm is queued for replay.

## What happened

This is the first cycle of a fresh loop (`continuous-openui-local-r3`) on a
freshly provisioned session/container. Two schema/allowlist bugs in
`src/slm_training/autoresearch/schemas.py` blocked cycles 1–3 before any
experiment could even be hypothesized (fixed by commit, see below); this is
the first cycle where an experiment actually ran.

### Prerequisite schema/allowlist fixes (cycles 1–3, no run occurred)

`run_autotrain_continuous.py`'s screening role bakes
`base["generate_batch_size"] = 1` onto every hypothesis so tiny smoke suites
decode one record per chunk instead of one baked batch. `ExperimentKnobs`
(strict, `extra="forbid"`) never declared that field, and separately
`DEFAULT_ALLOWED_KNOBS` never listed it either — so every screening-role
hypothesis failed `validate()` before a campaign could ever run. Fixed in two
commits:

- `harness.autoresearch.experiment_campaign` v180 → v181: declare
  `generate_batch_size` on `ExperimentKnobs` (paired with its sibling
  `decode_timeout_seconds`).
- v181 → v182: add `generate_batch_size` to `DEFAULT_ALLOWED_KNOBS` (the
  schema fix alone was not sufficient — `validate_experiment()` separately
  rejects any changed knob absent from the campaign's allowlist).

Both fixes have a regression test
(`tests/test_autoresearch/test_harness.py::test_generate_batch_size_is_a_valid_measurement_knob`).

### Cycle 4 harness bug (this cycle)

With the schema fixed, cycle 4 hypothesized, validated, and ran two arms
(`control`, `steps`). Training completed for both, but evaluation failed:

- **control** (`...c4-control`, exit=2): `scripts.evaluate_model --ship-gates`
  crashed inside `publish_agentv_evaluation` with
  `ERR_MODULE_NOT_FOUND: node_modules/typebox/build/typebox.mjs` (imported by
  `node_modules/typebox/build/index.mjs`).
- **steps** (`...c4-steps`, exit=124): hit the experiment wall-seconds cap
  (`run_bounded_process` timeout / `KeyboardInterrupt`) — a soft timeout, not
  evidence of a lever regression, and expected on this CPU-only fixture host.

**Root cause:** the repo-root `node_modules` was stale/mismatched against the
committed `package-lock.json` — the installed `typebox` resolved to `1.3.6`
but was missing `build/typebox.mjs`, which that version's `index.mjs`
imports. `package-lock.json` itself was untouched by the repair, confirming
this was environment staleness (a fresh-container `node_modules` that
predated the currently pinned dependency tree), not a lockfile or
dependency-version bug.

**Repair:** `NODE_OPTIONS= npm ci` at the repo root. `NODE_OPTIONS` had to be
explicitly cleared for the command because the ambient `NODE_OPTIONS` value
in this container is a malformed literal-quoted string
(`"--import tsx" --max-old-space-size=8192`, quotes included as literal
characters) that breaks every node/npm invocation outright — the same
workaround `continuous.md`'s prerequisite `npm ci` steps for
`src/apps/openui_bridge` and `src/apps/design_md_bridge` also needed. No
tracked file changed by this repair (node_modules is gitignored); this
document is the durable evidence for the `repair_harness` handoff action.

## Results (partial — measurement incomplete, not an honest comparison)

| Arm | exit | train | eval | parse_rate | mpr | structural_similarity | binder_reference_f1 | latency p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 2 | completed | crashed pre-repair (harness bug) | 1.0 | 0.3333 | 0.41667 | 0.95238 | 27562.0 |
| steps | 124 | wall_timeout | not run | 1.0 | 0.3333 | 0.51 | 0.82222 | 18456.8 |

These numbers come from partial telemetry captured before each arm's
respective failure (control's crash happened after training but during eval
publish; steps timed out mid-run). They are **not** a completed
`scoreboard.json` for either arm and must not be read as a control/candidate
win — `cycle_handoff.json` and `sdlc_delivery.json` both flag
`measurement_incomplete:...:missing_scoreboard` for both arms. Recorded here
for diagnostic continuity only.

## SDLC Phase A

**Non-positive** (`measurement_incomplete` on both arms +
`harness_failure:control:experiment_failed`). No stacked PR layer for this
cycle's training result; the harness fixes (v181/v182) that unblocked
hypothesize/validate already shipped as their own incremental commits since
they are infrastructure fixes, not model results.

## Next priorities

1. (rank 1, confidence 0.95, infrastructure) Replay the exact frozen arm
   (`frozen_manifest_sha256`
   `ec197e9107e56d008ae1bf461b5b17794f4c3edd17c9fde8b79b6d3b712651b4`,
   experiment `c20260804-continuous-openui-local--c8650581-c4-steps`) now
   that the harness is repaired, before testing a new hypothesis.
