# Continuous autotrain loop gd37mit6 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801-g37mit6` |
| Source | `0199664091cdb9dfd0f8af49edc91ef925db7046` (top of open stack, PR #1262) |
| Device | CPU |
| Steps | 20 / batch 2 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes per run |

## Cycle 1-2: harness_failure -> executable unblocking (POSITIVE)

The scheduled session's container is a fresh checkout with no pre-installed
Python/Node toolchain. Every experiment arm failed closed before reaching a
scoreboard:

1. **c1** — `ModuleNotFoundError: No module named 'torch'`. The bare venv had
   no torch/onnxruntime/transformers. Fixed by installing `torch==2.5.1+cpu`
   (matching the CI-pinned wheel), `onnxruntime>=1.18,<2`, and `transformers`.
2. **c2** — `RuntimeError: AgentV SDK is unavailable`. `node_modules/@agentv/core`
   was missing until `npm ci` at repo root. After installing it, the identical
   arm still crashed: the sandbox's session-level `NODE_OPTIONS` env var is a
   malformed `"--import tsx" --max-old-space-size=8192` (literal quotes
   included), which this Node build rejects with exit 9, silently killing the
   AgentV runner child process spawned from
   `publish_agentv_evaluation`'s `subprocess.run`.

   Fix: `src/slm_training/evals/agentv.py` now sanitizes `NODE_OPTIONS=""` in
   the child env before spawning `node`, mirroring the existing
   `graphql_js.py` bridge sanitization (`_sanitized_env()`).

   While isolating this, the repo's own `tests/test_evals` +
   `tests/test_harnesses/model_build` suites (swept in by touching
   `src/slm_training/evals/`) surfaced a **pre-existing, unrelated** bug:
   `tests/test_evals/test_emptiness_probe.py`'s `HERO`/`CTA` fixtures used
   named markers (`:hero.title`, `:hero.body`, `:cta.label`) instead of the
   opaque `:slot_N` identities `assert_canonical_template_marker_inventory`
   has required since #1167 (2026-07-27). Verified via `git stash` bisection
   that this fails identically without any of this cycle's changes, and that
   CI is green on the base commit only because no PR since #1167 has touched
   `src/slm_training/evals/` widely enough to sweep this suite in. Fixed the
   fixture's markers; all 5 tests in the file now pass.

   **Not fixed (documented, out of scope for this cycle):** the same
   `:slot_N` migration debt exists at much larger scale in
   `src/slm_training/evals/metric_gaming.py`'s `_archetypes()` (used by
   `oracle_scoring_replay.py`) — ~113 non-canonical marker occurrences across
   7 archetypes, 5 derived adversarial-variant dicts, and their prompt text.
   A same-shaped attempt to fix just the archetype `positive`/`slot_contract`
   fields regressed `test_metric_gaming.py` (21 new
   `binding_aware_meaningful_v2` false negatives) because the semantic scorer
   also correlates prompt-text marker mentions against the openui markers —
   the prompt strings need consistent, coordinated updates too. Reverted.
   Also observed but not diagnosed: `tests/test_harnesses/model_build/test_v4_levers.py::test_generate_batch_requests_consumes_harness_slot_contract`
   fails deterministically in this CPU container (`packed completion session
   rejected an advertised decode token`) while CI is green on the same
   commit — likely floating-point non-reproducibility between this
   container's torch/BLAS build and the CI runner's, not a code defect.
   Follow-up: route both through `improve-openui-harnesses`.

**Classification:** POSITIVE (executable unblocking, SDLC delivery law #3) —
the identical c3 arm spec now completes end-to-end with a usable scoreboard,
which it could not do before this cycle.

## Cycle 3: screening run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 | Ship gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c3-control | `grammar_completion_bounds=off` | 3 | 1.0 | 0.0 | 0.1725 | 8745.39 | **fail** (insufficient n + quality) |
| c3-both | `grammar_completion_bounds=on` | 3 | 1.0 | 0.0 | 0.1725 | 8551.94 | **fail** (insufficient n + quality) |

Primary metric delta (control - bounds) p50 latency: **193.45 ms** improvement,
but `mpr=0.0 < 0.333` — rejected as a pure latency blip per
`_classify_metric_tradeoff` (latency wins require held mpr ≥ ~1/3).
`meaningful_program_rate=0.0` on both arms at `n=3` is expected fixture-scale
noise, not a quality signal.

**Classification:** NON_POSITIVE (`fixture_insufficient_n`,
`latency_win_rejected_low_mpr`). Local commits + docs only; no new stack
layer for this cycle per the delivery law.

## Next-run priorities

1. **harness (evals):** finish the `:slot_N` canonicalization migration in
   `metric_gaming.py`/`oracle_scoring_replay.py`, updating archetype
   `positive` DSL, `slot_contract`, prompt text, and all 5 derived variant
   dicts together so `binding_aware_meaningful_v2` semantics stay correct.
2. **harness (model_build):** root-cause `test_v4_levers`'s
   `packed completion session rejected` failure — confirm floating-point
   drift vs a real grammar/decode defect before dismissing as environment-only.
3. **model:** re-run c3-shaped arms at higher `steps`/larger `train_version`
   within the wall cap now that the pipeline is unblocked, to get past
   `insufficient_n` before judging the bounds lever.

## Artifacts

- Campaigns: `outputs/autoresearch/continuous-loop-20260801-c{1,2,3}/`
- Runs: `.../continuous-loop-20260801-c3/runs/c20260801-c3-{control,both}/`
- JSON twin: `continuous-openui-20260801-g37mit6-results.json`
