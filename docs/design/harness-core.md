# Harness core: frozen, DSL-agnostic machinery

Status: **active contract.** Owner: `src/slm_training/harness_core/`.
Registered as component `harness.core` in
`src/slm_training/resources/versions.json`.

## Why

The training harnesses mix two kinds of code: **frozen machinery** (version
stamping, immutable lineage records, the checkpoint reference schema, the
ship-gate and promotion decision engines, scaling-law math, eval bookkeeping)
and **actively iterated OpenUI-specific logic** (thresholds, metric names,
parsers, synthesis, decode optimizations). `harness_core` gives the frozen
machinery one owner so harness families can iterate on their DSL-specific
layers without copy-pasting the stable parts, and so a harness for a
different DSL can reuse the same core.

The extraction that created this package was **purely structural**: no
behavior, output byte, or public API changed
(`tests/test_harness_core/test_gate_engine_golden.py` pins the ship-gate
payload; the pre-existing test suite ran unchanged).

## Contract

1. **DSL-agnostic.** Modules under `harness_core` never import
   `slm_training.dsl`, `slm_training.models`, `slm_training.evals`,
   `slm_training.harnesses`, `slm_training.web`, `slm_training.autoresearch`,
   `slm_training.data`, `slm_training.runtime`, or
   `slm_training.integrations` at module level — enforced by
   `tests/test_harness_core/test_import_hygiene.py`. DSL- or metric-specific
   behavior enters through parameters and callbacks (see seams below). One
   grandfathered lazy exception is allowlisted in that test
   (`lineage/data_cycle.py` → `slm_training.data.store`, itself stdlib-only).
2. **Frozen.** Any change under `src/slm_training/harness_core/` must bump the
   `harness.core` component — or carry a same-version `no-bump:` history note —
   in `versions.json` (`docs/design/version-stamp-contract.md`). The generic
   gate-check loop `harness_core/gate_engine.py` is additionally watched by
   `gates.ship`: gate logic stays under gate discipline after the move.
3. **Import-light.** Importing `slm_training.harness_core` never pulls torch;
   heavy dependencies stay behind function-level imports (e.g.
   `lineage/merge.py`).
4. **Old paths stay valid.** Every pre-extraction import path is a shim that
   aliases the real module in `sys.modules`, so old and new paths resolve to
   the *same* module object (class identity and monkeypatching behave
   identically through either path). New code should import from
   `slm_training.harness_core`.

## What lives here (old → new)

| Old path | New path |
| --- | --- |
| `src/slm_training/versioning.py` | `harness_core/versioning.py` |
| `src/slm_training/lineage/` (9 modules) | `harness_core/lineage/` |
| `harnesses/model_build/checkpoint_reference.py` | `harness_core/checkpoint_reference.py` |
| `harnesses/experiments/efficiency_gain.py` | `harness_core/efficiency_gain.py` |
| `harnesses/experiments/scaling_fit.py` | `harness_core/scaling_fit.py` |
| `evals/record_schema.py` | `harness_core/record_schema.py` |
| `evals/eval_cache.py` | `harness_core/eval_cache.py` |
| `evals/score_policy.py` | `harness_core/score_policy.py` |
| (extracted from `model_build/ship_gates.py`) | `harness_core/gate_engine.py` |
| (extracted from `experiments/promotion.py`) | `harness_core/promotion_engine.py` |

## DSL seams (how harnesses bind their specifics)

- **Ship gates** — `gate_engine.run_gate_checks(suites, policy, *,
  normalize_suite, default_min_n)` owns the frozen check loop (missing-suite
  fail-closed, fallback certification, `min_n` evidence floor, per-metric
  threshold loop). `harnesses/model_build/ship_gates.py` remains the OpenUI
  policy owner: `DEFAULT_SHIP_GATES`, `MEANINGFUL_METRIC_POLICY`, the
  OpenUI slim-metric normalizer, payload assembly, and `gates.json` writing.
- **Promotion** — `promotion_engine.evaluate_promotion(...,
  hard_categories, gate_evaluator)` owns the frozen promotion checks;
  `harnesses/experiments/promotion.py` remains the policy owner
  (`HARD_CATEGORIES`, the DSL-touching `check_data_integrity`, the ship-gate
  binding) and keeps the original public signatures.
- A harness for a different DSL supplies its own normalizer, categories, and
  gate evaluator; the frozen loops are shared.

## Explicitly not moved (follow-ups)

- `evals/suite_sharding.py` — imports `dsl.schema.ExampleRecord`; needs a
  record-protocol seam first.
- `model_build/{checkpoint_bucket,checkpoint_path_manifest,decode_path}.py` —
  bound to the OpenUI bucket config and decode-path registry.
- `experiments/ladder.py`, `train_data/{report,feedback,integrity}.py`,
  `ModelBuildConfig` / `ModelPlugin` seams, `runtime/telemetry/`,
  `data/store.py`.
- Consolidating the five copy-pasted `ActivationGate`/`BudgetCap` manifest
  dataclasses and the repeated experiment `Arm`/`Result`/`Report` evidence
  convention across `experiments/` — behavior-affecting dedup; when it
  happens, the shared owner lands here.
- Retiring the old-path shims by rewriting call sites (mechanical cleanup).
- `Track = Literal["twotower", "causal_lm"]` stays in `lineage/records.py`
  (model naming, not DSL coupling).
