# RSC-A06 (SLM-242): fail-closed numeric weight/schedule validation gate

Date: 2026-07-21
Issue: [SLM-242](https://linear.app) "RSC-A06: Add a fail-closed property gate
for every TwoTower weight and schedule vector" (team SLM, project "Recurrent
Semantic Computation & Looped-Latent Gates", milestone "RSC-A · Objective
correctness & reproducible controls").
Non-goal reminder: this change adds validation only. No scientifically
intended default weight or schedule was changed; every found behavior defect
below is documented, not fixed, here.

## Summary

Generalizes the SLM-138 recursive deep-supervision fix pattern into a
repository-wide fail-closed validation gate for TwoTower's numeric
weight/schedule config surface: typed validation primitives
(`slm_training.harness_core.schedule_validation`), a TwoTower-specific
capability matrix (`slm_training.models.twotower_schedule_policy`), wiring
into `TwoTowerConfig`/`ModelBuildConfig` construction, `apply_runtime_overrides`,
checkpoint save/load, a static AST guard for the source-shaped defect
patterns (`scripts/verify_numeric_schedule_guard.py`), and a
`NumericScheduleValidationReportV1` field inventory
(`docs/design/rsc-a06-numeric-schedule-validation-report-20260721.json`).

## Scope: what this audit actually covers

Per the issue's own instruction to prefer "a smaller, fully-correct,
honestly-scoped-and-documented slice" over a shallow full-repo pass, this
audit is scoped to:

- **Fully covered**: every numeric vector/schedule-shaped field declared on
  `TwoTowerConfig` (`src/slm_training/models/twotower.py`) and its mirror on
  `ModelBuildConfig` (`src/slm_training/harnesses/model_build/config.py`),
  found by grepping every `tuple[...]`-typed field in both dataclasses and
  reading every consumer of each in `twotower.py`, `factory.py`, and
  `checkpoint_migrate.py`. **10 fields**, listed in full in the JSON report.
- **Explicitly deferred** (not read/audited in this pass, listed honestly
  rather than silently skipped):
  - `src/slm_training/harnesses/quality/` curriculum/retrieval modules
    (curriculum stage weights, retrieval-bank scoring vectors).
  - `mixture_manifest`-driven quota **scalars** on `ModelBuildConfig`
    (`mixture_exposure_target_profile`, `mixture_total_decision_budget`,
    `mixture_per_root_cap`, `mixture_per_template_cap`,
    `mixture_max_importance_weight`) — not vectors, and the *contents* of an
    external `mixture_manifest` JSON file are not schema-validated here.
  - `grammar_diffusion.py` / `tree_edit_diffusion.py`'s own config surfaces
    (`topology_max_*`, `block_size`, `production_loss_weight`,
    `slot_loss_weight`, `confidence_loss_weight`, `scope_*`) — separate model
    families from TwoTower, out of scope per the issue's TwoTower framing.
  - ~30 individual `semantic_plan_*`/`schema_*`/`binder_*`/`root_reference_*`/
    `component_*` per-role **scalar** decode weights on `TwoTowerConfig` —
    each is conceptually in-scope ("a per-role weight") but has no vector
    shape or cross-field pairing to validate beyond finite/non-negative; a
    full per-field capability audit of all ~30 was not completed here.
  - `action_alias_manifest` / `targeted_margin_manifest` / `mixture_manifest`
    `Path` fields — only their presence/type is implied by the dataclass;
    external file *schema* validation is out of scope.
  - `checkpoint_path_manifest.py` / `decode_path.py` — read during this
    audit; they contain no vector/schedule-shaped config fields, so there was
    nothing to add.

## What was built

### 1. Typed validation primitives — `src/slm_training/harness_core/schedule_validation.py`

All 11 primitives named in the issue (`finite_scalar`, `non_negative_scalar`,
`positive_scalar`, `exact_length_vector`, `non_empty_vector`,
`positive_sum_vector`, `normalized_probability_vector`,
`strictly_increasing_sequence`, `paired_equal_length_sequences`,
`unique_enum_sequence`, `supported_capability_requirement`), plus the one
typed error class `ScheduleValidationError(field, reason)`. Pure stdlib, no
torch, DSL-agnostic — placed in `harness_core` because that package is
explicitly "the stable contracts every harness builds on"
(`src/slm_training/harness_core/__init__.py`'s own docstring) and is enforced
torch-free/DSL-agnostic by `tests/test_harness_core/test_import_hygiene.py`,
which this module satisfies unchanged.

### 2. TwoTower capability matrix — `src/slm_training/models/twotower_schedule_policy.py`

One rule function per field (`validate_recursive_depth_supervision_weights`,
`validate_grammar_ltr_stages`, `validate_diffusion_policies`,
`validate_diffusion_length_buckets`, `validate_diffusion_length_loss_weight`,
`validate_mask_range`, `validate_targeted_margin_family_weights`,
`validate_slot_component_class_weights`, `validate_slot_component_priors`)
plus `validate_twotower_numeric_schedule(cfg)` that runs all of them. Every
rule reads its fields via `getattr(cfg, name, <real TwoTowerConfig default>)`
so it works against `TwoTowerConfig`, `ModelBuildConfig`, or a legacy/partial
namespace identically (the migration contract — see below). Deliberately
placed in `models/` (not `harnesses/model_build/`) because `twotower.py`
already imports from `harnesses/model_build/plugin` at module level, and
`ModelBuildConfig` imports this module back via a function-level import in
`__post_init__` — this module is pure stdlib (no torch), so it does not pull
torch into `ModelBuildConfig`'s otherwise-light import.

### 3. Validation timing (wired call sites)

| Timing (issue requirement) | Call site |
| --- | --- |
| Parsing/building `ModelBuildConfig`/`TwoTowerConfig` | `ModelBuildConfig.__post_init__` (`harnesses/model_build/config.py`); `TwoTowerConfig.__post_init__` (`models/twotower.py`) — fires on every construction, including `from_records` and `from_checkpoint`'s `TwoTowerConfig(**kwargs)` |
| After hierarchical/CLI overrides resolved | `apply_runtime_overrides` (`harnesses/model_build/factory.py`) re-validates `model.config` after every `setattr` override, scoped to `isinstance(cfg, TwoTowerConfig)` |
| Before checkpoint compatibility fingerprints/manifests are emitted | `TwoTowerModel.save()` validates `self.config` before writing `.pt`/`.meta.json` |
| Before training/evaluation dispatch | Covered transitively: both `scripts/train_model.py` and `scripts/evaluate_model.py` construct `ModelBuildConfig(...)` (triggering `__post_init__`) before any remote-job dispatch or training/eval loop starts |
| Checkpoint load requiring migration | `TwoTowerModel.load()` re-validates `self.config` after restoring `slot_component_lexeme_priors`/`slot_component_span_priors` from the checkpoint payload; `from_checkpoint` re-validates transitively via `TwoTowerConfig(**kwargs)` |

An invalid configuration now fails at construction time, which in every real
call path precedes remote-job allocation and evidence-artifact writes.

### 4. Property-based test suite

`hypothesis` is **not** a repo dependency (checked `pyproject.toml` and
`uv.lock` — absent, and not importable in the project venv), so per the
issue's fallback instruction these are hand-rolled `pytest.mark.parametrize`
table-driven tests instead of adding a new dependency:

- `tests/test_harness_core/test_schedule_validation.py` — 86 cases covering
  every primitive: valid vectors of varying length/scale, empty/all-zero/
  negative/NaN/inf/duplicate/unsorted/mismatched inputs, the
  positive-uniform-rescaling invariant for `normalized_probability_vector`,
  and one parametrized sweep proving every primitive raises the same
  `ScheduleValidationError` type with a populated `.field`/`.reason`.
- `tests/test_models/test_twotower_schedule_policy.py` — capability-matrix
  and integration tests against `ModelBuildConfig`, `TwoTowerConfig`,
  `TwoTowerModel`, and `apply_runtime_overrides` (torch-gated section uses
  the repo's standard `pytest.importorskip("torch")` convention).
- `tests/test_scripts/test_verify_numeric_schedule_guard.py` — the static
  guard's own unit tests plus two full-repo-scan regression tests.

### 5. Static guard — `scripts/verify_numeric_schedule_guard.py`

A narrow AST guard scanning `src/slm_training/models/` and
`src/slm_training/harnesses/model_build/` for three source-shaped patterns:

- `TRUNCATE`: `min(len(a), len(b))` (both arguments literal `len(...)` calls).
- `UNGUARDED_SUM`: `total = sum(x)` followed by `if total > 0:` in the same
  function.
- `UNUSED_LOOP_WEIGHT`: a `for` loop binds a weight-shaped variable name
  (`weight`/`^w$`/`_w$`) that is never read in the loop body.

The fourth section-5 pattern ("capability guards that silently fall back")
is **not** attempted as a generic AST rule — it is covered far more precisely
by the typed capability matrix (§2) than any text/AST heuristic could manage
without broad false positives across unrelated `if X and hasattr(...)` code
in the repo. This is stated explicitly in the guard's own module docstring
rather than left as a silent scope gap.

Supports a documented suppression comment on the hit line or the line
immediately above:
`# schedule-guard: allow <PATTERN> reason=<text> test=<path::test>`.

**Run against the real repo (80 files scanned): exactly 3 hits, all in
`twotower.py`'s SLM-138 deep-supervision block (lines ~2101–2140,
`training_loss`), zero false positives anywhere else.** All 3 are true
positives, not false positives — see "Found defects" below for what each
one is and why each is suppressed rather than fixed.

## Migration behavior for historical configs

- A config **missing a field entirely** (a dict/namespace that predates the
  field, e.g. an old checkpoint's `config` payload) validates exactly as if
  the field were present at **its real `TwoTowerConfig` default** — never a
  hard fail and never a blanket empty-tuple/zero fallback (which would be
  wrong for fields whose real default is non-empty, e.g. `diffusion_policies`
  defaults to the full 11-policy tuple, not `()`). Implemented via
  `twotower_schedule_policy._FIELD_DEFAULTS`, exercised by
  `test_missing_attributes_default_like_an_absent_legacy_field`.
- A config carrying a **genuinely invalid historical value** (e.g. the
  pre-fix combination `denoiser_arch="stacked"` with a non-empty
  `recursive_depth_supervision_weights` — reachable before this change
  because the runtime silently ignored it) now raises
  `ScheduleValidationError` on load instead of loading silently. The
  documented remedy is an explicit migration: clear the field, or use
  `slm_training.models.checkpoint_migrate.migrate_to_shared_recursive_denoiser`
  to actually adopt the recursive architecture the weights were meant for.
  No such checkpoint exists in this repo (no `.pt` checkpoints are committed
  outside `src/slm_training/resources/checkpoints`, and none reference this
  field per a repo grep), so no historical evidence artifact needed
  migrating as part of this change.
- No field in this audit is "genuinely new and mandatory" — every field
  already had a default before this change; validation adds constraints on
  the *values* a field may hold, not a requirement that it be present.

## Found defects (documented, not fixed — see non-goals)

### 1. Per-depth deep-supervision weight ratio is never applied (real bug)

`TwoTowerModel.training_loss`, `src/slm_training/models/twotower.py` (deep
supervision block, guard-flagged `UNUSED_LOOP_WEIGHT` and `UNGUARDED_SUM`):

```python
for d, w in enumerate(ds_weights[:usable]):
    d_logits = depth_logits[d]
    ...
    d_loss = (d_ce * weights)[mask_flat].mean()   # `w` is never used here
    depth_losses.append(d_loss)
    ...
normalized = torch.stack(depth_losses).sum() / total_w
```

`w` (the per-depth weight from `recursive_depth_supervision_weights`) is read
only via `total_w = sum(ds_weights[:usable])` for the final normalization
divisor — it is **never multiplied into `d_loss`**. The result: every
supervised depth contributes to `normalized` with an **equal, unweighted**
mean; only the *aggregate sum* of the configured weights changes the overall
term's scale, never the *ratio* between depths. A user configuring
`recursive_depth_supervision_weights=(0.01, 100.0)` to heavily favor the
final recursion gets byte-identical per-depth contributions to
`(1.0, 1.0)` — confirmed empirically by
`test_per_depth_weight_ratio_is_not_applied_known_defect`
(`tests/test_models/test_twotower_schedule_policy.py`), which pins the
current (defective) behavior so a future intentional fix changes the test
deliberately.

**Suggested fix direction (not applied here — non-goal):** multiply `d_loss`
by `w` before appending (`depth_losses.append(w * d_loss)`), then either
normalize by `total_w` (weighted mean) or drop the normalization and let
`total_w`'s scale be the deliberate lever.

**Severity/blast radius:** low in current practice — `denoiser_arch=
"shared_recursive"` is wiring/fixture-only per
`docs/design/iter-slm138-recursive-denoiser-20260720.md` ("Wiring-only
evidence... GPU training are deferred"); no shipped/promoted checkpoint uses
it. Should be filed as its own issue before any production
`shared_recursive` + non-uniform `recursive_depth_supervision_weights`
training run.

### 2. `diffusion_length_loss_weight` default is inert under the default `mask_pattern`

Both `TwoTowerConfig()` and `ModelBuildConfig()` ship
`diffusion_length_loss_weight = 0.1` under the default `mask_pattern =
"random"`. `TwoTowerModel` only builds `self.length_head` (and only ever
applies this weight) when `mask_pattern == "diffusion"`
(`src/slm_training/models/twotower.py`, denoiser construction block). So the
*out-of-the-box default* is already in the "configured but currently inert"
state this issue's capability-matrix rules elsewhere reject — but rejecting
it here would reject the bare default construction, which the non-goals
forbid working around by changing the default. `validate_diffusion_length_loss_weight`
is therefore shape-only (finite, `>= 0`) with no capability rule, and this
asymmetry is documented in its docstring rather than silently accepted.

**Suggested fix direction (not applied):** either default
`diffusion_length_loss_weight` to `0.0` and require `mask_pattern="diffusion"`
experiments to opt in explicitly, or move the `0.1` default into the
diffusion-specific factory branch instead of the base dataclass. Low
severity (zero runtime cost; `self.length_head is None` short-circuits the
term), but confusing for anyone reading the default and assuming it does
something.

### 3. `targeted_margin_family_weights` is dead configuration

`targeted_margin_family_weights` is declared on `TwoTowerConfig` and
`ModelBuildConfig`, threaded through `factory.py`'s
`_twotower_config_from_build` and `apply_runtime_overrides`'s allowed-override
list — but has **zero readers** anywhere in `twotower.py`'s loss or decode
code (confirmed by `grep -rn targeted_margin_family_weights src/`: only
declarations and plumbing, no consumer). `targeted_margin_manifest` and
`targeted_margin_value` (its scalar siblings) are similarly unread. Only
shape validation (unique keys, non-negative weights) is implemented here;
there is no downstream consumer to write a capability rule against.

**Suggested fix direction (not applied):** either wire it into the
`legal_margin_mode` contrastive-margin loss it appears designed to pair with,
or remove the dead fields. Zero runtime impact today (the field is
literally never read), so no urgency, but worth filing to avoid future
confusion (someone will eventually set it expecting an effect).

### 4. (Minor, lower confidence) `slot_component_lexeme_priors` broadcast on a length-1 score vector

`_slot_component_logits`, `src/slm_training/models/twotower.py`:
`bias[row] += scores` where `bias[row]` has shape `[num_classes]` and
`scores` is a per-key prior vector. If `len(scores) == num_classes`, this is
a normal elementwise add; if `len(scores) not in (1, num_classes)`, PyTorch
raises a broadcast `RuntimeError` (a crash, not silent). But if
`len(scores) == 1`, PyTorch **silently broadcasts** that single score
uniformly across every class — a fail-open path this audit did not design a
config-time rule for, because the correct length (`num_classes`) is only
known once the tokenizer's component inventory is built, and
`slot_component_lexeme_priors` is not on the user-facing config surface
(only ever populated internally by `TwoTowerModel.from_records`, where
`classes = len(component_index)` and every generated score vector already
has that exact length — so this path is not reachable through normal
training). Recorded for completeness in case a future change exposes this
field to direct configuration or checkpoint tampering.

## CI registration

- **Wired, in both places, via one change**: `scripts/check_changed.py` (the
  shared "run lightweight checks for the suites affected by local changes"
  entry point) now runs `python -m scripts.verify_numeric_schedule_guard`
  whenever any changed path falls under the guard's scan roots
  (`src/slm_training/models/`, `src/slm_training/harnesses/model_build/`),
  alongside its existing unconditional `verify_version_stamps --check` call.
  This one function (`check()`) is invoked from **two** existing places, so
  extending it registers the guard in both without further changes:
  - locally via `.githooks/check-changed` → `python -m scripts.check_changed`;
  - in CI via `.github/workflows/ci.yml`'s `python` job, step "Run changed
    regression tests": `python -m scripts.check_changed --changed-tests-only
    --base-ref "$BASE_SHA"`.
  The same CI job also runs `ruff check .` and
  `python -m compileall -q src scripts tests` unconditionally, which the new
  files pass (`ruff check` verified clean for every file listed below).
- The property-based test suites
  (`tests/test_harness_core/test_schedule_validation.py`,
  `tests/test_models/test_twotower_schedule_policy.py`,
  `tests/test_scripts/test_verify_numeric_schedule_guard.py`) are ordinary
  `pytest` files under `tests/`; `scripts/check_changed.py`'s existing
  prefix-based suite selection (`src/slm_training/harness_core/` →
  `tests/test_harness_core` + others, `src/slm_training/models/` →
  `tests/test_models` + `tests/test_harnesses/model_build`, `scripts/` →
  `tests/test_scripts`) already covers all three without a new mapping entry.
  No further CI wiring is needed for the test suite itself. Manual run:
  ```bash
  python -m scripts.verify_numeric_schedule_guard
  pytest -q tests/test_harness_core/test_schedule_validation.py \
            tests/test_models/test_twotower_schedule_policy.py \
            tests/test_scripts/test_verify_numeric_schedule_guard.py
  ```

## Files touched

- `src/slm_training/harness_core/schedule_validation.py` (new)
- `src/slm_training/harness_core/__init__.py` (export the new primitives)
- `src/slm_training/models/twotower_schedule_policy.py` (new)
- `src/slm_training/models/twotower.py` (wire `__post_init__`/`save`/`load`;
  add 3 documented `schedule-guard: allow` suppressions)
- `src/slm_training/harnesses/model_build/config.py` (wire `__post_init__`)
- `src/slm_training/harnesses/model_build/factory.py` (revalidate after
  `apply_runtime_overrides`)
- `scripts/verify_numeric_schedule_guard.py` (new)
- `scripts/check_changed.py` (wire the guard into the existing local hook)
- `scripts/run_slm138_recursive_denoiser_fixture.py` (stop passing
  `recursive_depth_supervision_weights` on the non-recursive arch — see
  `versions.json`'s `model.recursive_denoiser` no-bump entry for the
  byte-identical-output verification)
- `tests/test_harness_core/test_schedule_validation.py` (new)
- `tests/test_models/test_twotower_schedule_policy.py` (new)
- `tests/test_scripts/test_verify_numeric_schedule_guard.py` (new)
- `src/slm_training/resources/versions.json` (`harness.core` → v2,
  `model.twotower` → v81, `model.recursive_denoiser` no-bump)
- `docs/design/rsc-a06-numeric-schedule-validation-report-20260721.json` (new)
- `docs/design/rsc-a06-numeric-schedule-validation-20260721.md` (this file)
