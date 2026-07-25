# SLM-292 (AP-010): default-off semantic-contrast objective

**Claim class: `fixture_wiring` -- objective implemented and wired, NOT promoted.**
Machine-readable record: [`iter-slm292-semantic-contrast-objective-20260725.json`](iter-slm292-semantic-contrast-objective-20260725.json).
Matched control/treatment smoke evidence:
[`iter-slm292-semantic-contrast-smoke-20260725.json`](iter-slm292-semantic-contrast-smoke-20260725.json) /
[`.md`](iter-slm292-semantic-contrast-smoke-20260725.md).

## Decision

Implement one preregistered objective (pairwise margin) that pairs each
SLM-290 `openui_hard_valid_v1` positive OpenUI program with its hard-valid
negative (parser/schema/reference-valid, semantically wrong) and pulls the
pooled context-encoder representation of the gold program toward the shared
prompt anchor while pushing the mutated program's representation away by a
margin. The objective is **default off** (`semantic_contrast_loss_weight =
0.0`) and, when off, is bit-exact with the prior `TwoTowerModel.training_loss`
path.

InfoNCE and semantic-regret were considered (per the issue) but not
implemented -- the task instructions named margin/InfoNCE as the lower-risk,
more standard choices given the session's time budget, and "pick ONE" kept
scope, test surface, and the bit-exactness proof tight.

## What was built

- **`src/slm_training/models/semantic_contrast_loss.py`** (new): framework-thin
  pure module -- `load_contrast_pairs` (streams `pairs.jsonl`, keeps only
  `admitted=True` rows, i.e. pairs where the positive passes
  `binding_aware_meaningful_v2` and the matched negative fails it),
  `sample_contrast_pairs` (deterministic seeded sampling, optional per-family
  weighting), `pairwise_margin_contrast_loss` (cosine-distance margin loss),
  and `compute_semantic_contrast_step` (orchestrates one training step given a
  caller-supplied `rep_fn`, returning a `SemanticContrastStepResult` with every
  field the issue's logging requirement names). It never imports
  `twotower.py` and never calls an encoder itself, so it is unit-testable
  without constructing a model.
- **`src/slm_training/models/twotower.py`** (modified): new default-off
  `TwoTowerConfig` fields (`semantic_contrast_loss_weight`,
  `semantic_contrast_corpus_path`, `semantic_contrast_objective`,
  `semantic_contrast_margin`, `semantic_contrast_temperature`,
  `semantic_contrast_batch_pairs`, `semantic_contrast_sampling_seed`,
  `semantic_contrast_split`, `semantic_contrast_family_weights`) next to the
  existing SLM-164 `legal_margin_mode` / `targeted_margin_*` precedent for a
  default-off contrast lever. `training_loss` gains one guarded block
  immediately before its final `return mask_loss`, matching the exact
  `w = float(getattr(...) or 0.0); if w > 0.0: ...; mask_loss = mask_loss + w
  * term` pattern used by every other optional loss term
  (`binder_topology_loss_weight`, `component_plan_loss_weight`, etc.). A new
  private `_semantic_contrast_loss_term` method lazily loads and caches the
  corpus on the model instance (keyed by `(path, split)`; never re-parses per
  step) and reuses the existing `_encode_context` + `_pool_context` machinery
  as the representation function -- one extra forward pass on
  `3 * semantic_contrast_batch_pairs` short strings (prompt + positive program
  + negative program) per enabled step, zero extra work when disabled.
  Also fixed a latent bug the new `Path`-typed-then-`str`-typed config field
  exposed: `TwoTowerModel.save()`'s `meta.json` write used a bare
  `json.dumps(...)` on `asdict(self.config)`, which already contained a
  `Path`-typed field (`targeted_margin_manifest`); added `default=str` so any
  current or future `Path`-typed config field serializes cleanly. Also
  discovered and worked around a related constraint: `torch.load(...,
  weights_only=True)` (used by `TwoTowerModel.save`/`from_checkpoint`) does
  not allowlist `pathlib.Path`, so `semantic_contrast_corpus_path` is typed
  `str | None`, not `Path | None`.
- **`src/slm_training/harnesses/model_build/config.py` /
  `factory.py`** (modified): mirrored the same fields onto `ModelBuildConfig`
  and the `TwoTowerConfig` construction kwargs, matching every other optional
  loss lever's plumbing.
- **`src/slm_training/harnesses/model_build/feature_flags.py`** (modified):
  classified `semantic_contrast_corpus_path` as artifact plumbing (not an
  OpenFeature lever), matching the existing `targeted_margin_manifest`
  precedent; the paired `semantic_contrast_*` behavior levers auto-register as
  flags with no additional code.
- **`src/slm_training/harnesses/experiments/slm292_semantic_contrast_smoke.py`**
  + **`scripts/run_slm292_semantic_contrast_smoke.py`** (new): the fixture-scale
  matched control/treatment harness (see below).
- Tests: `tests/test_models/test_semantic_contrast_loss.py` (16 tests, pure
  logic), 6 new tests appended to
  `tests/test_harnesses/model_build/test_twotower.py`, and
  `tests/test_harnesses/experiments/test_slm292_semantic_contrast_smoke.py`
  (5 tests).

## Evidence collected (all fixture/smoke scale)

1. **Bit-exact-when-disabled** (the one hard acceptance requirement this
   session *can* fully verify):
   `test_semantic_contrast_disabled_by_default_is_bit_exact` compares a
   baseline model (no `semantic_contrast_*` kwargs) against one built with
   `semantic_contrast_loss_weight=0.0` explicitly -- identical `state_dict()`
   after construction, and identical loss tensors after a
   RNG-state-controlled `training_loss()` call (mirroring the existing
   `test_optional_heads_do_not_shift_training_rng` /
   `test_auxiliary_loss_does_not_change_base_gradients` pattern in the same
   file). `test_semantic_contrast_disabled_does_not_change_optimizer_updates`
   additionally proves an AdamW step produces byte-identical parameters.
   Both also assert the corpus loader is never called and the step counter
   stays at 0 when disabled.
2. **Enabled correctness** on a tiny synthetic fixture corpus (schema-matched
   to the real `pairs.jsonl`) and, separately, against a real 25-row slice of
   the SLM-290 corpus: finite loss, successful `backward()`, and every
   required field present in `last_training_metrics`.
3. **Matched control/treatment fixture smoke**
   (`python -m scripts.run_slm292_semantic_contrast_smoke`): two
   `TwoTowerModel`s (d_model=32, 2 training records, 15 steps each), identical
   seed/architecture/records/steps, differing only in
   `semantic_contrast_loss_weight` (0.0 vs 1.0). Total run time ~5s. Logged,
   per treatment step: `semantic_contrast_loss_weight=1.0`,
   `semantic_contrast_objective="margin"`, `semantic_contrast_margin=0.2`,
   `semantic_contrast_pairs=6`, `semantic_contrast_sampling_seed=0`, per-step
   `semantic_contrast_family_counts`/`transform_counts`, and
   positive/negative distance means. Per-mutation-family aggregate over all
   logged steps (content/binding/contract) is reported in the smoke JSON/MD.

## What the acceptance bar requires and is **NOT** met

| Acceptance criterion | Status |
| --- | --- |
| meaning-v2 +0.05 absolute OR binder/reference F1 +0.10, paired CI excluding zero | **not measured** -- requires a full training run + `evaluate_model --ship-gates` on frozen held-out suites |
| Syntax/contract validity regression <= 0.01 | **not measured** |
| Replicate on >=3 seeds before promotion | **not measured** (this session: 1 seed, fixture scale) |
| Objective is zero and legacy training bit-exact when disabled | **met** -- see evidence above |

Do not read the smoke's "treatment final loss > control final loss" (13.57 vs
12.04 after 15 steps) as a quality signal one way or the other: it is 2
training records, 15 steps, an untrained margin term with random-init
representations, and no eval suite in the loop. It only demonstrates the
objective computes, backprops, and logs correctly.

## Constrained-decode scope note

The smoke intentionally does **not** call `TwoTowerModel.generate` at all: a
single constrained-decode call (unconstrained generation is explicitly
refused by `slm_training.models.grammar.require_constrained_generation`) took
over 60 seconds even on this toy architecture in this session's environment,
which alone would exceed the repo's `MAX_RUN_MINUTES = 3` hard run cap
(`src/slm_training/levers.py`) for a fixture-scale smoke. Raw/constrained/
repaired decode-outcome comparisons are explicit follow-up work via
`scripts.evaluate_model` at a scale where that cost is budgeted for.

## Recommended follow-up (not filed as a new Linear issue by this session)

1. A `>=3`-seed promotion campaign: real SFT training corpus for the token-CE
   loss, SLM-290 `openui_hard_valid_v1` (`train` split) for the contrast
   objective, frozen held-out/adversarial/`rico_held` suites for meaning-v2 and
   binder/reference F1, `evaluate_model --ship-gates` for the syntax/contract
   validity regression check, and a paired bootstrap CI across seeds excluding
   zero before any promotion claim -- per
   `docs/design/experiment-campaign-governance.md`'s `ExperimentCampaignV1`
   contract.
2. If promoted, extend the SLM-290 corpus's currently small/absent
   `topology_*` / `binding_introduce_incompatible_symbol` /
   `contract_unresolve` mutation families so the contrast objective is not
   trained only on `content_swap_family` / `content_invert_role` /
   `binding_swap_symbol(_reverse)` / `contract_archetype_mismatch`.
3. A real (non-toy) decode-outcome comparison (raw / constrained / repaired)
   between control and treatment checkpoints once a full training run exists,
   run through `scripts.evaluate_model` where decode-budget cost is already
   accounted for.

## Version stamp

- `model.twotower`: `v240`
- `data.semantic_contrast`: `v2` (consumed, not modified)
- `harness.experiments.slm292_semantic_contrast_smoke`: `v1` (new)
- `harness.experiment_feature_flags`: `v4`
- Code commit: `b8fabf4c21f27c2d019570a3bc153b9dd1467b4c` (dirty worktree)
