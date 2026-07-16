# Canonical harness improvement map

Select the one primary owner. Read sibling sections only when the change crosses
their actual boundary.

## Autoresearch and hypothesizer

- Owner: `src/slm_training/autoresearch/`; CLI: `scripts/autoresearch.py`.
- Outputs: `outputs/autoresearch/<campaign>/artifacts/<kind>/<sha>.json`, hash-chained
  `events.jsonl`, `checksums.jsonl`, `results.tsv`, and `runs/<experiment>/`.
- Improve evidence capture, provider isolation, typed schemas, matrix diversity,
  selection, feedback calibration, or diagnosis. A completed experiment must
  become typed hypothesizer feedback for the next matrix.
- Reject provider shell/code, uncited experiments, repeated knob signatures,
  fewer than five candidates, unacknowledged feedback, and unready RL.
- Check: `pytest -q tests/test_autoresearch`, `python -m scripts.autoresearch --help`,
  and `evaluate-hypothesizer` plus documented AgentV evidence for provider changes.
- Docs: `docs/design/autoresearch-autotraining.md`, `research-lineage.md`.

## Annotation harness

- Owner: `src/slm_training/harnesses/annotations/`; export CLI:
  `scripts/export_annotations.py`; API consumers live under `src/slm_training/web/`.
- Improve validation, append safety, attempt provenance, pair conversion, or feedback
  visibility. Preserve stable IDs and atomic writes.
- Keep raw stores under their configured output root; derived preference artifacts
  belong to the consuming run, not a new root folder.
- Check: `pytest -q tests/test_web/test_annotation_store.py tests/test_web/test_annotations.py tests/test_web/test_bad_outputs.py`.

## Distillation harness

- Owner: `src/slm_training/harnesses/distill/`; CLIs: `scripts/collect_trajectories.py`,
  `scripts/self_distill.py`, and `scripts/resume_climb.py`.
- Improve trace selection, failure-cone repair, lineage labels, or SFT conversion.
  Preserve checkpoint hashes and keep selection data disjoint from frozen evals.
- Store traces and records inside the owning run/campaign; never train on held-out
  benchmark traces or silently relabel generated records as gold.
- Check: `pytest -q tests/test_harnesses/distill tests/test_models/test_trace_store.py`.

## Experiment, scaling, and promotion harness

- Owner: `src/slm_training/harnesses/experiments/`; CLIs:
  `scripts/run_scaling_ladder.py`, `run_mixture_search.py`, and matrix scripts.
- Improve confidence bounds, matched controls, efficiency estimates, stop rules,
  or immutable promotion registration. Quality gates outrank speed and cost.
- Results belong in the existing run root and matching `docs/design/*results.json`
  plus markdown matrix. Register only fully evaluated checkpoints.
- Check: `pytest -q tests/test_harnesses/experiments` plus focused matrix tests.

## Model-build and evaluation harness

- Owner: `src/slm_training/harnesses/model_build/`; CLIs:
  `scripts/train_model.py`, `evaluate_model.py`, `diagnose_eval.py`, and `model_cycle.py`.
- Improve plugin/config contracts, training, resume state, decoding diagnostics,
  suite evaluation, checkpoint sync, or ship gates through shared code paths.
- Runs belong under `outputs/runs/<run-id>/`; every evaluation emits AgentEvals and
  AgentV. Every checkpoint updates `docs/MODEL_CARD.md` and the README summary.
- Never infer ship readiness from fixtures, partial suites, or missing scoreboards.
- Check: `pytest -q tests/test_harnesses/model_build tests/test_models` plus CLI tests.

## Preference harness

- Owner: `src/slm_training/harnesses/preference/`; CLI:
  `scripts/train_preference.py`; pair production may consume annotations/trajectories.
- Improve reward components, pair validity, reference semantics, or bounded training.
  Keep the documented surrogate-DPO distinction and pair provenance.
- Store pairs/checkpoints under the owning run; do not create a second corpus tree
  or train on eval-feedback holdouts.
- Check: `pytest -q tests/test_harnesses/quality/test_preference_corpora.py tests/test_harnesses/quality/test_quality_helpers.py`.

## Quality and retrieval harness

- Owner: `src/slm_training/harnesses/quality/`; consumers are train-data,
  model-build, preference, and quality/grammar matrix scripts.
- Improve retrieval, curriculum, corruption, compact schema context, or adversarial
  synthesis at the shared owner. Preserve deterministic fingerprints.
- Keep derived indexes in the owning data/run output. Never expose gold placeholders
  under honest mode.
- Check: `pytest -q tests/test_harnesses/quality`.

## RL harness

- Owner: `src/slm_training/harnesses/rl/`; CLI: `scripts/train_rl.py`; external
  backends integrate through `scripts/model_cycle.py` and `src/slm_training/integrations/`.
- Improve trajectory validation, rewards, advantages, replay, or adapters. Call the
  shared RL readiness assertion; there is no override.
- Store trajectories/checkpoints under the owning run and reconcile external jobs
  into canonical lineage. Hardware smoke is never promotion evidence.
- Check: `pytest -q tests/test_harnesses/rl tests/test_harnesses/quality/test_rl_curriculum_telemetry.py`.

## Held-out test-data harness

- Owner: `src/slm_training/harnesses/test_data/`; CLI: `scripts/build_test_data.py`.
- Improve suite coverage, adversarial/OOD generation, leakage fingerprints, or
  structure-disjoint enforcement without fitting training data to the holdout.
- Version outputs under `outputs/data/eval/<version>/`; require the train manifest
  for disjointness and keep suite sizes explicit in all claims.
- Check: `pytest -q tests/test_harnesses/test_data tests/test_integration/test_ship_disjoint.py`.

## Training-data harness

- Owner: `src/slm_training/harnesses/train_data/`; CLI: `scripts/build_train_data.py`.
- Improve source adapters, catalog classification, synthesis, verification, mixture,
  or immutable derivation. Preserve source/license/governance and structural hashes.
- Version outputs under `outputs/data/train/<version>/`; committed tiny fixtures
  belong in `src/slm_training/resources/`, never a new root data directory.
- Check: `pytest -q tests/test_harnesses/train_data tests/test_data` and the
  train/test disjoint integration test.
