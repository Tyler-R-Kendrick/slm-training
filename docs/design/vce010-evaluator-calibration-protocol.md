# VCE-010 (SLM-469): evaluator calibration, blinded adjudication, and risk-coverage protocol

New module: [`src/slm_training/evals/evaluator_calibration_protocol.py`](../../src/slm_training/evals/evaluator_calibration_protocol.py).
Runner: [`scripts/run_vce010_evaluator_calibration_protocol.py`](../../scripts/run_vce010_evaluator_calibration_protocol.py).

SLM-106 (EFS0-04, [`judge-independence-audit.md`](judge-independence-audit.md))
already defines judge-independence/human-audit principles; this issue reuses
its `JudgeEvidenceV1`/blinding/independence concepts rather than reopening
them. This module composes existing owners instead of reimplementing any of
them:

| Piece | Owner (unchanged) |
| --- | --- |
| frozen stratified sample | `data.semantic_contrast.SemanticContrastBuilder` (VCE-006/007) |
| deterministic evaluator verdict | the corpus's own `verifier_ok` ground truth (already computed via the real compiler gate stack when the corpus was built -- never recomputed here) |
| independent external-family judge, mocked adapter | `evals.judge_independence.ExternalJudgeAdapter` / `JudgeEvidenceV1` |
| blinded human labels, >=2 raters + adjudication | `harnesses.annotations.judge_audit.freeze_blinded_pairs` / `import_blinded_labels` |
| agreement / pass-set divergence / calibration error | `evals.judge_independence.analyze_triple_judges` |
| risk/coverage curve + Brier/ECE | `dsl.grammar.fastpath.trust_train.gate_calibration_report` |
| preregistered thresholds | `autoresearch.experiment_campaign.ExperimentCampaignV1` + `CampaignStore` |

The only genuinely new piece is `reason_family_confusion_matrix` -- no
actual-vs-predicted reason-family crosstab existed anywhere in the repo
(`analyze_triple_judges` only tallies a flat disagreement-reason `Counter`).

## What this runs

Four arms against one frozen fixture slice (a small, freshly generated
`SemanticContrastBuilder` corpus, `wide_sources=True, strict_delta=True`):

- `deterministic_only` (control) -- the corpus's own admitted-pair ground
  truth; never consults external or human evidence.
- `external_only_missing_human` -- a mocked, no-credentials-required
  `ExternalJudgeAdapter` call, but with no human evidence gathered.
- `human_only_missing_external` -- a real blinded pair study (synthetic
  seeded raters + an adjudicator on forced disagreement), but with no
  external judge gathered.
- `full_triple_judge` -- deterministic + external + human evidence merged
  into `analyze_triple_judges` rows, plus `gate_calibration_report`
  risk/coverage and `reason_family_confusion_matrix`.

## How this satisfies each acceptance criterion

- **"Protocol is executable without external credentials using export/import
  packages and mocked adapters"** -- `ExternalJudgeAdapter` is given an
  injected `transport` callable (`_mock_transport`) that reads only
  `prompt`/`left_openui`/`right_openui`, never a real network/API call; the
  human side uses `judge_audit`'s real export (`freeze_blinded_pairs`) /
  import (`import_blinded_labels`) package flow with synthetic labels.
  `test_run_protocol_end_to_end_with_real_arms`.
- **"Independence provenance is machine-checked"** --
  `JudgeEvidenceV1.independent` runs on every external-judge record;
  `result["independence_ok"]` aggregates it, and a misconfigured judge
  (declared family overlapping a candidate family) flips
  `full_triple_judge`'s disposition to `"mismatch"`, never silently passing.
  `test_independence_is_machine_checked_and_contamination_is_reported`.
- **"Human/independent evidence cannot become compiler legality"** --
  structural: this module never imports `slm_training.data.verify` (the
  G0-G12 gate stack); statically pinned by
  `test_module_never_imports_the_compiler_gate_stack`.
- **"Promotion thresholds are preregistered before external run"** --
  `run_protocol` calls `store.lock_experiment_campaign(protocol.manifest())`
  -- which fixes `calibration_error_max` as the rollback-gate threshold --
  *before* the frozen sample is even built, let alone before any
  external-judge or human-label evidence exists.
  `test_manifest_rollback_gate_is_not_vacuous_and_is_preregistered`.
- **"Missing external evidence yields blocked/inconclusive, not assumed
  validity"** -- `external_only_missing_human` /
  `human_only_missing_external` both report `disposition="blocked"` with an
  explicit `reason`, never a fabricated triple-judge result; a
  too-small frozen sample reports `disposition="inconclusive"`.
  `test_missing_evidence_arms_are_blocked_not_assumed_valid`,
  `test_run_protocol_with_tiny_sample_is_inconclusive_not_fabricated`.
- **"Result docs are generated/checked from structured artifacts"** -- the
  runner script writes a version-stamped JSON from `run_protocol`'s return
  value (a plain dict written to `CampaignStore` first); nothing here is
  hand-authored prose.

## Additional protocol requirements

- **>=2 human labels per pair plus adjudication policy for disagreement** --
  `_human_arm_rows` always generates `max(2, min_raters)` rater labels per
  pair and, whenever raters disagree, an adjudicator label resolving it --
  reusing `import_blinded_labels`'s own `needs_adjudication`/`complete`
  logic to validate the policy, not reimplementing it.
- **ambiguous/UNKNOWN cases retained separately** -- an incomplete aggregate
  (disagreement without an adjudicator, or missing raters) gets
  `human_ambiguous=True`, `human_verdict="error"`, `human_pass=False`
  rather than defaulting to a winner; excluded from the risk/coverage
  calibration set, but never dropped from `ambiguous_pair_ids`.
  `test_ambiguous_human_evidence_is_retained_separately_not_defaulted`.
- **audit sample prohibited from training** -- `freeze_blinded_pairs` marks
  every frozen pair `"training_use": False, "audit_holdout": True` by
  construction (reused unmodified from `judge_audit.py`, which already
  enforces this for the SLM-106 pair study this issue explicitly reuses).

## Evidence

Recipe (deterministic, ~5s per run):

```bash
python -m scripts.run_vce010_evaluator_calibration_protocol --mode fixture
```

Full stamped result:
[`vce010-evaluator-calibration-results.json`](vce010-evaluator-calibration-results.json).

| Metric | Value |
| --- | --- |
| Arms | 4 (1 control, 3 candidates) |
| `pair_n` | 12 (frozen `source_count=4`, `wide_sources=True`, `strict_delta=True`) |
| `external_human_calibration_error` | 0.158 (preregistered ceiling: 0.35) |
| `external_human_cohen_kappa` | 0.0 |
| `admission_divergence_rate` (deterministic vs. human) | 0.0 |
| `independence_ok` | `true` |
| `ambiguous_pair_ids` | `[]` |
| `rollback_gate_fired` | `false` |

Verified deterministic: two independent runs against separate output roots
produce an identical `manifest_sha256` and identical per-arm dispositions
(`test_run_protocol_is_deterministic_on_repeat`).

Regression coverage:
`tests/test_evals/test_evaluator_calibration_protocol.py`, 12/12 passing.

## Honest scope

- This is fixture-scale evidence over a handful of freshly generated
  semantic-contrast pairs, not a production-representative sample --
  consistent with `claim_class="diagnostic"` and the deliberately vacuous
  `diagnostic_only` promotion gate (mirrors PCT-008/VCE-009's own scoping).
- The external judge is a labeled mock transport, not a real external-model
  API call; human raters are synthetic, seeded labels standing in for a real
  blinded pair study. Neither claims real-world judge/human-agreement
  numbers -- only that the plumbing, statistics, and honesty invariants
  (blocked-on-missing-evidence, ambiguous-retained-separately,
  independence-machine-checked, preregistered-before-run) are real and
  exercised end to end.
- `cohen_kappa`/`fleiss_kappa` reading near zero at n=12 is expected small-n
  noise given the synthetic 80%-correct/85%-correct rater simulation, not a
  claim about real evaluator quality.
