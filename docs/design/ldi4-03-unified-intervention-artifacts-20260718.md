# LDI4-03 — Unified intervention artifacts, evaluation, and promotion gates (SLM-137)

**Date:** 2026-07-18. **Status:** unified intervention manifest + registry + standard
evaluation-bundle schema + deterministic promotion state machine + one-active/cyclic
lineage rules, implemented and tested. **Torch-free consolidation core.** It wraps the
existing per-kind owners; it does **not** replace the model plug-in protocol, add a
trainer/scheduler, or invent a new event schema. Real-model loading, merge/export parity,
dashboard rendering, and bucket upload are deferred to the follow-on integration.

Unblocked by the merged SLM-132 (remine campaign), SLM-134 (ReFT/DiffMean artifact
contract), and SLM-136 (SAE diagnostic artifact contract), all on `main` — together with
the earlier causal PEFT (SLM-121) and TwoTower delta (SLM-123) adapter contracts.

## What this delivers

`src/slm_training/lineage/interventions.py` gives all four intervention kinds one common
identity/loading/evaluation/lineage/promotion contract:

- **`InterventionManifest`** — a versioned, **fail-closed** common manifest generalizing
  the richest existing shape (the TwoTower `adapter_manifest.json`): `kind`
  (`causal_peft | twotower_delta | reft | sae_diagnostic`), `method`, `status`,
  `deployable`, a `BaseIdentity` block (architecture / base id+revision / tokenizer sha /
  base compatibility fingerprint), module/site map, parameter shapes, content-addressed
  `artifact_files` (reusing `checkpoint_reference.FileArtifact`), a `config_fingerprint`,
  corpus/action-evidence/materializer/objective/eval identities, explicit acyclic
  `parent_intervention_ids`, and a kind-specific tagged `kind_payload`. `from_dict`
  rejects unknown kinds/fields/versions; `fingerprint()` uses `lineage.records.content_sha`.
- **`InterventionRegistry`** — one `kind → validator` surface to `validate`/`inspect` an
  artifact **without loading the base model**; fails closed on an unregistered kind. A
  diagnostic-only SAE can never be reported deployable.
- **`EvaluationBundle`** — one standard result schema (identity / event / locality /
  end-to-end); a **missing required field makes the artifact ineligible, never a silent
  pass/zero**, and `end_to_end.ship_gates` reuses `evaluate_ship_gates` output.
- **`promote` / `PROMOTION_TRANSITIONS`** — the deterministic state machine
  `wiring → diagnostic → rejected | eligible → promoted`. `diagnostic → eligible`
  requires a complete, ship-gate-passing bundle; `eligible → promoted` requires a
  deployable (non-diagnostic) artifact whose protected ship gates pass. **No scalar score
  overrides a failed protected gate**, and `expired / stopped / blocked_by_corpus /
  no_safe_direction` are **run outcomes, not promotion states**.
- **`assert_single_active`** (one active intervention; composition rejected),
  **`detect_lineage_cycle`** (DFS over `parent_intervention_ids`), and
  **`build_closeout_index`** (every artifact + status, with the current best
  eligible/promoted artifact or an explicit "none qualifies").

## Consolidation notes

Every kind already carries a base identity, `tokenizer_sha`, a site/module map, and a
config fingerprint; the manifest standardizes those and generalizes `parameter_shapes`
(previously only in the TwoTower manifest). Fingerprints across the kinds were
inconsistent (`content_sha` full-hex vs `sha256[:16]` vs folded fields); the unified
manifest standardizes on `content_sha`. This module is additive — it wraps the existing
`artifact_identity()` / `compatibility_fingerprint()` / `save_*` / `load_*` on each model
rather than modifying `ModelPlugin` or `factory.build_model`.

## Verification

```bash
python -m pytest tests/test_lineage_interventions.py -q     # 15 passed
python -m pytest tests/test_lineage/test_lineage.py -q
python -m ruff check src/slm_training/lineage tests/test_lineage_interventions.py
python -m scripts.repo_policy
```

Tests cover: manifest round-trip + stable fingerprint; fail-closed on unknown
kind/field/version/status; all four kinds validate through one interface; diagnostic-only
cannot be deployable (construction + inspect); integrity/compat validation (missing
compat fingerprint, non-content-addressed file, missing module map); one-active-
intervention enforcement; cyclic + self-parent lineage rejection; evaluation-bundle
eligibility (missing field or failed gate ⇒ ineligible); deterministic promotion
transitions incl. illegal-skip, diagnostic→eligible evidence requirement, failed-gate
promotion prevention, SAE non-promotion, and run-outcome-not-a-status; and the closeout
index (best deployable or an explicit none). Committed evidence:
`docs/design/ldi4-03-intervention-closeout-index-20260718.json` (four kinds, all
`diagnostic`, `best_deployable: null`).

## Scope / deferred

The torch-light contract (manifest / registry / evaluation bundle / promotion state
machine / one-active + cyclic-lineage) is complete and tested here. Deferred to the
follow-on integration (torch / network / UI): actually loading a named intervention into
a live model and enable/disable/unload parity, merge/export parity where a kind proves
it, the `checkpoint_bucket` upload plan, the dashboard/data-API rendering of the manifest
graph, and appending real trained/evaluated rows to `docs/MODEL_CARD.md`. A diagnostic
SAE stays in the research inventory, never the deployable checkpoint table, unless a
separately promoted steering artifact exists. No production default is changed here.
