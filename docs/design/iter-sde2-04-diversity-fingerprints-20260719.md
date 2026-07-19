# SDE2-04 diversity fingerprinting (first slice)

Date: 2026-07-19  
Issue: SLM-171  
Branch: `agent/slm-171-sde2-04-diversity-fingerprints`

## What this slice delivers

A reusable, versioned diversity-fingerprinting module under
`src/slm_training/harnesses/train_data/diversity.py` and an audit script at
`scripts/audit_corpus_diversity.py`. The module computes six resolution
fingerprints per record:

1. **canonical_root_id** — full typed AST after alpha-renaming/nonsemantic
   normalization (via `canonical_fingerprint`).
2. **binding_aware_sketch** — component topology + role-bearing props +
   placeholders + refs.
3. **topology_sketch** — tree shape + list cardinalities without lexical leaves
   or binder names.
4. **type_action_multiset** — component-type counts + placeholder inventory.
5. **prompt_intent_fingerprint** — structural words from the prompt with slot
   namespaces stripped.
6. **source_lineage_id** — parent/root + transformation lineage + source/synth.

It also exposes `exact_structure_fingerprint` (existing leakage-normalized hash)
for split-leakage-style checks.

## Files changed

- `src/slm_training/harnesses/train_data/diversity.py` — core module
- `scripts/audit_corpus_diversity.py` — audit runner
- `tests/test_harnesses/train_data/test_diversity.py` — regression tests
- `tests/test_harnesses/train_data/test_quality_report.py` — updated expected
  `harness.train_data` version to `v3`
- `src/slm_training/resources/versions.json` — bumped `harness.train_data` to `v3`
- `docs/design/iter-sde2-04-diversity-fingerprints-20260719.{json,md}` — evidence

## Verification

- `pytest tests/test_harnesses/train_data/test_diversity.py` — 8 passed.
- `.githooks/check-changed` — 198 passed.
- `python -m scripts.verify_version_stamps --check` — ok.
- `python -m scripts.repo_policy` — ok.
- `git diff --check` — clean.
- Sample audit of `src/slm_training/resources/train_seeds.jsonl`: 20 records,
  20 unique canonical roots.

## Honest caveats / remaining work

- This slice provides the fingerprinting primitives only. The full data-economics
  audit, fixed-budget diversity scaling law, nested root sets, and matched
  training experiments remain.
- The prompt-intent fingerprint is a simple normalized token multiset; a more
  robust template/lineage fingerprint will be added once the generator/template
  provenance format stabilizes.
- The binding-aware sketch does not yet compare graphs modulo permitted
  alpha-renaming; it relies on the canonical root ID for equivalence.

## Next steps

1. Integrate fingerprints into `build_train_data()` to emit a diversity report.
2. Run the full data-economics audit over existing corpora.
3. Build nested root sets and the fixed-budget scaling experiment.
