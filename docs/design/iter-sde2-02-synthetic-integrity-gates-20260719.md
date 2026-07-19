# SDE2-02 synthetic-data integrity gates

Date: 2026-07-19  
Issue: SLM-169  
Branch: `agent/slm-169-sde2-02-synthetic-integrity-gates`

## What this slice delivers

A reusable, versioned synthetic-data integrity gate under
`src/slm_training/harnesses/train_data/integrity.py` and an audit script at
`scripts/audit_synthetic_integrity.py`. The gate checks every record for:

- parser/schema/compiler validity;
- canonical AST round-trip (surface → parser → canonical fingerprint);
- production-codec round-trip idempotence;
- choice-codec round-trip idempotence;
- slot-contract consistency (declared vs extracted placeholders);
- binding-graph hash and reference-scope validity;
- optional request-target contract match;
- root-lineage hash;
- optional split/held-out fingerprint leakage.

## Files changed

- `src/slm_training/harnesses/train_data/integrity.py` — core module
- `scripts/audit_synthetic_integrity.py` — audit runner
- `tests/test_harnesses/train_data/test_integrity.py` — regression tests
- `tests/test_harnesses/train_data/test_quality_report.py` — updated expected
  `harness.train_data` version to `v3`
- `src/slm_training/resources/versions.json` — bumped `harness.train_data` to `v3`
- `outputs/runs/sde2-02-integrity-audit/audit-train-seeds-20260719.json` —
  sample audit of committed train seeds
- `docs/design/iter-sde2-02-synthetic-integrity-gates-20260719.md` — this memo

## Verification

- `pytest tests/test_harnesses/train_data/test_integrity.py` — 11 passed.
- `.githooks/check-changed` — 201 passed.
- `python -m scripts.verify_version_stamps --check` — ok.
- `python -m scripts.repo_policy` — ok.
- `git diff --check` — clean.

Sample audit of `src/slm_training/resources/train_seeds.jsonl`:

```json
{
  "n_records": 20,
  "n_passed": 20,
  "n_failed": 0,
  "pass_rate": 1.0
}
```

## Honest caveats / remaining work

- The gate is **not yet wired into `build_train_data()` as a hard admission
  gate**. It runs standalone via `audit_synthetic_integrity.py` and can be
  imported by pipeline stages in a follow-up change.
- A full audit of existing training/validation/test corpora and a matched
  clean-corpus retrain (per SLM-169 acceptance criteria) has not been run.
- Request-target contract matching currently compares slot inventories; richer
  component/role/binding contracts will land as the public request contract
  matures.
- The binding-graph hash is currently a coarse stability hash over unresolved /
  orphaned bindings; a full structural graph comparison with alpha-renaming is
  reserved for the next slice.

## Next steps

1. Wire `evaluate_integrity()` into `build_train_data()` behind an
   `integrity_gate_mode` config (`off`/`audit`/`enforce_new`/`rebuild`).
2. Run the full corpus audit and publish defect rates by source/generator.
3. Build matched clean-corpus training run and report H10 disposition.
