# RICO domain-shift audit

SLM-265 freezes deterministic, outcome-blind prompt and program-distance strata. The committed [disposition JSON](iter-slm265-domain-shift-audit-20260724.json) records the complete-manifest hash, recipe, and suite summary; rerun the exact command to regenerate the row-level manifest.

```bash
python -m scripts.audit_domain_shift --train-records src/slm_training/resources/data/train/remediated/records.jsonl --eval-dir src/slm_training/resources/data/eval/remediated --out outputs/audits/slm265-domain-shift.json
```

## Measured local feature audit — 2026-07-24

The bounded CPU audit used 585 committed training records and 19 committed evaluation records. It used lexical TF-IDF prompt distance plus canonical-root, binding, and topology fingerprints, froze near/mid/far strata before loading any model outcomes, and found no exact leakage in the 19 rows.

All three fixture `rico_held` records are near under this descriptive manifest. This is **not** evidence that covariate shift explains RICO failure: no durable checkpoint references exist for a frozen re-evaluation, and SLM-263 has only 33 smoke replay rows across eight checkpoint hashes. The disposition is therefore **`insufficient_power_or_provenance`**.

The full `rico_held` ship bar is unchanged. Near-stratum metrics cannot replace it, and no human rating, checkpoint, model training, or serving-default change occurred.
