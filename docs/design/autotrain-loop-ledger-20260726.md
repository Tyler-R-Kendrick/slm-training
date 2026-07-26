# Autotrain loop ledger (fixture smoke) — 2026-07-26

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

Continuation of [`autotrain-loop-ledger-20260725.md`](autotrain-loop-ledger-20260725.md)
in a fresh container (no cached `outputs/`), which surfaced and fixed a real
`build_train_data` marker-canonicalization regression — see
[`autotrain-wf-smoke-20260726-measured-results.md`](autotrain-wf-smoke-20260726-measured-results.md).

| run_id | ok | steps | stopped_on | last_loss | wall_s (sft) | max_wall | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `autotrain_wf_smoke_20260726` | True | 8 | steps | 33.309722900390625 | 1.84 | 2.5833333333333335 | fresh build; fixed `harness.train_data` marker-canonicalization bug (v20→v21) |
