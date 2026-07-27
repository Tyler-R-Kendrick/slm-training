# SLM-289 corpus certification

**Claim class:** local deterministic corpus certification; not a model ship result.

- source snapshots: 21
- source rows: 6337
- canonical Gold/Silver core records: 1682
- Gold: 20; Silver: 2863; Bronze: 2882; Quarantine: 572
- quarantine rate: 9.03% (matched clean-subset control threshold: >10%)
- G12 remains explicit optional evidence and is never a human-rating admission gate.

The immutable `openui_verified_v1` snapshot retains every source row in compact membership manifests, records exact-dedup lineage, and admits only canonical Gold/Silver rows to `records.jsonl`. Bronze and Quarantine rows are excluded from core SFT by construction.

Reproduce from a clean checkout: `python -m scripts.audit_data_corpora --mode certify --output-dir src/slm_training/resources/data/train/openui_verified_v1`. The immutable output refuses overwrite. Opt in to the cleaned snapshot with `--train-version openui_verified_v1`; existing training defaults are unchanged.

Verification: `DataStore.verify('train', 'openui_verified_v1')` validated every declared artifact hash.
