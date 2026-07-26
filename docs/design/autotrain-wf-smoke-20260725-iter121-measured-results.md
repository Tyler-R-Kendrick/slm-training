# autotrain_wf_smoke_20260725_iter121

**Honesty:** fixture_or_scratch. **Not ship.**

Harness fix landed in this iteration: `harnesses/train_data/pipeline.py` `_normalize_record` now canonicalizes every persisted marker to an opaque `:slot_<ordinal>` identity before returning (mirrors `harnesses/test_data/pipeline.py`). Previously the train-data pipeline never canonicalized markers, so a fresh fixture rebuild persisted named markers (e.g. `:auth.title`) that `TwoTowerModel.from_records` unconditionally rejects — every SFT run on a freshly built `wf_smoke_v2` corpus failed before this fix. `harness.train_data` bumped v20 -> v21 (`src/slm_training/resources/versions.json`). record_count dropped 103 -> 101 vs the last recorded iter120 build: 2 records now correctly collapse as duplicates once markers are opaque.

train_version=wf_smoke_v2 last_loss=38.951087951660156 stopped_on=steps wall=2.5422888279999825 max_wall=2.5833333333333335 n=3 seed=1 record_count=101
