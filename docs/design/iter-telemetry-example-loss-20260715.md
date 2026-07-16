# Per-example token-loss telemetry (2026-07-15)

The TwoTower training path now exposes masked token cross-entropy aggregated
per record for diagnostics only. A two-step accumulation probe produced eight
record IDs and eight aligned proxy values per metrics row. The highest observed
values were `rico_train_96_syn_0` (98.36) and `rico_train_3_aug_dir` (87.20).

The proxy does not alter optimization and excludes batch-level auxiliary
length/fidelity terms. Repeated high-loss IDs should be investigated before
rebalancing or editing the training corpus.
