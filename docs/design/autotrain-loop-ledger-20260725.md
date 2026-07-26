# Autotrain loop ledger (fixture smoke)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

Total iterations: **370** (latest `autotrain_wf_smoke_20260726_iter370`).

Iterations 311-340 (2026-07-26) hit a real, reproducible blocker before they
could run at all: a freshly built `wf_smoke_v2` fixture corpus in this
container failed `TwoTowerModel.from_records` with `persisted template
markers must use opaque :slot_<ordinal> identities`. This is the same defect
independently fixed on several sibling never-merged autotrain-smoke-loop
branches (`claude/great-dirac-occ98f`, `-gjrfue`, `-eexsx0`) but never merged
into this lineage; their fix (`_normalize_record` canonicalization) was
reapplied on this branch (`harness.train_data` v21 -> v22), and the
`wf_smoke_v2` fixture corpus itself is now committed under
`src/slm_training/resources/data/train/wf_smoke_v2/` so it no longer needs
regenerating (and re-triggering this class of bug) in every fresh container.
`last_loss` is identical across all 30 rows below because the recipe is fully
deterministic (same fixture, default `--seed 0`, same steps) — not a
copy-paste artifact; see `docs/design/autotrain-wf-smoke-20260726-iter311-measured-results.md`.

## Latest 30

| run_id | ok | steps | stopped_on | last_loss | wall_s |
| --- | --- | --- | --- | --- | --- |
| `autotrain_wf_smoke_20260726_iter341` | True | 8 | steps | 32.610084533691406 | 2.04 |
| `autotrain_wf_smoke_20260726_iter342` | True | 8 | steps | 32.610084533691406 | 2.09 |
| `autotrain_wf_smoke_20260726_iter343` | True | 8 | steps | 32.610084533691406 | 2.18 |
| `autotrain_wf_smoke_20260726_iter344` | True | 8 | steps | 32.610084533691406 | 2.24 |
| `autotrain_wf_smoke_20260726_iter345` | True | 8 | steps | 32.610084533691406 | 2.21 |
| `autotrain_wf_smoke_20260726_iter346` | True | 8 | steps | 32.610084533691406 | 2.36 |
| `autotrain_wf_smoke_20260726_iter347` | True | 8 | steps | 32.610084533691406 | 2.35 |
| `autotrain_wf_smoke_20260726_iter348` | True | 8 | steps | 32.610084533691406 | 2.25 |
| `autotrain_wf_smoke_20260726_iter349` | True | 8 | steps | 32.610084533691406 | 2.23 |
| `autotrain_wf_smoke_20260726_iter350` | True | 8 | steps | 32.610084533691406 | 2.21 |
| `autotrain_wf_smoke_20260726_iter351` | True | 8 | steps | 32.610084533691406 | 2.24 |
| `autotrain_wf_smoke_20260726_iter352` | True | 8 | steps | 32.610084533691406 | 2.24 |
| `autotrain_wf_smoke_20260726_iter353` | True | 8 | steps | 32.610084533691406 | 2.11 |
| `autotrain_wf_smoke_20260726_iter354` | True | 8 | steps | 32.610084533691406 | 2.05 |
| `autotrain_wf_smoke_20260726_iter355` | True | 8 | steps | 32.610084533691406 | 2.4 |
| `autotrain_wf_smoke_20260726_iter356` | True | 8 | steps | 32.610084533691406 | 2.41 |
| `autotrain_wf_smoke_20260726_iter357` | True | 8 | steps | 32.610084533691406 | 2.1 |
| `autotrain_wf_smoke_20260726_iter358` | True | 8 | steps | 32.610084533691406 | 2.52 |
| `autotrain_wf_smoke_20260726_iter359` | True | 8 | steps | 32.610084533691406 | 2.19 |
| `autotrain_wf_smoke_20260726_iter360` | True | 8 | steps | 32.610084533691406 | 2.32 |
| `autotrain_wf_smoke_20260726_iter361` | True | 8 | steps | 32.610084533691406 | 2.52 |
| `autotrain_wf_smoke_20260726_iter362` | True | 8 | steps | 32.610084533691406 | 2.15 |
| `autotrain_wf_smoke_20260726_iter363` | True | 8 | steps | 32.610084533691406 | 2.21 |
| `autotrain_wf_smoke_20260726_iter364` | True | 8 | steps | 32.610084533691406 | 2.27 |
| `autotrain_wf_smoke_20260726_iter365` | True | 8 | steps | 32.610084533691406 | 2.2 |
| `autotrain_wf_smoke_20260726_iter366` | True | 8 | steps | 32.610084533691406 | 2.15 |
| `autotrain_wf_smoke_20260726_iter367` | True | 8 | steps | 32.610084533691406 | 2.48 |
| `autotrain_wf_smoke_20260726_iter368` | True | 8 | steps | 32.610084533691406 | 2.35 |
| `autotrain_wf_smoke_20260726_iter369` | True | 8 | steps | 32.610084533691406 | 2.11 |
| `autotrain_wf_smoke_20260726_iter370` | True | 8 | steps | 32.610084533691406 | 2.03 |
