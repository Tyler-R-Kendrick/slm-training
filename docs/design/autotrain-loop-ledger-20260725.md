# Autotrain loop ledger (fixture smoke)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

Total iterations: **340** (latest `autotrain_wf_smoke_20260726_iter340`).

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
| `autotrain_wf_smoke_20260726_iter311` | True | 8 | steps | 32.610084533691406 | 2.2 |
| `autotrain_wf_smoke_20260726_iter312` | True | 8 | steps | 32.610084533691406 | 2.27 |
| `autotrain_wf_smoke_20260726_iter313` | True | 8 | steps | 32.610084533691406 | 2.21 |
| `autotrain_wf_smoke_20260726_iter314` | True | 8 | steps | 32.610084533691406 | 2.41 |
| `autotrain_wf_smoke_20260726_iter315` | True | 8 | steps | 32.610084533691406 | 2.19 |
| `autotrain_wf_smoke_20260726_iter316` | True | 8 | steps | 32.610084533691406 | 2.27 |
| `autotrain_wf_smoke_20260726_iter317` | True | 8 | steps | 32.610084533691406 | 2.39 |
| `autotrain_wf_smoke_20260726_iter318` | True | 8 | steps | 32.610084533691406 | 2.22 |
| `autotrain_wf_smoke_20260726_iter319` | True | 8 | steps | 32.610084533691406 | 2.23 |
| `autotrain_wf_smoke_20260726_iter320` | True | 8 | steps | 32.610084533691406 | 2.06 |
| `autotrain_wf_smoke_20260726_iter321` | True | 8 | steps | 32.610084533691406 | 2.16 |
| `autotrain_wf_smoke_20260726_iter322` | True | 8 | steps | 32.610084533691406 | 2.27 |
| `autotrain_wf_smoke_20260726_iter323` | True | 8 | steps | 32.610084533691406 | 2.25 |
| `autotrain_wf_smoke_20260726_iter324` | True | 8 | steps | 32.610084533691406 | 2.11 |
| `autotrain_wf_smoke_20260726_iter325` | True | 8 | steps | 32.610084533691406 | 2.13 |
| `autotrain_wf_smoke_20260726_iter326` | True | 8 | steps | 32.610084533691406 | 2.1 |
| `autotrain_wf_smoke_20260726_iter327` | True | 8 | steps | 32.610084533691406 | 2.15 |
| `autotrain_wf_smoke_20260726_iter328` | True | 8 | steps | 32.610084533691406 | 2.27 |
| `autotrain_wf_smoke_20260726_iter329` | True | 8 | steps | 32.610084533691406 | 2.1 |
| `autotrain_wf_smoke_20260726_iter330` | True | 8 | steps | 32.610084533691406 | 2.48 |
| `autotrain_wf_smoke_20260726_iter331` | True | 8 | steps | 32.610084533691406 | 2.26 |
| `autotrain_wf_smoke_20260726_iter332` | True | 8 | steps | 32.610084533691406 | 2.02 |
| `autotrain_wf_smoke_20260726_iter333` | True | 8 | steps | 32.610084533691406 | 2.06 |
| `autotrain_wf_smoke_20260726_iter334` | True | 8 | steps | 32.610084533691406 | 2.12 |
| `autotrain_wf_smoke_20260726_iter335` | True | 8 | steps | 32.610084533691406 | 2.05 |
| `autotrain_wf_smoke_20260726_iter336` | True | 8 | steps | 32.610084533691406 | 2.04 |
| `autotrain_wf_smoke_20260726_iter337` | True | 8 | steps | 32.610084533691406 | 2.29 |
| `autotrain_wf_smoke_20260726_iter338` | True | 8 | steps | 32.610084533691406 | 2.23 |
| `autotrain_wf_smoke_20260726_iter339` | True | 8 | steps | 32.610084533691406 | 2.24 |
| `autotrain_wf_smoke_20260726_iter340` | True | 8 | steps | 32.610084533691406 | 2.25 |
