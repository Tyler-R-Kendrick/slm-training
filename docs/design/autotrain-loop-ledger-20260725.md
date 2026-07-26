# Autotrain loop ledger (fixture smoke)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

Total iterations: **561** (latest `autotrain_wf_smoke_20260726_iter561`).

Iterations 552-561 are real measured `scripts.train_model` runs (twotower,
CPU, `DEFAULT_TRAIN_DATA_DIR`, 8 steps, `--no-full-state-checkpoint`), each
with a JSON+markdown pair under `docs/design/` carrying the writer-emitted
`version_stamp` — closing the Iron Law JSON-companion gap left by earlier
rows in this ledger (markdown-only, no JSON sidecar; iter551's row in
particular has no matching measured-results doc at all and predates this
gap fix). No ship claim; `fixture_or_scratch` wiring only.

## Latest 30

| run_id | ok | steps | stopped_on | last_loss | wall_s |
| --- | --- | --- | --- | --- | --- |
| `autotrain_wf_smoke_20260725_iter532` | True | 8 | steps | 23.04449462890625 | 2.8 |
| `autotrain_wf_smoke_20260725_iter533` | True | 8 | steps | 35.9559211730957 | 2.98 |
| `autotrain_wf_smoke_20260725_iter534` | True | 8 | steps | 29.153064727783203 | 2.4 |
| `autotrain_wf_smoke_20260725_iter535` | True | 8 | steps | 37.61457824707031 | 2.42 |
| `autotrain_wf_smoke_20260725_iter536` | True | 8 | steps | 35.899044036865234 | 2.58 |
| `autotrain_wf_smoke_20260725_iter537` | True | 8 | steps | 31.397918701171875 | 2.24 |
| `autotrain_wf_smoke_20260725_iter538` | True | 8 | steps | 34.47773361206055 | 2.28 |
| `autotrain_wf_smoke_20260725_iter539` | True | 8 | steps | 27.201704025268555 | 2.08 |
| `autotrain_wf_smoke_20260725_iter540` | True | 8 | steps | 30.195186614990234 | 2.21 |
| `autotrain_wf_smoke_20260725_iter541` | True | 8 | steps | 40.50148391723633 | 2.06 |
| `autotrain_wf_smoke_20260725_iter542` | True | 8 | steps | 38.233333587646484 | 2.9 |
| `autotrain_wf_smoke_20260725_iter543` | True | 8 | steps | 27.086015701293945 | 3.89 |
| `autotrain_wf_smoke_20260725_iter544` | True | 8 | steps | 28.21722412109375 | 3.19 |
| `autotrain_wf_smoke_20260725_iter545` | True | 8 | steps | 38.80730438232422 | 2.75 |
| `autotrain_wf_smoke_20260725_iter546` | True | 8 | steps | 26.906417846679688 | 2.42 |
| `autotrain_wf_smoke_20260725_iter547` | True | 8 | steps | 32.99607849121094 | 2.43 |
| `autotrain_wf_smoke_20260725_iter548` | True | 8 | steps | 27.587923049926758 | 2.2 |
| `autotrain_wf_smoke_20260725_iter549` | True | 8 | steps | 31.088762283325195 | 2.13 |
| `autotrain_wf_smoke_20260725_iter550` | True | 8 | steps | 27.85626792907715 | 2.23 |
| `autotrain_wf_smoke_20260725_iter551` | True | 8 | steps | 33.277069091796875 | 2.12 |
| `autotrain_wf_smoke_20260726_iter552` | True | 8 | steps | 35.1600341796875 | 3.24 |
| `autotrain_wf_smoke_20260726_iter553` | True | 8 | steps | 40.51862716674805 | 3.48 |
| `autotrain_wf_smoke_20260726_iter554` | True | 8 | steps | 33.008544921875 | 3.34 |
| `autotrain_wf_smoke_20260726_iter555` | True | 8 | steps | 25.35677146911621 | 3.38 |
| `autotrain_wf_smoke_20260726_iter556` | True | 8 | steps | 33.0897216796875 | 4.06 |
| `autotrain_wf_smoke_20260726_iter557` | True | 8 | steps | 35.35194396972656 | 3.76 |
| `autotrain_wf_smoke_20260726_iter558` | True | 8 | steps | 26.30024528503418 | 3.6 |
| `autotrain_wf_smoke_20260726_iter559` | True | 8 | steps | 32.689674377441406 | 3.6 |
| `autotrain_wf_smoke_20260726_iter560` | True | 8 | steps | 33.653297424316406 | 3.26 |
| `autotrain_wf_smoke_20260726_iter561` | True | 8 | steps | 46.510379791259766 | 3.51 |
