# Autotrain loop ledger (fixture smoke)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

Total iterations: **250** (latest `autotrain_wf_smoke_20260725_iter250`).

**iter251 attempted 2026-07-26, blocked before training** — fresh-checkout
`slm data build-train --source fixture` output fails `assert_canonical_template_markers`
at SFT time (harness bug, not a training result). Not counted toward the
250 total above; see
[autotrain-wf-smoke-20260726-iter251-blocked.md](autotrain-wf-smoke-20260726-iter251-blocked.md)
for repro, root cause, and why the obvious one-line fix regresses dedup.

## Latest 30

| run_id | ok | steps | stopped_on | last_loss | wall_s |
| --- | --- | --- | --- | --- | --- |
| `autotrain_wf_smoke_20260725_iter221` | True | 8 | steps | 31.807716369628906 | 59.54 |
| `autotrain_wf_smoke_20260725_iter222` | True | 8 | steps | 35.218605041503906 | 66.21 |
| `autotrain_wf_smoke_20260725_iter223` | True | 8 | steps | 31.524505615234375 | 69.02 |
| `autotrain_wf_smoke_20260725_iter224` | True | 8 | steps | 31.552364349365234 | 68.99 |
| `autotrain_wf_smoke_20260725_iter225` | True | 8 | steps | 29.563432693481445 | 65.07 |
| `autotrain_wf_smoke_20260725_iter226` | True | 8 | steps | 31.798160552978516 | 62.94 |
| `autotrain_wf_smoke_20260725_iter227` | True | 8 | steps | 30.436264038085938 | 69.44 |
| `autotrain_wf_smoke_20260725_iter228` | True | 8 | steps | 27.768230438232422 | 69.47 |
| `autotrain_wf_smoke_20260725_iter229` | True | 8 | steps | 31.69894790649414 | 60.56 |
| `autotrain_wf_smoke_20260725_iter230` | True | 8 | steps | 29.74860382080078 | 63.29 |
| `autotrain_wf_smoke_20260725_iter231` | True | 8 | steps | 28.217512130737305 | 66.69 |
| `autotrain_wf_smoke_20260725_iter232` | True | 8 | steps | 29.739126205444336 | 65.28 |
| `autotrain_wf_smoke_20260725_iter233` | True | 8 | steps | 27.394638061523438 | 78.02 |
| `autotrain_wf_smoke_20260725_iter234` | True | 8 | steps | 25.815250396728516 | 64.54 |
| `autotrain_wf_smoke_20260725_iter235` | True | 8 | steps | 34.30736541748047 | 76.11 |
| `autotrain_wf_smoke_20260725_iter236` | True | 8 | steps | 29.511457443237305 | 64.03 |
| `autotrain_wf_smoke_20260725_iter237` | True | 8 | steps | 34.949222564697266 | 74.87 |
| `autotrain_wf_smoke_20260725_iter238` | True | 8 | steps | 39.26030731201172 | 67.83 |
| `autotrain_wf_smoke_20260725_iter239` | True | 8 | steps | 37.69666290283203 | 68.14 |
| `autotrain_wf_smoke_20260725_iter240` | True | 8 | steps | 22.895488739013672 | 67.49 |
| `autotrain_wf_smoke_20260725_iter241` | True | 8 | steps | 33.473480224609375 | 73.39 |
| `autotrain_wf_smoke_20260725_iter242` | True | 8 | steps | 32.77983093261719 | 70.44 |
| `autotrain_wf_smoke_20260725_iter243` | True | 8 | steps | 30.764629364013672 | 67.49 |
| `autotrain_wf_smoke_20260725_iter244` | True | 8 | steps | 35.58620834350586 | 69.47 |
| `autotrain_wf_smoke_20260725_iter245` | True | 8 | steps | 33.59168243408203 | 70.02 |
| `autotrain_wf_smoke_20260725_iter246` | True | 8 | steps | 25.17486572265625 | 81.83 |
| `autotrain_wf_smoke_20260725_iter247` | True | 8 | steps | 41.349456787109375 | 73.56 |
| `autotrain_wf_smoke_20260725_iter248` | True | 8 | steps | 35.379425048828125 | 74.12 |
| `autotrain_wf_smoke_20260725_iter249` | True | 8 | steps | 26.19162368774414 | 68.13 |
| `autotrain_wf_smoke_20260725_iter250` | True | 8 | steps | 27.758634567260742 | 82.96 |
