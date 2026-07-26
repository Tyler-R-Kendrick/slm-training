# Autotrain loop ledger (fixture smoke)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

Total iterations: **252** (latest `autotrain_wf_smoke_20260725_iter252`).

**iter251 blocker + fix:** a fresh container rebuild of the `wf_smoke_v1`
fixture (ephemeral `outputs/`) tripped the canonical-marker gate again
(`ValueError: persisted template markers must use opaque :slot_<ordinal>
identities`) — the fix from the (unmerged, now-orphaned) `d537485e` /
`71cfff72` line of commits was never in this branch's history. Reapplied the
same fix in `_normalize_record` (`harnesses/train_data/pipeline.py`):
canonicalize + assert opaque `:slot_N` markers on both return paths before
persisting. Bumped `harness.train_data` v20 -> v21 and updated the 3 tests
whose fixtures baked in the leaky named-marker behavior
(`test_prompt_contracts_expose_component_counts_and_slots`,
`test_semantic_role_contract_uses_only_visible_slots_and_types`,
`test_build_train_data_from_rico_fixtures`). See
`docs/design/autotrain-wf-smoke-20260725-iter251-measured-results.md`.

## Latest 30

| run_id | ok | steps | stopped_on | last_loss | wall_s |
| --- | --- | --- | --- | --- | --- |
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
| `autotrain_wf_smoke_20260725_iter251` | True | 8 | steps | 38.951087951660156 | 2.37 |
| `autotrain_wf_smoke_20260725_iter252` | True | 8 | steps | 38.951087951660156 | 6.10 |
