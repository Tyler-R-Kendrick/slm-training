# autotrain_wf_smoke_20260725_iter251

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_v1 last_loss=38.951087951660156 stopped_on=steps wall=2.374531679000029 max_wall=2.5833333333333335 n=3

## Harness fix required before this iteration

Rebuilding the `wf_smoke_v1` train fixture from scratch in this container hit
the canonical-marker gate:

```
ValueError: persisted template markers must use opaque :slot_<ordinal> identities
```

`_normalize_record` in `src/slm_training/harnesses/train_data/pipeline.py`
persisted named markers (e.g. `:hero.title`) instead of canonicalizing them to
opaque `:slot_N` before returning, unlike its `test_data` mirror —
`TwoTowerModel.from_records` rejects named markers unconditionally, so every
fresh-fixture SFT run in a new container failed at load time. A fix for this
exact defect already existed on an orphaned, never-merged commit pair
(`d537485e`, `71cfff72`) from a prior session's branch that was not an
ancestor of this branch's history; reapplied the same fix here (both
`_normalize_record` return paths now run
`canonicalize_example_template_markers` + `assert_canonical_template_markers`
before returning) and bumped `harness.train_data` v20 -> v21
(`src/slm_training/resources/versions.json`).

Three tests whose fixtures baked in the leaky named-marker behavior were
updated to assert the canonical `:slot_N` spelling instead
(`tests/test_harnesses/train_data/test_pipeline.py`):
`test_prompt_contracts_expose_component_counts_and_slots`,
`test_semantic_role_contract_uses_only_visible_slots_and_types`,
`test_build_train_data_from_rico_fixtures` (RICO fixture slice widened from
`rico_limit=10` to `rico_limit=80` since canonical markers correctly collapse
more near-duplicate structural templates, so the old `record_count >= 5`
threshold at `rico_limit=10` is not reliably achievable).

`tests/test_harnesses/train_data/test_source_families.py::test_pipeline_manifest_source_families`
and 4 cases in `test_staged_materialization.py` fail on this branch's HEAD
independent of this change (`generator version mismatch for
'pack.corpus_generator': plan='v18', active='v20'` before this patch,
`active='v21'` after) — confirmed pre-existing via `git stash`; out of scope
for this iteration.

## Phase status

| Phase | Status |
| --- | --- |
| data_train | rebuilt (fresh container; artifacts not present) |
| data_test | rebuilt (fresh container; artifacts not present) |
| quality | ok |
| sft | ok (after harness fix) |
| eval | ok (wiring; criteria fail expected) |
| closeout | ok |
| verify | ok |

## Eval (suite=smoke, n=3, no --ship-gates)

| metric | value |
| --- | --- |
| meaningful_program_rate | 0.3333333333333333 |
| decode_timeout_count | 0 |
| placeholder_fidelity | 0.5277777777777778 |
| structural_similarity | 0.15416666666666667 |
| component_type_recall | 0.25 |
| reward_score | 0.7793333333333333 |

## Artifacts

- `outputs/data/train/wf_smoke_v1`
- `outputs/data/eval/wf_smoke_v1`
- `outputs/runs/autotrain_wf_smoke_20260725_iter251/train_summary.json`
- `outputs/runs/autotrain_wf_smoke_20260725_iter251/checkpoints/last.pt`
- `outputs/runs/autotrain_wf_smoke_20260725_iter251/scoreboard.json`

`gates.json` is only written with `--ship-gates` (not this recipe). `outputs/`
is ephemeral per container; artifacts above are not committed.
