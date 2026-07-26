# Autotrain workflow smoke — measured results (autotrain_wf_smoke_20260726)

**Honesty:** `fixture_or_scratch` wiring only. **Not a ship claim.**

Full JSON: [`autotrain-wf-smoke-20260726-results.json`](autotrain-wf-smoke-20260726-results.json).

## Headline: fresh smoke build was broken, root-caused, and fixed

Continuing the `docs/design/autotrain-loop-ledger-20260725.md` smoke loop, this
run started from a clean container with no cached `outputs/`, so it exercised
the **fresh** `slm data build-train` path instead of reusing an
already-canonical fixture corpus. It failed immediately:

```
ValueError: persisted template markers must use opaque :slot_<ordinal> identities
```

Root cause: `build_train_data`
(`src/slm_training/harnesses/train_data/pipeline.py`) never canonicalized
persisted records' template markers to opaque `:slot_<ordinal>` identities
before writing `records.jsonl`/`manifest.json`; `build_test_data` already did
this (`harnesses/test_data/pipeline.py:119`). All 103/103 fixture records
therefore persisted named markers (e.g. `:auth.title`) and unconditionally
failed `assert_canonical_template_markers` at SFT load time
(`TwoTowerModel.from_records`). The 2026-07-25 smoke loop (iter1-iter50) never
hit this because every iteration after the first reused the one fixture
corpus built at the very start of that session (`data_train`/`data_test`
reported `skipped (artifacts present)` throughout).

Fix (`harness.train_data` v20 → v21,
`src/slm_training/resources/versions.json`):

- Canonicalize the final record set via `canonicalize_example_template_markers`
  (+ `assert_canonical_template_markers` safety net) once, **after**
  decontamination/dedup/curation-score gates (which need the more distinctive
  named-marker text — canonicalizing earlier collapsed real diversity into
  generic `:slot_N` n-grams and cratered RICO yield in testing) and **before**
  the prompt-contract enrichments (whose `ensure_prompt_inventory` /
  `semantic_role_contract` already assert canonical-only placeholders).
- Canonicalize both sides of the `--dedup-against` cross-corpus fingerprint
  comparison so it compares corpora in the same opaque-marker form they are
  persisted in.
- `staged_materialization.validate_staged_record` now validates a locally
  canonicalized copy (GenerationRequest/SymbolTable/tokenizer round-trip all
  require opaque markers) without mutating the record the caller persists.
- Refreshed the stale `dsh0_cap0_fixture.json` synthesis-plan pin (generator/
  validator/gate versions v18/v18/v1/`openui_ship_gates_v2` → v21/v21/v4/
  `openui_ship_gates_v4`) — it had drifted stale across several unrelated
  version bumps and was unconditionally failing `require_executable()`,
  masking a second bug once fixed (see below). `harness.synthesis_plan`
  no-bump history note.
- Two prompt-contract tests and one staged-materialization test asserted
  named-marker/dead-flag behavior that the opaque-marker invariant already
  made unreachable; updated their expectations to the correct canonical
  output (`tests/test_harnesses/train_data/test_pipeline.py`,
  `test_scope_corpus.py`, `test_source_families.py`).

Tests: `tests/test_harnesses/train_data` + `tests/test_data` + `test_versioning`
— 22 pre-existing failures before this change; 15 after (7 net fixed: the
canonicalization crash, the cross-corpus dedup regression it caused, the
stale synthesis-plan pin it unmasked, and the staged-validation opaque-marker
crash it unmasked in turn), 0 new failures. The remaining 15 are unrelated to
markers/dedup/staged-materialization (RICO progspec-topology API drift,
semantic-contrast fixtures, one content-hash-pinned staged-materialization
golden test) and are out of scope for this fix.

## Run

| Field | Value |
| --- | --- |
| recipe | smoke |
| run_id | `autotrain_wf_smoke_20260726` |
| train | fixture (fresh build, 103 records, `wf_smoke_20260726`) |
| test | fixture disjoint suite (`wf_smoke_20260726`, smoke n=3) |
| model | twotower / choice / scratch / cpu |
| steps | 8 (`--fast-train`, `--no-sync-checkpoints`) |
| seed | 1 |
| ship_gates | false (`--eval-limit 3 --suites smoke`) |
| train last_loss | 33.309722900390625 |
| stopped_on | steps |
| elapsed_wall_seconds (sft) | 1.84 |
| max_wall_minutes | 2.5833333333333335 |
| smoke n | 3 (diagnostic subset) |
| meaningful_program_rate | 0.0 |
| structural_similarity | 0.25193333333333334 |
| placeholder_fidelity | 0.0 |
| decode_timeout_count | 0 |
| AgentV criteria.pass | false (expected at 8 steps from scratch) |

## Phase status

| Phase | Status |
| --- | --- |
| data_train | ok (fresh build; fixed the marker-canonicalization regression) |
| data_test | ok (fresh build) |
| sft | ok |
| eval | ok (wiring; criteria fail expected at 8 steps) |
| closeout | ok (this doc + ledger + version bump) |
| verify | ok (tests rerun; no new failures) |

## Artifacts

- `outputs/data/train/wf_smoke_20260726/manifest.json`
- `outputs/data/train/wf_smoke_20260726/quality_report.json`
- `outputs/data/eval/wf_smoke_20260726`
- `outputs/runs/autotrain_wf_smoke_20260726/train_summary.json`
- `outputs/runs/autotrain_wf_smoke_20260726/checkpoints/last.pt`
- `outputs/runs/autotrain_wf_smoke_20260726/scoreboard.json`

`gates.json` is only written with `--ship-gates` (not this recipe).
