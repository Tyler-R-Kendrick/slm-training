# Certified eval suites from the certified corpus (P7, 2026-09-02)

JSON mirror: [`certified-eval-suites-20260902.json`](certified-eval-suites-20260902.json)
(stamped `data.test_build v9`). Claim class: **data infrastructure /
fixture-demo** — no model was trained, evaluated or promoted here.

## Problem

The screening "data heal" grew the smoke suite from
`_EXTRA_SMOKE_FIXTURES`, a hand-written tuple of 21 seeds that were all
already in use, so every deficit request returned nothing; `TARGET_SMOKE_N
= 24` was the pool ceiling. The climb trained on `wf_smoke_v2` (101
augmentations of the same 21 seeds) while `openui_verified_v1` (1,682
certified records) and `hillclimb_strict_v2` (676) sat unused, and neither
train set carried leakage fingerprints against the certified eval families.

## What was built

| Piece | Where |
| --- | --- |
| Root-family sampler over the certified corpus | `src/slm_training/harnesses/test_data/certified.py` |
| `build_test_data --source certified` (+ `--publish`, sidecars) | `scripts/build_test_data.py`, `harnesses/test_data/pipeline.py` |
| Train bucket materializer | `scripts/build_certified_train_bucket.py` |
| Deficit growth uses the sampler | `autoresearch/screening_sample_size.py::extra_smoke_fixtures_for_deficit` |
| Published train bucket | `src/slm_training/resources/data/train/openui_verified_train_v1/` |
| Published suites | `.../data/eval/e938_role_safe_all_targets_smoke96_v1/`, `..._heldout24_v1/` |
| Leakage test | `tests/test_harnesses/test_data/test_certified.py` |

Split: `RootFamilySplitPolicyV1` (sha256 of the family id mod 100; 0-79
train, 80-89 validation → smoke, 90-99 test → held_out). Families are the
connected components of `id / root_parent_id / split_group_id / parent_id`,
closed under identical program text (raw at assignment, normalized after
admission — see below). Every record goes through the same `_normalize` as
the fixture path (structure strip, enforce-mode sanitize, `symbol_only/v2`
contract); failures are rejected and ledgered, never patched. Exact
normalized prompt/program duplicates are dropped; distinct variants that
share a source id are re-identified as `<id>__<pair-sha8>`. Candidates are
decontaminated with `find_leakage` (id, split group, prompt, program,
structure, pair) against the train-bucket manifest, then selected
deterministically: round-robin over `source` strata, inside a stratum
unseen structure → unseen source id → seeded hash. Ids already in the
target suite (or reserved by a sibling suite) are never returned.

## Counts

Corpus `openui_verified_v1`: 1,682 rows, 700 distinct ids, 190 link
families.

| Partition variant | Families | train / validation / test (records) | families | Eligible smoke / held_out | Decontaminated |
| --- | --- | --- | --- | --- | --- |
| links only (first build) | 190 | 1102 / 190 / 125 | 135 / 25 / 20 | 172 / 104 | 18 smoke + 21 held_out = 39 |
| + raw program text | 167 | 1103 / 187 / 127 | 116 / 22 / 19 | 172 / 104 | 15 + 23 = 38 |
| **+ normalized program text (published)** | **120** | **1083 / 198 / 136** | **84 / 16 / 10** | **197 / 136** | **1 (prompt-only) + 0** |

Published suites (seed 0, train manifest `openui_verified_train_v1`,
`leakage_rejected = 0`, `error_count = 0`):

| Suite | n | Source stratification | Distinct source ids / structures / families |
| --- | --- | --- | --- |
| `smoke96_v1` smoke | 96 | corruption_repair 16, fixture 16, language_contract 16, language_contract+aug 15, programspec_generated 5, programspec_generated+aug 11, programspec_generated+design_md_contrastive 1, programspec_generated+template 3, rico 9, scope_canonical_document 2, scope_identity_document 2 | 62 / 42 / 16 (all 16 validation families) |
| `heldout24_v1` held_out (also carried by `smoke96_v1`) | 24 | corruption_repair 2, fixture 2, language_contract 2, language_contract+aug 2, programspec_generated 1, programspec_generated+aug 2, programspec_generated+design_md_contrastive 1, programspec_generated+template 2, rico 2, scope_canonical_document 2, scope_identity_document 2, web_distilled 1, web_distilled+aug 1, web_distilled+design_md_contrastive 1, web_distilled+template 1 | 22 / 13 / 7 (of 10 test families) |

Train bucket `openui_verified_train_v1`: 1,083 records, 84 families, all
with `design_md`; manifest carries `ids`, `root_family_ids`,
`split_group_ids` and prompt / program / structure / pair / design_md
fingerprint lists, so `build_test_data --train-manifest` and the leakage
test consume it like any `build_train_data` snapshot.

### Screening sample-size sidecars (`screening_sample_size.json`)

Computed with `compute_screening_sample_size` at the exact decidability
floor: alpha 1/20, arm wall 70 s (symmetric two-stage wall), decode floor
2 s, usable wall 42 s → budget ceiling 21; sign-test floor **6**; power
floor **not declared** (no measured paired SD for the screening primary —
never a borrowed SD); `chosen_n = 6`, verdict `feasible`,
`promotion_authority = false` for both suites (suite ceilings 96 and 24).
The held-out sidecar names `suite = held_out` (it previously reported
`smoke_n = 0` / `must_generate`, which was false for a held-out-only build).

## Rejection histogram (data-quality law)

From `openui_verified_train_v1/quality_report.json` / `rejected.jsonl`:

| Reason | Total | train | validation | test |
| --- | --- | --- | --- | --- |
| `OutputContractError` (`symbol_only/v2`) | 90 | 56 | 34 | 0 |
| `duplicate_pair` (exact normalized prompt+program) | 175 | 137 | 20 | 18 |

Re-identified id collisions: 1,106 (700 ids for 1,682 rows). Sanitize
fallbacks in the smoke96 build: 5 (`ValueError`, structure-only fallback;
0 in heldout24). Nothing was patched; no gate was weakened.

### Builder improvements made from that evidence

1. **Normalized program-text family closure.** The first build discarded
   39 eval candidates whose normalized program also existed in the train
   bucket under a different recorded family; closing under raw text
   recovered almost nothing (38 still discarded) because the collisions
   only appear after style stripping and sanitizing. Admitted records are
   now re-closed under normalized `fingerprint_openui`; link families
   190 → 120, discards 39 → 1 (a prompt-only match, correctly rejected),
   eligible pools 172 → 197 and 104 → 136, smoke covers 42 distinct
   structures instead of 36.
2. **One partition path.** The bucket-restricted normalization fast path
   could label a record `validation` that the full path had merged into
   `train`; it was removed (partition cached per corpus path/size/mtime,
   ~11 s cold) so the driver's deficit sampler and the build see one
   identical partition.
3. **Honest held-out sidecar** (`suite`, `suite_records`).
4. **Deficit growth decontaminates against the policy train set** as well
   as the certified bucket (`extra_smoke_fixtures_for_deficit`).

Feedback routed to the corpus-certification harness
(`synthesis_feedback.json`, family `corpus_certification`, gates
untouched): `stale_output_contract` (90 records certified before
`symbol_only/v2`), `id_collision` (namespace ids by source snapshot),
`redundant_expansion` (dedupe exact pairs at admission).

## Policy

`policy.v3.json` `defaults.train_version` and
`data_intervention.fixture_train_version` now both read
`openui_verified_train_v1` (were `hillclimb_strict_v2` / `wf_smoke_v2`).
The v2-parity test exempts exactly these two keys.

## Leakage test

`tests/test_harnesses/test_data/test_certified.py`: no root family in both
the train bucket and any published certified suite; smoke and held_out
families disjoint; no exact normalized program overlap and
`find_leakage == []` for every eval record against the bucket manifest;
manifests/ids consistent; sidecars at the exact floor; family closure unit
test; sampler determinism / exclusion / stratification.

## Commands (each ≤ 12 s wall, well under `MAX_RUN_MINUTES`)

```
python -m scripts.build_certified_train_bucket --publish
python -m scripts.build_test_data --source certified --version e938_role_safe_all_targets_smoke96_v1 \
  --suites smoke,held_out --train-manifest src/slm_training/resources/data/train/openui_verified_train_v1/manifest.json --publish
python -m scripts.build_test_data --source certified --version e938_role_safe_all_targets_heldout24_v1 \
  --suites held_out --certified-smoke-n 0 --train-manifest src/slm_training/resources/data/train/openui_verified_train_v1/manifest.json --publish
python -c "from slm_training.autoresearch.screening_sample_size import extra_smoke_fixtures_for_deficit as f; print(len(f(set(), 72)))"  # 72
```

Frozen snapshots (`e938_role_safe_all_targets_v2`, `..._smoke6_v1`) were
not touched; `assert_eval_publish_target_writable` guards every certified
publish.
