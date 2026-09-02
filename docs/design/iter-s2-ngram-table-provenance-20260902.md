# S2: speculative n-gram table provenance and pinned branch-point numbers

**Status: `closed`** — provenance repaired and pinned; hypothesis N1 **not
supported** as stated. Claim class: `fixture` (fixture-suite diagnostic on a
default-`off` lever; never a ship, checkpoint, or production-default claim).
The `speculative_rank` lever stays `off` and `speculative_rank_margin` stays
`0.0` (never commit).

- Card: S2 (n-gram table provenance). Decode invariant: I3
  ([`decode-invariants.md`](decode-invariants.md), "The committed table").
- JSON mirror: [`iter-s2-ngram-table-provenance-20260902.json`](iter-s2-ngram-table-provenance-20260902.json).
- Builder: `scripts/build_speculative_ngram_table.py`; artifact:
  `src/slm_training/resources/decode/speculative_ngram_v1.json`; pin:
  `tests/test_dsl/test_speculative_rank.py::test_committed_table_pins_the_documented_branch_points`.

## What was wrong

`decode-invariants.md` claimed the committed table was built from
`openui_verified_v1` (1,682 records, 89,415 tokens, order 3, 523 contexts) and
that after `root = ` it picks `Stack(` at margin 1.0 and after
`root = Stack([` picks `<BIND_1>` at margin 1.59. None of that was true of the
artifact in the tree:

| | doc claim (before) | artifact (before, measured) | artifact (after, measured) |
| --- | --- | --- | --- |
| corpus | `openui_verified_v1` | `wf_smoke_v2` (builder default fell back to the smoke fixture) | `openui_verified_train_v1` (certified TRAIN bucket, root-family buckets 0–79) |
| records / sequences | 1,682 | 101 | 1,083 records, 1,054 encoded (29 refused by the symbol codec: placeholder in a non-content property) |
| tokens | 89,415 | 4,633 | 54,434 |
| order / contexts | 3 / 523 | 4 / 538 | 3 / 493 |
| `root = ` (n candidates → pick @ margin) | 27 → `Stack(` @ 1.0 | 27 → `Stack(` @ 1.941 | 27 → `Stack(` @ 15.000 |
| `root = Stack([` | 25 → `<BIND_1>` @ 1.59 | 26 → `b1` @ 15.916 | 26 → `b1` @ 1.738 |

No test pinned the numbers and `--check` only compared the artifact to a
rebuild from the same (wrong) default, so the drift was invisible to CI.

## What changed

1. **Builder default = certified train bucket.** `--records` defaults to
   `src/slm_training/resources/data/train/openui_verified_train_v1/records.jsonl`
   (card P7's certified TRAIN bucket, decontaminated from the validation/test
   root-family buckets by its own build). The builder no longer re-derives a
   split and no longer has a `wf_smoke_v2` fallback. It still refuses eval
   manifests / eval-split records and keeps only declared-`train` records.
2. **Provenance in the artifact.** The table carries a `source` block:
   `dataset_id`, `manifest_kind`, `manifest_content_fingerprint`
   (`8fe079f51d269ab213d5083d3b693ec6272839f6934a449d8b7d0eb34f22b59c`),
   `records` (repo-relative path), `records_sha256`
   (`7cf64cdda534a1121a228128732cf3a5920981d647af7b90275f793c443ad6ee`, which
   the bucket manifest certifies) and `records_total`. `NgramTableV1.from_dict`
   ignores the extra key, so the ranker sees the same table.
3. **`--check` binds artifact to manifest.** It now fails when (a) the records
   file on disk does not match the sha256 its manifest certifies, (b) the
   artifact's `source` block differs from the live manifest, (c) the recorded
   `corpus_fingerprint` (`96ef8ffd409637f53fa9e79abe2f96b4df79ba260e428862c35732c3ac3f08e5`)
   is not the rebuilt train-sequence fingerprint, or (d) any other field drifts
   from the rebuild. Verified: tampering `manifest_content_fingerprint` in a
   copy of the artifact → exit 1 naming the drifted field.
4. **Order stays 3.** See the order 3 vs 4 arm below: order 4 doubles the
   context count (1,239 vs 493) and raises the share of floor-saturated
   margins (≥15: 207 vs 129 of 650 pooled branch points) for a hit-rate
   difference inside noise (+6 of 213 on held_out+adversarial).
5. **Doc numbers are measured and pinned** (test above, margin ± 0.01).

## Hypothesis N1 and the measurement

**N1:** the smoke-built (`wf_smoke_v2`) table is fixture-overfit — it ranks the
smoke suite's branch points more confidently than held-out ones and a
certified-corpus table generalizes better.

**Method.** For every record of the `smoke` (24), `held_out` (5) and
`adversarial` (4) suites of
`src/slm_training/resources/data/eval/e938_role_safe_all_targets_smoke24_v1`,
templatize and encode the target in the decoder codec, then at every token
position build the completion forest (`build_completion_forest`,
`remaining_tokens=32`). A **branch point** is a position whose forest is
`complete` and offers ≥ 2 non-empty paths (710 of 1,557 positions; the rest are
forced or partial). At each branch point every table ranks the same legal
paths with `SpeculativeRankerV1(margin=0.5)` (the threshold the existing test
uses; the serving default is `0.0`, never commit). Recorded per table: the
margin, whether the pick is a prefix of the gold continuation (`hit`; equal to
first-token agreement on every branch point here), and whether the ranker
would have committed (`margin ≥ 0.5`). CPU only; sharded into ≤ 3-minute runs
(`MAX_RUN_MINUTES` respected; four 2-record shards timed out and were re-run
as single-record / position-range shards, so every record is covered exactly
once).

Tables: `wf_smoke_v2_order4` = the previously committed artifact;
`certified_train_order3` = the new committed artifact;
`certified_train_order4` = the same bucket at order 4 (control arm for the
order choice, not committed).

### Results (all 33 records, 710 branch points)

| suite (records / branch points) | table | margin p10 / p25 / median / p75 / p90 | ≥15 (floor) | commit rate @0.5 | pick hit rate | confident precision | wrong commits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smoke (24 / 497) | wf_smoke_v2 o4 | 0.063 / 0.405 / 1.386 / 15.916 / 30.0 | 159 | 0.678 | 0.513 | 0.677 | 109 |
| | certified o3 | 0.188 / 0.599 / 1.382 / 2.983 / 15.458 | 94 | 0.835 | 0.479 | 0.482 | 215 |
| | certified o4 | 0.188 / 0.742 / 1.439 / 15.458 / 30.0 | 159 | 0.835 | 0.495 | 0.506 | 205 |
| held_out (5 / 137) | wf_smoke_v2 o4 | 0.063 / 0.294 / 1.386 / 30.0 / 30.0 | 51 | 0.657 | 0.555 | 0.744 | 23 |
| | certified o3 | 0.131 / 0.599 / 1.386 / 3.997 / 30.0 | 32 | 0.832 | 0.467 | 0.483 | 59 |
| | certified o4 | 0.117 / 0.599 / 1.567 / 30.0 / 30.0 | 52 | 0.825 | 0.518 | 0.540 | 52 |
| adversarial (4 / 76) | wf_smoke_v2 o4 | 0.063 / 0.210 / 0.875 / 15.802 / 30.0 | 20 | 0.632 | 0.513 | 0.688 | 15 |
| | certified o3 | 0.188 / 0.599 / 1.439 / 2.653 / 15.0 | 9 | 0.855 | 0.553 | 0.554 | 29 |
| | certified o4 | 0.125 / 0.587 / 1.439 / 15.458 / 30.0 | 20 | 0.816 | 0.540 | 0.548 | 28 |
| held_out+adversarial (9 / 213) | wf_smoke_v2 o4 | 0.063 / 0.294 / 1.099 / 15.916 / 30.0 | 71 | 0.648 | 0.540 | 0.725 | 38 |
| | certified o3 | 0.154 / 0.599 / 1.439 / 2.983 / 30.0 | 41 | 0.840 | 0.498 | 0.508 | 88 |
| | certified o4 | 0.117 / 0.599 / 1.442 / 15.458 / 30.0 | 72 | 0.822 | 0.526 | 0.543 | 80 |

Margin histograms (pooled, 710 branch points; bins `[0,0.5) [0.5,1) [1,2)
[2,5) [5,15) [15,∞)`): wf_smoke_v2 o4 `235 / 101 / 130 / 14 / 0 / 230`;
certified o3 `116 / 124 / 256 / 73 / 6 / 135`; certified o4
`120 / 103 / 171 / 85 / 0 / 231`. A margin of 30.0 is the scorer's floor
(`_MIN_LOG_PROB`) — the runner-up was never seen in the domain — and 15.0 is
the same floor length-normalized over a two-token path; those are saturation
artifacts, not calibrated confidence.

### Contamination check

Canonical-program overlap between each corpus and the three suites:
`wf_smoke_v2` — 0 / 0 / 0 (smoke / held_out / adversarial); the certified
train bucket — 2 / 1 / 0 (`smoke_tabs_01`, `smoke_tabs2_01`,
`held_out_tabs_01`). P7 decontaminated the bucket against the corpus's own
validation/test root-family buckets, not against the e938 smoke24 suites.
Excluding those three records (30 records, 650 branch points) moves nothing
material: held_out+adversarial confident precision wf_smoke_v2 0.724 vs
certified o3 0.525 vs o4 0.535; commit rate 0.658 vs 0.839 vs 0.824. The
excluded-variant numbers are in the JSON mirror.

## Reading

- **N1 is not supported.** The smoke-built table has no smoke-specific
  advantage: its confident precision on `smoke` (0.677) is *lower* than on
  `held_out` (0.744) and `adversarial` (0.688), and it shares no program with
  any suite. "Fixture-built" was a provenance defect, not a measurable
  overfit at n = 33.
- **The certified table commits more and is right less often at the 0.5
  threshold** (pooled: commit 0.837 vs 0.669; confident precision 0.490 vs
  0.691). The larger corpus sharpens second-order contexts (fewer floor
  margins, more mass in `[1,2)`), so more branch points clear 0.5 — but the
  branch points it clears are dominated by which-sibling-next decisions
  (`b1` vs `TextContent(` after `root = Stack([`) that a corpus prior cannot
  resolve for an unseen program. The 0.5 threshold is a test convenience,
  not a calibrated operating point for either table.
- **Why this cannot change the lever's status.** I3's verify-before-commit
  gate checks *legality* against the grammar oracle, not agreement with the
  reference program. Every "wrong commit" above is a legal path that
  diverges from the gold program, so with the lever on at margin 0.5 those
  decisions would be committed. Confident precision of 0.49–0.72 is far below
  anything that would justify turning the lever on; it stays `off`, and any
  activation needs the preregistered `ExperimentCampaignV1` binding the
  table's `corpus_fingerprint` that `decode-invariants.md` already requires.
- **Order 3 is kept.** Order 4's held_out+adversarial hit rate is +6/213
  over order 3 with a 2.5× larger context table and 71% more floor-saturated
  margins; not a reason to change the documented order.

## Caveats

- Fixture scale: 33 eval records, 710 branch points; no confidence intervals
  are claimed and none of the differences between the two certified orders is
  distinguishable from noise. The smoke-vs-certified precision gap (0.69 vs
  0.49 pooled, 0.72 vs 0.51 on held_out+adversarial) is large relative to n
  but is a property of the 0.5 threshold interaction, not evidence for any
  serving configuration.
- `hit` is agreement with one reference program; legal alternatives that
  render equivalently count as misses.
- The three overlapping records were left in the headline numbers and
  reported excluded alongside; they do not change any reading.

## Repro

```
PYTHONPATH=$PWD/src .venv/bin/python -m scripts.build_speculative_ngram_table          # rebuild
PYTHONPATH=$PWD/src .venv/bin/python -m scripts.build_speculative_ngram_table --check  # provenance + rebuild equality
PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q tests/test_dsl/test_speculative_rank.py
PYTHONPATH=$PWD/src .venv/bin/python -m scripts.verify_decode_invariants
```

The branch-point sweep is a scratch script (`SpeculativeRankerV1.choose` over
`build_completion_forest` at every position of each suite record, three
tables, `margin=0.5`, `remaining_tokens=32`); the per-branch-point rows are
summarized in the JSON mirror. Version stamp: `decode.invariants` v10,
`model.twotower` v320.
