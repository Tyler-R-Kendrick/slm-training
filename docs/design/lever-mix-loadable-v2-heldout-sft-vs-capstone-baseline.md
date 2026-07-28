# Larger-corpus / short-SFT lever vs the capstone `held_out` baseline — NOT SHIP

**Honesty:** `fixture_or_scratch`, `held_out` suite, **n=5, 1 rep** (shrunk from
the capstone's n=5/2 reps — see "What was shrunk"). **Not ship.**

## Task

This iteration's own assignment, taken verbatim from
[PR #1196's capstone "Named next lever"](lever-hard-decode-timeout-wall-heldout-capstone.md#named-next-lever-for-the-next-iteration):
build a larger training corpus than the 107-record `exposure12` fixture, run a
short SFT (steps ≤ 32), and re-eval with the **exact same** `held_out` /
`seed=47` / `--decode-timeout-seconds 30` / `fixed_asap` recipe as a direct A/B
against the capstone's `0/5` meaningful, `runtime_timeout`-saturated baseline.
Tests whether a **better-trained** checkpoint completes decode inside the
existing 30s wall, since the prior 6-PR decode-plumbing stack provably could
not move that number on its own.

## Corpus: `lever_mix_loadable_v2` (231 records, 2.16x `exposure12`)

Reused the exact "loadable mix" approach already proven in this stack
(`lever-mix-loadable-under-hard-wall-measured-results.md`,
`lever-programspec-doc-loadable-filter-measured-results.md`) — not a new
synthesizer or gate, a canonical-builder invocation plus the same offline
loadable filter `TwoTower.from_records` / `load_train_records` already apply
at load time:

```bash
# 1. Canonical builder — programspec source, document target kind (existing
#    documented workaround; NODE_OPTIONS="" needed this session — see below)
NODE_OPTIONS="" python -m scripts.build_train_data \
  --source programspec --target-kinds document \
  --programspec-count 12 --programspec-seed 47 \
  --version lever_programspec_doc_v2 --output-root outputs/data/train --no-publish
# -> 128 admitted records (quality_report.json: admitted=128, rejected_total=240)

# 2. Merge lever_exposure12_v1 (107, all already loadable) with the subset of
#    lever_programspec_doc_v2 that clears the same four load-time gates
#    TwoTower.from_records enforces (assert_no_template_semantic_labels,
#    assert_canonical_template_markers, assert_symbol_only_output,
#    assert_role_safe_output) plus load_train_records' harness_dsl canonical-
#    serialization check. 126/128 programspec-doc records pass (2 rejected:
#    1 "non-canonical Harness serialization", 1 "placeholder ':slot_1' in
#    non-content property ImageBlock.src"); 2 duplicate ids dropped against
#    exposure12 -> lever_mix_loadable_v2 = 107 + 124 = 231 records.
```

`load_train_records(lever_mix_loadable_v2)` loads all 231 cleanly (verified
before training).

**Environment note:** `build_train_data --source programspec` failed on first
attempt — the shell's ambient `NODE_OPTIONS` (`"--import tsx" --max-old-space-size=8192`,
malformed quoting) crashed the OpenUI Node bridge (`--import tsx is not
allowed in NODE_OPTIONS`, exit 9). Unsetting `NODE_OPTIONS` for the build
subprocess fixed it; not a repo/harness bug, an ambient-shell artifact of this
session's sandbox.

## SFT recipe (unchanged conventions, steps=32 ≤ cap)

Same champion micro-recipe this whole stack has used, only the corpus and
step count changed — **no capacity/size lever**: default `twotower` dims,
identical architecture to the `exposure12` baseline checkpoint (goal invariant
VI — this is a pure data lever, not a scaling arm).

```bash
python -m scripts.train_model \
  --train-dir outputs/data/train/lever_mix_loadable_v2 \
  --model twotower --context-backend scratch \
  --steps 32 --batch-size 2 --lr 0.001 --structural-bias 1.5 --seed 47 \
  --device cpu --asap-decode \
  --run-id exp_lever_mix_loadable_v2_s32_lr1e3_bs2_sb15_seed47 \
  --no-sync-checkpoints
```

`train_summary.json`: `steps=32`, `stopped_on="steps"` (did not hit the wall
budget), `last_loss=11.379` (vs `exposure12` baseline `last_loss=7.189` at
s16 — higher loss, consistent with more/harder mixed data at the same short
step budget, not a regression signal by itself). Scratch checkpoint, deliberate
`--no-sync-checkpoints` (fixture/scratch run, no bucket sync, no
`MODEL_CARD.md`/README update per the model-card trigger table — this is not a
promoted checkpoint).

## Re-eval: exact capstone recipe, 1 rep instead of 2

```bash
python -m scripts.evaluate_model \
  --test-dir src/slm_training/resources/data/eval/e938_role_safe_all_targets_v2 \
  --suite held_out --model twotower --device cpu \
  --checkpoint outputs/runs/exp_lever_mix_loadable_v2_s32_lr1e3_bs2_sb15_seed47/checkpoints/last.pt \
  --grammar-constrained --decode-timeout-seconds 30 --seed 47 \
  --constraint-debt-routing-mode fixed_asap --run-class scratch_matrix \
  --run-id mix_v2_s32_heldout_rep1_seed47
```

**What was shrunk, and why:** the capstone already used most of a
`MAX_RUN_MINUTES=3` (170s-interrupt) budget on 2 reps at `held_out` n=5 (5
records × up to 30s hard wall = up to 150s per rep). This iteration's
invocation additionally had to fit a corpus build and an SFT run inside their
own bounded calls, so the eval step was shrunk to **1 rep** — the smallest
sample that still lets us compare directly to the capstone's per-rep numbers
(each of the capstone's 2 reps was itself `0/5` meaningful, so a single rep is
directly comparable to either).

## Measured: `0/5` meaningful (unchanged), decode outcomes shift slightly

| metric | capstone baseline (`exp12`, per rep, n=5) | this run (`mix_loadable_v2`, n=5) |
| --- | ---: | ---: |
| `meaningful_program_rate` | 0.0 | **0.0** |
| `parse_rate` | 0.0 | 0.2 |
| `structural_similarity` | 0.0 | 0.0 |
| `component_type_recall` | 0.0 | 0.0 |
| `placeholder_fidelity` | — (n/a, all timed out) | 0.2 |
| `reward_score` | — | 0.1874 |
| `decode_outcome_counts` | `{runtime_timeout: 5}` | `{runtime_timeout: 4, model_invalid: 1}` |
| `failure_breakdown` | `{parse_error: 5}` | `{parse_error: 4, low_component_recall: 1}` |

The primary ship-relevant number, `meaningful_program_rate`, is **unchanged at
0.0** — the larger corpus + short SFT did not produce a single meaningful
`held_out` completion. One of the 5 records did finish decoding inside the 30s
wall this time (`model_invalid` instead of `runtime_timeout`), lifting
`parse_rate` from 0.0 to 0.2 and `placeholder_fidelity` from undefined-by-
timeout to 0.2 — a real but non-ship-relevant shift (the record that got
through was invalid, not meaningful). `evidence_class` remains
`fixture_under_minimum_n` (n=5 < ship minimum n=20).

## Ship-gate check (`honest-ship-eval` default `held_out` bars)

| criterion | bar | actual | pass? |
| --- | ---: | ---: | --- |
| `insufficient_n` | ≥ 20 | 5 | fail |
| `meaningful_program_rate` | ≥ 0.40 | 0.0 | fail |
| `structural_similarity` | ≥ 0.30 | 0.0 | fail |
| `component_type_recall` | ≥ 0.30 | 0.0 | fail |
| `placeholder_fidelity` | ≥ 0.15 | 0.2 | **pass** |
| `decode_timeout_count` | = 0 | 4 | fail |

Fails 5 of 6 gates. **Verdict: `fixture_or_scratch`, not ship** — same as
every doc in this thread. The single passing gate (`placeholder_fidelity`) is
driven by one record that parsed-but-was-invalid, not a meaningful program; it
does not change the readiness verdict.

## Interpretation

**Consistent with, not contradicting, the prior smoke-scale finding.** The
`lever-mix-loadable-under-hard-wall-measured-results.md` smoke run (n=3,
s16) already found the loadable mix corpus **iso** on `meaningful_program_rate`
(0.333→0.333) with a `parse` improvement (0.889→1.000); `lever-mix-loadable-
s32-under-hard-wall` (s32 on the same corpus) found **no further lift**
(0.333→0.333, `wall_time_budget`-truncated at 26 steps). This run repeats that
exact pattern at `held_out` scale against the capstone's decode-fixed
checkpoint: `meaningful_program_rate` stays flat (0.0→0.0) while `parse_rate`
ticks up (0.0→0.2) because one more record finishes inside the wall. **A
2.16x larger, contract-valid, loadable corpus plus a 2x step count (16→32)
moves decode-completion odds for individual records, but not the odds of
producing a semantically meaningful program.** The bottleneck is not corpus
size at this scale (107→231 records) or step count (16→32) either — the
compiler-tree witness search's real full-record cost still exceeds 30s wall-
clock for the overwhelming majority of `held_out` records even after training
on more data.

## Decision

**REJECT** — larger-corpus + short-SFT does not lift `meaningful_program_rate`
vs the capstone `held_out` baseline (0.0→0.0); the training-data lever, like
the decode-plumbing lever before it, cannot fix a checkpoint whose completions
need more compute than the fixed 30s wall allows, at this corpus size (231
records) and step budget (32 steps).

## Named next lever (for the next iteration)

Both independent levers probed by this stack — decode-side speed (PRs
#1189-#1196) and training-data size/short-SFT (this doc) — leave
`meaningful_program_rate` at 0.0 on `held_out`. What is left, evidenced by
elimination:

1. **A genuinely longer SFT** (hundreds of steps, not ≤32) on the loadable-mix
   corpus or a still-larger one — this run's `steps=32` stopped on `steps`,
   not `wall_time_budget`, so there is bounded headroom to extend step count
   within a single invocation before hitting the run cap; a multi-invocation
   / checkpoint-resume campaign could go further.
2. **Curriculum/mixture changes** — order or reweight the mix so easier
   completions are seen first (a lever named but not yet run in
   `lever-train-signal-under-hard-wall-measured-results.md`).
3. **Revisit the 30s wall itself** as the bottleneck, not the checkpoint —
   this grammar's compiler-tree witness search may need either an
   algorithmic change (not a cache-level micro-optimization, already probed
   and rejected in PRs #1189-#1195) or a documented, honestly-labeled longer
   timeout for this eval protocol specifically, re-validated against
   `honest-ship-eval` bars rather than assumed.

## Validation

```text
python -m scripts.repo_policy
python -m scripts.verify_version_stamps --check
python -m scripts.verify_decode_invariants
```

No harness/metric/gate/matrix file changed this session (pure corpus
build + SFT + eval using existing canonical scripts) — 0 version-stamp
component bumps required.

## Scope note

- Diagnostic lever measurement only. No `--ship-gates` scoreboard claim, no
  checkpoint promotion, no `MODEL_CARD.md` update — checkpoint is
  scratch/32-step, `--no-sync-checkpoints`.
- `outputs/data/train/lever_programspec_doc_v2/`,
  `outputs/data/train/lever_mix_loadable_v2/`,
  `outputs/runs/exp_lever_mix_loadable_v2_s32_lr1e3_bs2_sb15_seed47/`, and
  `outputs/runs/mix_v2_s32_heldout_rep1_seed47/` are gitignored, not
  committed — this doc is the durable record.

Captured: 2026-07-28T13:31:00Z
