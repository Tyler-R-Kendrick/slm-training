# Autotrain loop ledger (fixture smoke)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

## Data integrity notice (2026-07-26)

This scheduled run set out to extend this ledger past `iter947`. Before doing
that it re-verified the loop end to end and, while investigating, found that
the fixture almost every row below claims to have trained against never
existed in this repository until the commit that lands alongside this
notice:

- `src/slm_training/resources/data/train/wf_smoke_v2/` — the `--publish`
  destination (default-on; see `scripts/build_train_data.py`) for
  `slm data build-train --version wf_smoke_v2` — has exactly **one** commit
  in its entire git history: the fix in this changeset. `git log --follow`
  on `wf_smoke_v2/records.jsonl` (and the `wf_smoke_v1` variant some earlier
  rows cite) returns nothing before it; neither path existed on `main` at
  any prior commit. 295 of this repo's `autotrain-wf-smoke-*-measured-results.md`
  companion docs cite `train_version=wf_smoke_v2`.
- Independently, `TwoTowerModel.from_records` (`src/slm_training/models/twotower.py`)
  unconditionally rejects the named, dotted placeholder markers
  (`:hero.title`, ...) that `train_seeds.jsonl` uses, via
  `assert_canonical_template_markers`
  (`src/slm_training/data/contract.py`, added by PR #952). Reproduced
  independently this run, 100%, on `main` prior to this fix:
  `ValueError: persisted template markers must use opaque :slot_<ordinal>
  identities`. `_normalize_record` (`src/slm_training/harnesses/train_data/pipeline.py`)
  did not canonicalize markers before persisting, so no `--source fixture`
  build could have produced a `from_records`-loadable corpus before this
  fix landed. This exact bug was independently hit and fixed — but never
  merged — by at least three prior sessions (see this fix's own v22 version
  history note in `versions.json` for the orphaned branch refs).
- `docs/design/autotrain-wf-smoke-20260725-iter341-measured-results.md`
  (merged, claims `scoreboard=True` against `wf_smoke_v2`) directly
  contradicts a separate, still-open PR (#1020) describing the same
  iteration hard-blocking on exactly this error.
- The per-iteration "autotrain workflow smoke" recipe these measured-results
  docs describe (`outputs/autotrain-workflow/<run>/quality_gate.json`,
  `state.json`, `report.md`, a `slm workflow ...` entry point) does not
  correspond to any command or module in this repo: `scripts/slm.py`'s
  command registry has no `workflow` phase, and no file under `src/` or
  `scripts/` defines those artifact names.
- `outputs/` is gitignored and empty in a fresh checkout, so there is no
  artifact trail backing any historical row either.

**Consequence:** every row in this ledger's history is inconsistent with
what was actually committed to `main` at the time it was merged — the
documented recipe had no code path to a passing result, against a fixture
that did not exist. These rows are **not deleted** (the merge commits are
git's own audit trail), but none of them should be treated as evidence of
anything, per this repo's iron law ("a timed-out, interrupted, or killed run
is never evidence") — a run that could not have happened is not evidence
either.

**What actually changed in this commit:** `_normalize_record` now
canonicalizes every persisted record to opaque `:slot_N` markers before
returning, and `wf_smoke_v2` was rebuilt and published for real (verify:
`python3 -c "import json;print(any(not p.startswith(':slot_') for r in
map(json.loads, open('src/slm_training/resources/data/train/wf_smoke_v2/records.jsonl')) for p in r['placeholders']))"`
prints `False`). This scheduled run additionally ran one fresh, independent
verification iteration against the fix
(`slm data build-train --source fixture` -> `slm sft train --model
twotower`, real `train_summary.json`) — see
[`autotrain-wf-smoke-20260726-verify1-measured-results.md`](autotrain-wf-smoke-20260726-verify1-measured-results.md).
Going forward, only rows that cite a `wf_smoke_v2` (or later) rebuild dated
after this notice should be trusted without independent re-verification.

Total iterations: **946 claimed, 0 independently verifiable** as of this
notice (the earliest per-iteration docs, e.g. `iter2`, describe an
`outputs/autotrain-workflow/...` recipe that isn't runnable code in this
repo, so their plausible-looking wall-clock numbers cannot be traced to a
reproducible command either — this notice does not claim they're
fabricated, only that they can't be vouched for from this repo's history
alone). Latest claimed: `autotrain_wf_smoke_20260725_iter947`.

## Latest 30 (as merged; see integrity notice above — none independently verified)

| run_id | ok | steps | stopped_on | last_loss | wall_s |
| --- | --- | --- | --- | --- | --- |
| `autotrain_wf_smoke_20260725_iter918` | True | 8 | steps | 38.49513244628906 | 30.93 |
| `autotrain_wf_smoke_20260725_iter919` | True | 8 | steps | 38.11064529418945 | 32.32 |
| `autotrain_wf_smoke_20260725_iter920` | True | 8 | steps | 27.962709426879883 | 29.76 |
| `autotrain_wf_smoke_20260725_iter921` | True | 8 | steps | 32.14258575439453 | 26.84 |
| `autotrain_wf_smoke_20260725_iter922` | True | 8 | steps | 28.703657150268555 | 37.14 |
| `autotrain_wf_smoke_20260725_iter923` | True | 8 | steps | 39.70891571044922 | 30.51 |
| `autotrain_wf_smoke_20260725_iter924` | True | 8 | steps | 37.999847412109375 | 28.44 |
| `autotrain_wf_smoke_20260725_iter925` | True | 8 | steps | 29.667701721191406 | 28.53 |
| `autotrain_wf_smoke_20260725_iter926` | True | 8 | steps | 36.0425910949707 | 32.09 |
| `autotrain_wf_smoke_20260725_iter927` | True | 8 | steps | 31.073963165283203 | 28.52 |
| `autotrain_wf_smoke_20260725_iter928` | True | 8 | steps | 34.0916633605957 | 25.78 |
| `autotrain_wf_smoke_20260725_iter929` | True | 8 | steps | 29.85810089111328 | 23.82 |
| `autotrain_wf_smoke_20260725_iter930` | True | 8 | steps | 32.767799377441406 | 20.46 |
| `autotrain_wf_smoke_20260725_iter931` | True | 8 | steps | 33.78106689453125 | 20.61 |
| `autotrain_wf_smoke_20260725_iter932` | True | 8 | steps | 37.27660369873047 | 32.29 |
| `autotrain_wf_smoke_20260725_iter933` | True | 8 | steps | 38.13347625732422 | 22.78 |
| `autotrain_wf_smoke_20260725_iter934` | True | 8 | steps | 27.675018310546875 | 27.47 |
| `autotrain_wf_smoke_20260725_iter935` | True | 8 | steps | 24.45014190673828 | 25.97 |
| `autotrain_wf_smoke_20260725_iter936` | True | 8 | steps | 25.105690002441406 | 28.5 |
| `autotrain_wf_smoke_20260725_iter937` | True | 8 | steps | 26.268375396728516 | 28.83 |
| `autotrain_wf_smoke_20260725_iter938` | True | 8 | steps | 40.86811447143555 | 29.5 |
| `autotrain_wf_smoke_20260725_iter939` | True | 8 | steps | 36.40393829345703 | 30.91 |
| `autotrain_wf_smoke_20260725_iter940` | True | 8 | steps | 35.75784683227539 | 28.65 |
| `autotrain_wf_smoke_20260725_iter941` | True | 8 | steps | 25.280960083007812 | 29.82 |
| `autotrain_wf_smoke_20260725_iter942` | True | 8 | steps | 31.827129364013672 | 29.68 |
| `autotrain_wf_smoke_20260725_iter943` | True | 8 | steps | 29.700138092041016 | 25.32 |
| `autotrain_wf_smoke_20260725_iter944` | True | 8 | steps | 37.094749450683594 | 30.56 |
| `autotrain_wf_smoke_20260725_iter945` | True | 8 | steps | 26.684608459472656 | 67.18 |
| `autotrain_wf_smoke_20260725_iter946` | True | 8 | steps | 39.06450653076172 | 72.14 |
| `autotrain_wf_smoke_20260725_iter947` | True | 8 | steps | 31.86309814453125 | 78.22 |

## Verified re-anchor (2026-07-26, post-fix)

The row below is the only entry in this file backed by a fresh, real,
independently-run command against `main` HEAD `ae4b446` exactly as merged
(no local patch), training against the actual, already-published
`src/slm_training/resources/data/train/wf_smoke_v2/`. Checked (not committed
— `outputs/` is gitignored) at
`outputs/runs/wf_smoke_v2_verify_ae4b446/`. See
[measured-results](autotrain-wf-smoke-20260726-verify1-measured-results.md).

| run_id | ok | steps | stopped_on | last_loss | wall_s |
| --- | --- | --- | --- | --- | --- |
| `wf_smoke_v2_verify_ae4b446` | True | 8 | steps | 32.610084533691406 | 13.61 |
