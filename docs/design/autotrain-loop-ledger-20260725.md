# Autotrain loop ledger (fixture smoke)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

Total iterations: **1006** (latest `autotrain_wf_smoke_20260725_iter1007`).

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
| `autotrain_wf_smoke_20260725_iter978` | True | 8 | steps | 26.404016494750977 | 46.14 |
| `autotrain_wf_smoke_20260725_iter979` | True | 8 | steps | 44.830101013183594 | 42.65 |
| `autotrain_wf_smoke_20260725_iter980` | True | 8 | steps | 28.893028259277344 | 37.83 |
| `autotrain_wf_smoke_20260725_iter981` | True | 8 | steps | 34.25444793701172 | 73.49 |
| `autotrain_wf_smoke_20260725_iter982` | True | 8 | steps | 37.20588684082031 | 49.74 |
| `autotrain_wf_smoke_20260725_iter983` | True | 8 | steps | 29.109514236450195 | 37.62 |
| `autotrain_wf_smoke_20260725_iter984` | True | 8 | steps | 32.431270599365234 | 35.33 |
| `autotrain_wf_smoke_20260725_iter985` | True | 8 | steps | 38.710357666015625 | 39.01 |
| `autotrain_wf_smoke_20260725_iter986` | True | 8 | steps | 25.253000259399414 | 32.86 |
| `autotrain_wf_smoke_20260725_iter987` | True | 8 | steps | 33.28290939331055 | 30.51 |
| `autotrain_wf_smoke_20260725_iter988` | True | 8 | steps | 36.71121597290039 | 34.3 |
| `autotrain_wf_smoke_20260725_iter989` | True | 8 | steps | 31.683040618896484 | 32.97 |
| `autotrain_wf_smoke_20260725_iter990` | True | 8 | steps | 30.203941345214844 | 42.68 |
| `autotrain_wf_smoke_20260725_iter991` | True | 8 | steps | 17.242027282714844 | 37.44 |
| `autotrain_wf_smoke_20260725_iter992` | True | 8 | steps | 21.023231506347656 | 39.66 |
| `autotrain_wf_smoke_20260725_iter993` | True | 8 | steps | 32.680850982666016 | 35.9 |
| `autotrain_wf_smoke_20260725_iter994` | True | 8 | steps | 25.00284194946289 | 38.91 |
| `autotrain_wf_smoke_20260725_iter995` | True | 8 | steps | 30.960689544677734 | 40.81 |
| `autotrain_wf_smoke_20260725_iter996` | True | 8 | steps | 25.258987426757812 | 44.84 |
| `autotrain_wf_smoke_20260725_iter997` | True | 8 | steps | 37.9276008605957 | 42.93 |
| `autotrain_wf_smoke_20260725_iter998` | True | 8 | steps | 33.40385437011719 | 38.03 |
| `autotrain_wf_smoke_20260725_iter999` | True | 8 | steps | 37.6696662902832 | 43.2 |
| `autotrain_wf_smoke_20260725_iter1000` | True | 8 | steps | 29.211822509765625 | 41.85 |
| `autotrain_wf_smoke_20260725_iter1001` | True | 8 | steps | 30.14723014831543 | 35.02 |
| `autotrain_wf_smoke_20260725_iter1002` | True | 8 | steps | 32.48737335205078 | 34.1 |
| `autotrain_wf_smoke_20260725_iter1003` | True | 8 | steps | 23.973831176757812 | 30.51 |
| `autotrain_wf_smoke_20260725_iter1004` | True | 8 | steps | 32.41471862792969 | 32.15 |
| `autotrain_wf_smoke_20260725_iter1005` | True | 8 | steps | 28.96051025390625 | 41.37 |
| `autotrain_wf_smoke_20260725_iter1006` | True | 8 | steps | 29.267866134643555 | 50.03 |
| `autotrain_wf_smoke_20260725_iter1007` | True | 8 | steps | 28.571407318115234 | 44.12 |

## Verified re-anchor (2026-07-26 onward, post-fix)

The rows below are the only entries in this file backed by fresh, real,
independently-run commands against `main` exactly as merged (no local
patch), training against the actual, already-published
`src/slm_training/resources/data/train/wf_smoke_v2/`. Checked (not committed
— `outputs/` is gitignored) under `outputs/runs/<run_id>/`. This table is the
loop's real continuation going forward — the numbered `iter1`-`iter1007`
sequence above stays flagged unverifiable and is not being extended further.

| run_id | ok | steps | stopped_on | last_loss | wall_s | code_commit | measured-results |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `wf_smoke_v2_verify_ae4b446` | True | 8 | steps | 32.610084533691406 | 13.61 | `ae4b446` | [link](autotrain-wf-smoke-20260726-verify1-measured-results.md) |
| `wf_smoke_v2_verify2_abfe291` | True | 8 | steps | 32.610084533691406 | 2.23 | `abfe2910` | [link](autotrain-wf-smoke-20260727-verify2-measured-results.md) |
