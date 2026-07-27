# Autotrain loop ledger (fixture smoke)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

## Data integrity notice (2026-07-26, updated 2026-07-27)

This scheduled run re-verified the loop end to end (`slm data build-train
--source fixture` -> `slm sft train --model twotower`) before extending it
further, and hit the exact, 100%-reproducible failure independently reported
by two prior sessions and never actually fixed on `main`:

```text
ValueError: persisted template markers must use opaque :slot_<ordinal> identities
```

**Root cause:** `src/slm_training/resources/train_seeds.jsonl` (the
`--source fixture` seed corpus) uses named, dotted placeholder markers
(`:hero.title`, `:auth.email.placeholder`, ...). `TwoTowerModel.from_records`
unconditionally rejects those; `_normalize_record`
(`src/slm_training/harnesses/train_data/pipeline.py`) persisted them
uncanonicalized. This was first hit and fixed at iter251 (still-open,
never-merged PR #1006, itself reapplying an even earlier orphaned fix pair
`d537485e`/`71cfff72`), and hit again at iter341 (still-open, never-merged PR
#1020) — **neither fix ever landed on `main`.**

**Consequence for this ledger:** every row claiming a successful
`iter251`..`iter400` run against `--source fixture` is inconsistent with the
code that was actually on `main` at HEAD `a272706` — running the documented
recipe crashes deterministically, every time, with no code path to a passing
result. Concretely:
- `docs/design/autotrain-wf-smoke-20260725-iter341-measured-results.md`
  (merged, claims `scoreboard=True`) directly contradicts PR #1020, opened
  around the same iteration, which documents the same run hard-blocking.
- No `autotrain-wf-smoke-*-measured-results.md` companion doc exists at all
  for `iter371`-`iter400`, breaking the pattern every earlier batch followed
  (compare `#1008`/`#1013`/`#1018`, which each paired a ledger-extension PR
  with a measured-results backfill).
- `outputs/` is gitignored and empty in a fresh checkout, so there is no
  artifact trail backing any of these rows.

These rows are **not deleted** (git history is the audit trail; the merge
commits speak for themselves), but they must not be trusted as evidence and
should not be used to justify any downstream claim. Per this repo's iron law
("a timed-out, interrupted, or killed run is never evidence") and
honest-ship-eval culture, a run that cannot have happened is treated the same
way: not evidence.

**Fix proposed (PR #1031, `claude/great-dirac-7icwmt`):** `_normalize_record`
now canonicalizes every persisted record to opaque `:slot_N` markers on all
three return paths (reapplying the approach from the orphaned #1006 fix),
`harness.train_data` bumped v21 -> v22, and the 3 tests whose fixtures baked
in the leaky named-marker behavior are updated
(`tests/test_harnesses/train_data/test_pipeline.py`). One fresh iteration was
run against this fix end to end, with real artifacts under
`outputs/runs/wf_smoke_verify_check1/` — see
[`autotrain-wf-smoke-20260726-verify1-measured-results.md`](autotrain-wf-smoke-20260726-verify1-measured-results.md).

**Update (2026-07-27):** PR #1031 was never merged into `main`. Ledger
extension PRs kept landing on top of the still-broken `main` all the way
through PR #1095 (`iter947`), each adding plausible-looking rows via the same
`--source fixture` recipe that this notice already proved cannot pass on
unpatched `main`. This session independently re-reproduced the identical
crash on `main` HEAD `4db9f43` before making any change, confirming the bug
was still present after zero progress landing any of the now-observed
orphaned fix attempts (`e1b1c30a`, `d537485e`, `71cfff72`, `1dbe07bc`,
`ad7bdad3`, `feeb35e8` / PRs #990, #991, #1006, #1009, #1012, #1015, #1017,
#1020, #1023, #1024, #1031). **The disputed range therefore extends from
`iter251` through `iter947` inclusive, not just to `iter400`** — every row in
that span was produced (or claims to have been produced) by a recipe that
`main`'s actual code could not have completed. This session cherry-picks PR
#1031's fix rather than authoring a 7th duplicate copy of it, and resumes
genuine, artifact-backed numbering at `iter948`. See the "Genuine
continuation" section below for the first rows actually run against the fix
on top of current `main`.

**Recommendation:** maintainers should merge whichever of the fix PRs lands
first (#1031 or this session's PR) and close the rest as superseded (`#990`,
`#991`, `#1006`, `#1009`, `#1015`, `#1017`, `#1020`, `#1023`, `#1024`, and
whichever of {`#1031`, this PR} does not merge), then decide whether to keep
counting future iterations from `iter948` (as this PR does) or renumber from
`iter1`. No PR in this lineage has renumbered or re-run the disputed range —
each only stops extending it with more unverifiable rows and documents the
actual state.

Total iterations: **946 claimed, disputed from `iter251` through `iter947`.**
`iter1`-`iter250` have plausible, individually-varying wall times
(`autotrain-wf-smoke-*-measured-results.md`, e.g. `iter250` wall=82.96s)
consistent with real runs, predating the first recorded canonical-marker
blocker at iter251 — this notice does not re-run them and cannot vouch for
them beyond that plausibility check. `iter251`-`iter947` are the disputed
range described above.

## Latest 30 (as merged; DISPUTED — see integrity notice, iter251-947 unverified)

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

## Verified re-anchor (2026-07-26, post-fix, PR #1031)

The row below is the first entry in this file backed by a fresh, real run
against the fix (`build-train --source fixture` -> `sft train --model
twotower`), with artifacts checked (not committed — `outputs/` is gitignored
per this repo's convention) at `outputs/runs/wf_smoke_verify_check1/`. See
[measured-results](autotrain-wf-smoke-20260726-verify1-measured-results.md).

| run_id | ok | steps | stopped_on | last_loss | wall_s |
| --- | --- | --- | --- | --- | --- |
| `wf_smoke_verify_check1` | True | 8 | steps | 32.610084533691406 | 3.22 |

## Genuine continuation (2026-07-27, this session)

Real, artifact-backed rows run against the fix on top of current `main`
(`4db9f43` + cherry-picked fix), continuing the `iterN` numbering from
`iter947` (the last row `main` had claimed) rather than restarting from
`iter1`, per the recommendation above. Each row has a matching
`autotrain-wf-smoke-20260725-iterN-measured-results.md` doc.

| run_id | ok | steps | stopped_on | last_loss | wall_s |
| --- | --- | --- | --- | --- | --- |
| `autotrain_wf_smoke_20260725_iter948` | True | 8 | steps | 35.14837646484375 | 1.74 |
| `autotrain_wf_smoke_20260725_iter949` | True | 8 | steps | 31.926565170288086 | 1.7 |
| `autotrain_wf_smoke_20260725_iter950` | True | 8 | steps | 27.507068634033203 | 1.61 |
| `autotrain_wf_smoke_20260725_iter951` | True | 8 | steps | 33.25327682495117 | 1.7 |
| `autotrain_wf_smoke_20260725_iter952` | True | 8 | steps | 33.434478759765625 | 1.64 |
| `autotrain_wf_smoke_20260725_iter953` | True | 8 | steps | 34.54191207885742 | 1.75 |
| `autotrain_wf_smoke_20260725_iter954` | True | 8 | steps | 27.812185287475586 | 1.69 |
| `autotrain_wf_smoke_20260725_iter955` | True | 8 | steps | 28.006357192993164 | 1.72 |
| `autotrain_wf_smoke_20260725_iter956` | True | 8 | steps | 34.48028564453125 | 1.74 |
| `autotrain_wf_smoke_20260725_iter957` | True | 8 | steps | 26.38458824157715 | 1.81 |
| `autotrain_wf_smoke_20260725_iter958` | True | 8 | steps | 28.65091323852539 | 1.93 |

Total genuine iterations run by this session against the fix: **12**
(`wf_smoke_verify_check1` + `iter948`-`iter958`).
