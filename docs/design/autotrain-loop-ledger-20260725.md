# Autotrain loop ledger (fixture smoke)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

Total successful iterations: **340** (latest
`autotrain_wf_smoke_20260725_iter340`).

## BLOCKED as of iter341 (2026-07-26)

The loop cannot produce further genuine iterations right now:
`data build-train --source fixture` → `sft train --model twotower` fails on
**100% of records** (all 103) because `train_seeds.jsonl` uses named
placeholder markers (`:auth.title`, …) that PR #952 (`7cddae64`, 2026-07-25)
now rejects via `assert_canonical_template_markers`. See
[`autotrain-wf-smoke-20260726-iter341-measured-results.md`](autotrain-wf-smoke-20260726-iter341-measured-results.md)
for full repro, root cause, and evidence. No new iteration is fabricated
past this point until the seed corpus is canonicalized (harness fix, out of
this loop's scope — see `improve-openui-harnesses`).

## Latest 30 (through last successful iteration, iter340)

| run_id | ok | steps | stopped_on | last_loss | wall_s |
| --- | --- | --- | --- | --- | --- |
| `autotrain_wf_smoke_20260725_iter311` | True | 8 | steps | 23.7349910736084 | 75.83 |
| `autotrain_wf_smoke_20260725_iter312` | True | 8 | steps | 32.074493408203125 | 72.29 |
| `autotrain_wf_smoke_20260725_iter313` | True | 8 | steps | 26.81902503967285 | 82.52 |
| `autotrain_wf_smoke_20260725_iter314` | True | 8 | steps | 33.04792785644531 | 68.71 |
| `autotrain_wf_smoke_20260725_iter315` | True | 8 | steps | 31.39019012451172 | 71.74 |
| `autotrain_wf_smoke_20260725_iter316` | True | 8 | steps | 35.060611724853516 | 75.64 |
| `autotrain_wf_smoke_20260725_iter317` | True | 8 | steps | 41.70116424560547 | 72.66 |
| `autotrain_wf_smoke_20260725_iter318` | True | 8 | steps | 37.04560089111328 | 71.67 |
| `autotrain_wf_smoke_20260725_iter319` | True | 8 | steps | 36.72319412231445 | 76.87 |
| `autotrain_wf_smoke_20260725_iter320` | True | 8 | steps | 30.42670440673828 | 71.25 |
| `autotrain_wf_smoke_20260725_iter321` | True | 8 | steps | 29.345230102539062 | 69.6 |
| `autotrain_wf_smoke_20260725_iter322` | True | 8 | steps | 23.372859954833984 | 71.96 |
| `autotrain_wf_smoke_20260725_iter323` | True | 8 | steps | 35.29148483276367 | 73.08 |
| `autotrain_wf_smoke_20260725_iter324` | True | 8 | steps | 21.302478790283203 | 75.06 |
| `autotrain_wf_smoke_20260725_iter325` | True | 8 | steps | 28.323143005371094 | 2.62 |
| `autotrain_wf_smoke_20260725_iter326` | True | 8 | steps | 40.26213073730469 | 10.04 |
| `autotrain_wf_smoke_20260725_iter327` | True | 8 | steps | 28.557283401489258 | 2.44 |
| `autotrain_wf_smoke_20260725_iter328` | True | 8 | steps | 29.846359252929688 | 2.21 |
| `autotrain_wf_smoke_20260725_iter329` | True | 8 | steps | 47.39080047607422 | 2.12 |
| `autotrain_wf_smoke_20260725_iter330` | True | 8 | steps | 25.401201248168945 | 2.23 |
| `autotrain_wf_smoke_20260725_iter331` | True | 8 | steps | 26.580276489257812 | 21.76 |
| `autotrain_wf_smoke_20260725_iter332` | True | 8 | steps | 20.949220657348633 | 42.61 |
| `autotrain_wf_smoke_20260725_iter333` | True | 8 | steps | 38.3284912109375 | 79.59 |
| `autotrain_wf_smoke_20260725_iter334` | True | 8 | steps | 34.85049819946289 | 89.77 |
| `autotrain_wf_smoke_20260725_iter335` | True | 8 | steps | 25.98305892944336 | 99.52 |
| `autotrain_wf_smoke_20260725_iter336` | True | 8 | steps | 38.58271789550781 | 85.82 |
| `autotrain_wf_smoke_20260725_iter337` | True | 8 | steps | 32.017005920410156 | 57.54 |
| `autotrain_wf_smoke_20260725_iter338` | True | 8 | steps | 27.528940200805664 | 37.14 |
| `autotrain_wf_smoke_20260725_iter339` | True | 8 | steps | 31.777908325195312 | 21.61 |
| `autotrain_wf_smoke_20260725_iter340` | True | 8 | steps | 36.01945877075195 | 2.55 |
