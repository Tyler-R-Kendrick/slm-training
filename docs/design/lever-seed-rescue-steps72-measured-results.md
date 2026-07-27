# Seed-brittleness rescue check: steps 30/36 → 72 (ASAP + decode_timeout=30)

**Honesty:** `fixture_or_scratch` / smoke, suite `n=3`. **Not a ship claim.**

## Why this run

The last merged lever campaign
([`lever-seed-sweep-asap-t30-measured-results.md`](lever-seed-sweep-asap-t30-measured-results.md),
PR #1135) hard-failed 3 of 6 seeds (43, 45, 46: `parse_rate=0`,
`empty_prediction_count=3/3`) at `steps∈{30,36}` on the `wf_smoke_v2` fixture,
and closed with an explicit evidence-backed next step: *"Target seed
brittleness: longer train wall / larger data / curriculum — not more smoke
iters."* This run acts on that note directly: same recipe (`twotower`,
`--context-backend scratch`, `--asap-decode`, `--device cpu`,
`--decode-timeout-seconds 30`, `--constraint-debt-routing-mode fixed_asap`),
`--steps` raised from 30/36 to **72**, same six seeds (42-47).

## Recipe

```bash
python -m scripts.train_model \
  --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps 72 --asap-decode \
  --run-id lever_seedrescue_s72_seed<N> --no-sync-checkpoints --device cpu --seed <N>

python -m scripts.evaluate_model \
  --test-dir <freshly built wf_smoke_test_v1/smoke, 3 records> --suite smoke \
  --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --device cpu --run-id lever_seedrescue_s72_seed<N> \
  --grammar-constrained --decode-timeout-seconds 30 \
  --constraint-debt-routing-mode fixed_asap --run-class scratch_matrix
```

Eval fixture built once via
`slm data build-test --source fixture --no-rico-path --train-manifest
src/slm_training/resources/data/train/wf_smoke_v2/manifest.json --suites smoke`
(3 records; not committed, `outputs/`/scratch is gitignored).

Environment: fresh `.venv-autotrain` (Python 3.12, `pip install -e ".[dev,torch]"`),
plus `npm ci` for the AgentV publish step (`node_modules/`, gitignored) — this
session's checkout had neither pre-installed.

## Scoreboard

```
# Seed-rescue sweep (steps=72, ASAP + decode_timeout=30) — NOT SHIP
seed steps  parse  meaningful  reward  empty  timeout       lat_ms   prior@steps30/36
  42    72  1.000       0.333   0.783      0        0     30002.66   parse=1.000 (steps36, ok)
  43    72  1.000       0.333   0.808      0        0     30002.55   parse=0.000 (steps30, HARD FAIL)
  44    72      -           -       -      -        -            -   parse=0.667 (steps36, ok) — eval KILLED, not evidence
  45    72  1.000       0.333   0.765      0        0     18402.69   parse=0.000 (steps30, HARD FAIL)
  46    72  1.000       0.333   0.765      0        0     30001.98   parse=0.000 (steps30, HARD FAIL)
  47    72  1.000       0.333   0.783      0        0     30002.30   parse=1.000 (steps30, ok)

success_rate (parse>=0.5 & empty=0), n=5 evidenced: 5/5
rescued_hard_fail_seeds: [43, 45, 46] — all three flip from empty=3/3 to parse=1.0, empty=0
```

## Killed run — not evidence

`seed=44`'s eval was killed twice (170s, then a retry at 178s — both under
this repo's `MAX_RUN_MINUTES=3` cap) with **zero stdout captured either
time**; training for seed=44 completed normally (`last_loss=6.2478`). Per
this repo's iron law — *"A timed out, interrupted, or killed run is never
evidence"* — seed 44's eval result at steps=72 is **excluded**, not
back-filled or guessed. Root cause not diagnosed in this session (candidates:
AgentV/node publish subprocess stall, or a genuine 3×30s decode-timeout worst
case plus overhead exceeding the 170-180s wrapper); worth isolating before
trusting a wider steps=72 sweep on CPU.

## Methodology confound (read before comparing across PRs)

This session built its own eval fixture
(`slm data build-test --source fixture ... --suites smoke`, 3 records) rather
than reusing whatever fixture PR #1135 evaluated against (that fixture's
content was not persisted — `outputs/` is gitignored and PR #1135 left no
committed eval-suite artifact). An ad hoc reproduction of PR #1135's
`s36_seed42` arm under *this session's* fixture gave
`meaningful_program_rate=0.0`, not PR #1135's `0.333`, for the nominally
identical recipe — confirming the two fixture builds are not guaranteed
byte-identical. **The steps=30-vs-72 delta is still trustworthy** because it
is internally controlled (same session, same freshly-built fixture, only
`--steps`/`--seed` varied) — but do not diff raw latency/reward magnitudes
between this table and PR #1135's without rebuilding a shared, committed eval
fixture first.

## Result

Longer train wall (steps 30/36 → 72) **rescues** all three seeds that hard-
failed at the shorter step counts (43, 45, 46: `parse_rate` 0.0 → 1.0,
`empty_prediction_count` 3/3 → 0/3), while seeds that already worked (42, 47)
stay at `parse_rate=1.0`. This is directional support for the "longer train
wall" arm of the prior campaign's evidence-backed next step, on `n=5`
evidenced seeds (44 excluded, not evidence) against a 3-record smoke suite.
Still `fixture_or_scratch`: one training-wall setting, one micro-fixture, no
held-out generalization claim, and the fixture-build confound above means the
*direction* (steps helps) is better supported than the *magnitude*.

## Next steps (evidence-backed)

1. Diagnose the seed=44 eval hang before trusting steps=72 broadly — it may
   indicate a real tail-latency failure mode masked as "not evidence" here.
2. Commit a reusable, version-controlled eval fixture for this smoke line so
   future sessions stop confounding "different recipe" with "different eval
   build" (the confound noted above has now recurred across at least two
   sessions).
3. Per the original PR #1135 note once step-count brittleness is genuinely
   covered: move off `wf_smoke_v2` onto real held-out data or the DSH5-10 /
   `AP-007+` threads — this smoke-scale seed check should not keep
   regenerating indefinitely.

Captured: 2026-07-27T14:46:00+00:00
