# CPU thread count is a real (small) source of run-to-run non-determinism

**Honesty:** `fixture_or_scratch`. **Not a ship claim — a reproducibility
root-cause investigation. `is_fix: false`, no code changed.**

## Why this run

[`lever-seed-rescue-steps72-measured-results.md`](lever-seed-rescue-steps72-measured-results.md)'s
"Next steps" section (item 2, still open) names diagnosing the `s36_seed42`
cross-session discrepancy — an ad hoc reproduction of PR #1135's
`s36_seed42` arm that gave `meaningful_program_rate=0.0` instead of the
original `0.333` for a nominally identical recipe — as unresolved, after a
same-day correction ruled out eval-fixture non-determinism as the cause.
Three candidates were left open: CPU thread-count differences across
sessions, `torch` version drift, or an uncaptured mistake in the original ad
hoc repro. This scheduled iteration tests the first candidate directly,
since it's the one testable without reverse-engineering an undocumented
recipe (`s36_seed42`'s exact command was never captured as a reproducible
script — only a raw log block survives in
[`lever-seed-sweep-asap-t30-measured-results.md`](lever-seed-sweep-asap-t30-measured-results.md)).

Two open PRs today (#1179, #1180, different scheduled session) are already
pursuing "Next steps" item 1 (the seed=44 eval hang) on a sibling branch —
this iteration deliberately picks the other open thread to avoid duplicate
work.

## Recipe

The repo's only fully reproducible, committed recipe is the `wf_smoke_v2`
smoke-loop line from `autotrain-loop-ledger-20260725.md` (batches #2-#4),
not the undocumented exposure12/ASAP recipe the original discrepancy used.
This run reuses it unchanged, varying only thread-count env vars:

```bash
OMP_NUM_THREADS=<N> MKL_NUM_THREADS=<N> python -m scripts.train_model \
  --train-dir src/slm_training/resources/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps 8 \
  --run-id <run_id> --no-sync-checkpoints --device cpu --seed 0
```

`N` in `{1, 2, 3, 4}` (host has 4 cores), 2 reps at `N=1,2,4` and 1 rep at
`N=3`. Environment: fresh `.venv-autotrain` (Python 3.12.3,
`torch==2.5.1+cu124`, `pip install -e ".[dev,torch]"`), created and torn
down in this session. Checked (not committed — `outputs/` is gitignored) at
`outputs/runs/autotrain_threadcheck_t<N>_<rep>/`.

## Results

| threads | rep | `last_loss` |
| ---: | --- | --- |
| 1 | a | 32.61008071899414 |
| 1 | b | 32.61008071899414 |
| 2 | a | 32.61008071899414 |
| 2 | b | 32.61008071899414 |
| 3 | a | 32.610084533691406 |
| 4 | a | 32.610084533691406 |
| 4 | b | 32.610084533691406 |

**Within a fixed thread count, every repeated run reproduces `last_loss`
bit-for-bit** (2/2 at threads=1, 2/2 at threads=2, 2/2 at threads=4) — same
result the ledger's earlier 16-row determinism batch already established at
the default thread count. **Across thread counts, `last_loss` takes exactly
two distinct values**, bucketed as `{1,2} -> 32.61008071899414` vs.
`{3,4} -> 32.610084533691406` — a float32-precision-scale difference
(absolute delta `3.81e-06`, relative delta `1.17e-07`).

## Decision

**HYPOTHESIS CONFIRMED (narrowly):** CPU thread count is a real, measured,
reproducible source of `last_loss` non-determinism in this harness, even
with an identical `--seed`, identical fixture, identical code. This is
expected behavior for CPU floating-point reduction (non-associative
summation order changes with thread partitioning) — not a bug — but it is
real enough to make a bit-for-bit cross-session loss comparison unsafe
unless thread count is pinned and recorded alongside the seed.

**This does not close the original `s36_seed42` discrepancy.** That gap was
`meaningful_program_rate` 0.333 → 0.0 (a categorical outcome flip on a
downstream constrained-decode eval), not a `~1e-7`-scale loss wobble — this
run measures loss-level float noise on a different (though related) recipe,
not the original SFT+eval pipeline, whose exact command was never captured
as a reproducible script. Thread-count is now a confirmed real phenomenon
in this harness, not a proven explanation for that specific magnitude of
discrepancy.

## Next steps (evidence-backed)

1. Record thread count (or pin it) alongside seed in every lever doc's
   recipe/environment section going forward.
2. Repeat this sweep against the actual exposure12/ASAP/`decode_timeout=30`
   SFT+eval recipe once its command is captured as a reusable script, to
   test whether thread-count variance alone can plausibly explain a
   `meaningful_program_rate` swing of that size (it would need to be a much
   larger effect than the loss-level noise measured here).
3. `torch` version drift across sessions and an uncaptured repro mistake
   remain untested candidates for the original discrepancy.

Captured: 2026-07-28T03:34:33Z
