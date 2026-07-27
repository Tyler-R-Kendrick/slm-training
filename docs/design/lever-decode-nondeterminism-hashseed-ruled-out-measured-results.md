# Decode non-determinism: `PYTHONHASHSEED` ruled out (NOT SHIP)

**Honesty:** `fixture_or_scratch` / smoke suite `n=3`. **Not a ship claim —
a reproducibility root-cause investigation.**

## Why this run

[`lever-decode-reproducibility-exposure12-seed47-measured-results.md`](lever-decode-reproducibility-exposure12-seed47-measured-results.md)
(PR #1164, merged) confirmed decode non-determinism across 4 re-evals of an
identical checkpoint+flags (`meaningful_program_rate` 0.0–0.67, hero outcome
`model_valid` only once) and named its next lever as: *"find/seed decode RNG
(`evaluate_model` seed if available) or enforce deterministic ranking."*

This session independently reproduced the same core finding (3 re-evals,
`reward_score`/`decode_outcome_counts` varying on an identical
`checkpoint_sha256`) before discovering PR #1164 had just landed on `main`
with a stronger version of the same result — so rather than file a
near-duplicate, this doc tests the *specific* next-lever hypothesis PR #1164
proposed: is the divergence caused by an uncontrolled RNG/hash source that a
fixed `PYTHONHASHSEED` would eliminate?

`--seed 47` already reaches a properly-seeded RNG in the decode path
(`src/slm_training/dsl/grammar/fastpath/lattice_search.py:trajectory_orders`
derives `random.Random(f"{seed}:{ranked.signature}:{trajectory}")` — seeded,
deterministic). The remaining plausible uncontrolled source: Python's
per-process string-hash randomization (`PYTHONHASHSEED`, randomized by
default since Python 3.3), which can silently affect `set`/`dict` iteration
order in any tie-breaking code that iterates a hash-keyed collection without
an explicit sort — and each `evaluate_model` invocation is a fresh process,
so a different hash seed every run.

## Recipe

Same frozen checkpoint as the within-session determinism check this session
also ran (`lever_determinism_seed47`, trained on `lever_exposure12_v1`,
`last_loss=7.188689231872559` — matches the historical exposure12 seed47
champion bit-for-bit, so training itself is not in question here):

```bash
for i in 1 2 3; do
  PYTHONHASHSEED=0 python -m scripts.evaluate_model \
    --test-dir outputs/data/eval/v1 --suite smoke \
    --train-dir src/slm_training/resources/data/train/lever_exposure12_v1 \
    --model twotower --device cpu --run-id lever_determinism_seed47 \
    --grammar-constrained --decode-timeout-seconds 30 --grammar-ltr-max-tokens 64 \
    --constraint-debt-routing-mode fixed_asap --run-class scratch_matrix
done
```

Environment: same session as the within-session check — fresh
`.venv` (Python 3.12, `torch==2.5.1+cu124`), `npm ci` at repo root and in
`src/apps/openui_bridge`.

## Results

| run | parse | meaningful | reward | empty | decode_outcome_counts | total_ms_sum |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 1.0 | 0.3333333333333333 | 0.86 | 0 | `{model_valid:1, model_invalid:1, model_abstain:0, runtime_timeout:0, fallback_output:1, harness_error:0}` | 124104.95 |
| 2 | 1.0 | 0.3333333333333333 | 0.7653333333333334 | 0 | `{model_valid:0, model_invalid:0, model_abstain:0, runtime_timeout:0, fallback_output:3, harness_error:0}` | 90012.42 |
| 3 | 1.0 | 0.6666666666666666 | 0.8543333333333333 | 0 | `{model_valid:1, model_invalid:0, model_abstain:0, runtime_timeout:0, fallback_output:2, harness_error:0}` | 132719.22 |

All three runs: identical `checkpoint_sha256`, identical `--seed 47`,
identical `PYTHONHASHSEED=0`, identical CLI flags, identical input suite.

## Decision

**HYPOTHESIS_REFUTED** — pinning `PYTHONHASHSEED` does **not** restore
determinism. `meaningful_program_rate` still swings 0.333↔0.667,
`decode_outcome_counts`' `model_valid` count varies 0/1/1, and
`total_ms_sum` varies **90012–132719ms** (a ~47% spread) with every other
input frozen. This rules out both "unseeded RNG" and
"hash-randomization-driven iteration order" as the root cause; the fastpath
lattice search's own RNG was already properly seeded and was not the source
of divergence tested here.

## Revised root-cause hypothesis

The wide `total_ms_sum` spread on a suite whose per-record
`decode_timeout_seconds=30` nominally caps each record is consistent with
the mechanism already flagged — but not fixed — in
[`decode-timeout-hang-seed44-steps72-finding.md`](decode-timeout-hang-seed44-steps72-finding.md):
`decode_timeout_seconds` is enforced via `signal.setitimer(ITIMER_REAL, ...)`
+ `SIGALRM`, a **soft** real-wall-clock limit CPython only delivers between
bytecode instructions. If a record's generation call is blocked inside a C
extension (a `torch` op, the DFA/trie compiler search) that doesn't yield
back to Python bytecode, the alarm is delayed until that call returns — so
whether a record finishes as `model_valid` (full search, potentially
minutes) or gets cut to `fallback_output` (~30s) depends on **real CPU
scheduling at the moment of the call**, not on any seedable state.

This sandbox was independently confirmed to be running concurrent sessions
during this window: three other autotrain-lever PRs (#1162, #1163, #1164)
landed on `main` within the same ~10-minute span as this run (this session's
local git history was stale relative to `origin/main` until a mid-run
`git fetch` surfaced them). CPU contention from sibling sessions is a
plausible — not excluded — contributor to the specific magnitudes measured
here, layered on top of the architectural soft-limit gap. This doc does not
attempt to separate those two factors.

## Next steps (evidence-backed)

1. **Correct the PR #1164 next-lever note**: "find/seed decode RNG" is not
   the fix — this run rules that framing out directly. The next lever
   belongs to `improve-openui-harnesses`: replace or augment
   `decode_timeout_seconds`'s `SIGALRM`-based soft limit with a mechanism
   that yields a hard bound regardless of what the blocked call is doing
   (a watchdog thread + `os.kill`, or subprocess isolation for the
   grammar-constrained decode/compiler-search step), per next step 2 of
   `decode-timeout-hang-seed44-steps72-finding.md`.
2. Until that lands, treat every single-run smoke-scale latency AND quality
   number in this repo's lever docs as **one draw from a process with a
   real, uncontrolled wall-clock race in it** — not just training-seed
   noise. Multi-rep medians over repeated evals of the *same* checkpoint are
   needed before trusting any decode-latency or small quality delta.
3. To separate "architectural soft-limit gap" from "this sandbox happened to
   be contended by sibling sessions," a repeat of this exact check on an
   otherwise-idle machine (or with `taskset`/cgroup CPU pinning) would
   isolate the contribution of shared-sandbox contention specifically —
   not attempted here (out of scope for a single scheduled docs iteration).

Captured: 2026-07-27T17:52:00+00:00
