# iter-s12 — per-step commit-authority histogram (N3)

**Date** 2026-09-02 · **Card** S12 · **Hypothesis** N3 · **Verdict: falsified**
· **Honesty: fixture-demo, not ship.** JSON mirror:
[`iter-s12-commit-authority-profile-20260902.json`](iter-s12-commit-authority-profile-20260902.json).

## Hypothesis and falsifier

**N3** — on grammar-constrained canvases the "first step commits the most
tokens" profile is dominated by **forced (I2)** commits: forced share at step 1
> 50 %. Falsifier: the **confidence-committed** share dominates.

**The falsifier holds.** Across all six suite × `parallel_unmask` arms the
step-1 forced share is **6.4 % – 38.2 %**; the confident share is
**61.8 % – 93.6 %** and dominates every single step of every single arm. The
speculative share is **0.0 % everywhere**. N3 is rejected.

## Instrument

New default-off telemetry on `DecodeStats`:

* `record_step_commits: bool = False` — the switch.
* `step_commits: list[dict]` — one row per denoising step that committed at
  least one position: `{step, committed, forced, confident, speculative,
  forwards, authority}`.

Recorded at two sites in `_generate_maskgit_one`: the I2 one-hole
`exact_commit` branch (one forced commit, `forwards = 0`) and the end of the
positionwise/cluster commit loop, after the block/hybrid joint validator has
had its chance to revert. Authority is **read off the real decision**, never
re-derived: `_propose` already computes the DFA `force_emit_token_id` proof for
each position, so `forced` counts positions the grammar proved and `confident`
counts positions the model's constrained argmax chose. `speculative` is the
per-step delta of `speculative_rank_commits` (I3).

S4's derived fraction properties are untouched. `merge()` now skips `bool`
fields — `bool` is an `int` subclass, so the old merge would have summed the
telemetry switch into a count.

**Off by default costs nothing.**
`tests/test_models/test_step_commit_histogram.py::
test_off_by_default_costs_nothing_and_stays_byte_identical` runs the same
prompts with the flag armed and unarmed and asserts an identical SHA-256 of the
joined outputs, plus identical `forwards_count` and `tokens_emitted`.

## Recipe

Same decode path, suites, substitute checkpoint and hardware as
[`iter-s6-context-ablation-20260902.md`](iter-s6-context-ablation-20260902.md)
— including the finding that the card's named measurement checkpoint
`src/slm_training/resources/checkpoints/playground_demo/last.pt` **cannot be
loaded** (`OutputContractError`, contract v0 vs required symbol_only/v2) and
that `scripts/bootstrap_playground.py` can no longer regenerate it. Arms here
are `parallel_unmask ∈ {topk, confidence, adaptive}` (the allowed values in
`harnesses/model_build/config.py`), suites smoke (8 of 96) and held_out (8 of
24), seed 0, `gen_steps = 8`. 6 runs, **0 timeouts**, slowest 50.8 s, all
inside `MAX_RUN_MINUTES = 3`.

## Measured results

Shares are of that step's commits; `share` is that step's fraction of the whole
committed canvas. Aggregated over 8 records × 8 steps per arm.

### smoke (n = 8)

| Step | `topk` share / forced / conf | `confidence` share / forced / conf | `adaptive` share / forced / conf |
| --- | --- | --- | --- |
| 1 | 11.3 % / 27.4 % / **72.6 %** | 14.2 % / 21.3 % / **78.7 %** | **34.5 %** / 13.2 % / **86.8 %** |
| 2 | 10.9 % / 22.2 % / 77.8 % | 13.7 % / 19.2 % / 80.8 % | 9.1 % / 38.9 % / 61.1 % |
| 3 | 11.2 % / 21.7 % / 78.3 % | 13.8 % / 20.0 % / 80.0 % | 9.9 % / 35.9 % / 64.1 % |
| 4 | 11.6 % / 17.4 % / 82.6 % | 13.8 % / 19.1 % / 81.0 % | 9.9 % / 35.9 % / 64.1 % |
| 5 | 12.2 % / 15.6 % / 84.4 % | **29.3 %** / 13.5 % / 86.6 % | 9.1 % / 38.9 % / 61.1 % |
| 6 | 12.7 % / 14.9 % / 85.1 % | 5.7 % / 20.9 % / 79.1 % | 9.1 % / 38.9 % / 61.1 % |
| 7 | 13.4 % / 14.1 % / 85.9 % | 5.5 % / 19.1 % / 81.0 % | 9.1 % / 38.9 % / 61.1 % |
| 8 | **16.7 %** / 11.3 % / 88.7 % | 4.0 % / 20.0 % / 80.0 % | 9.1 % / 38.9 % / 61.1 % |
| total commits | 741 | 760 | 394 |

### held_out (n = 8)

| Step | `topk` share / forced / conf | `confidence` share / forced / conf | `adaptive` share / forced / conf |
| --- | --- | --- | --- |
| 1 | 12.7 % / 38.2 % / **61.8 %** | 10.4 % / 31.7 % / **68.3 %** | **52.9 %** / 6.4 % / **93.6 %** |
| 2 | 10.7 % / 33.3 % / 66.7 % | 10.7 % / 32.1 % / 67.9 % | 6.8 % / 50.0 % / 50.0 % |
| 3 | 10.9 % / 32.8 % / 67.2 % | 10.9 % / 31.4 % / 68.6 % | 7.1 % / 47.8 % / 52.2 % |
| 4 | 11.1 % / 32.2 % / 67.8 % | 11.7 % / 29.4 % / 70.7 % | 6.8 % / 50.0 % / 50.0 % |
| 5 | 11.6 % / 30.7 % / 69.4 % | **22.2 %** / 17.7 % / 82.3 % | 6.8 % / 50.0 % / 50.0 % |
| 6 | 12.4 % / 30.3 % / 69.7 % | 14.4 % / 19.5 % / 80.5 % | 6.8 % / 50.0 % / 50.0 % |
| 7 | 14.6 % / 25.6 % / 74.4 % | 9.5 % / 22.7 % / 77.3 % | 6.5 % / 52.4 % / 47.6 % |
| 8 | **16.1 %** / 23.3 % / 76.7 % | 10.2 % / 23.8 % / 76.2 % | 6.5 % / 52.4 % / 47.6 % |
| total commits | 534 | 787 | 325 |

`speculative_share = 0.000` in **every cell** of both tables.
`parse_rate = 1.00`, `meaningful_program_rate = 0.00`,
`structural_similarity` 0.1381 (smoke) / 0.0955 (held_out) in all six runs —
the schedule changes *when* positions are revealed, not the certified program.
`forwards_count`: smoke 218 / 256 / 238, held_out 313 / 295 / 234
(topk / confidence / adaptive).

## Verdict

**N3 is falsified; the confidence-committed share dominates.**

1. **Forced never dominates step 1** — 27.4 / 21.3 / 13.2 % (smoke) and
   38.2 / 31.7 / 6.4 % (held_out), against a > 50 % claim. It never dominates
   *any* step under `topk` or `confidence` either. The only cells where forced
   reaches parity are steps 2–8 of `adaptive`, where the schedule has already
   spent the ambiguous positions in the opening burst and the tail is mostly
   grammar-determined lexemes.
2. **The "first step commits the most" premise is itself schedule-specific.**
   It holds only for `adaptive` (34.5 % smoke, 52.9 % held_out of the whole
   canvas in one step). `topk` is essentially flat and mildly *back*-loaded
   (11.3 % → 16.7 %), and `confidence` peaks at step 5, not step 1. N3's
   premise and its claim fail together.
3. **The anti-correlation is the real signal.** Where a step commits a lot, it
   commits *model* choices (adaptive step 1: 34.5 % of the canvas at 86.8 %
   confident); where a step commits little, the forced share rises toward 50 %.
   The big early burst is the model's, not the grammar's.
4. **I3 speculation is absent from this lane.** `speculative_rank` defaults to
   `off`, and the MaskGIT loop consults the ranker nowhere — `_select_compiler_path`
   is the only site that increments `speculative_rank_commits`. A three-way
   histogram on this path is therefore two-way in practice. Recorded as a
   structural fact, not a measurement.

## Caveats

1. **Fixture-demo, never ship.** 785 k parameters, 900 CPU steps, 8 records per
   suite; `meaningful_program_rate` is 0.00 everywhere, so no quality claim is
   available or implied.
2. **MaskGIT-only, seed 0.** The LTR-primary / compiler-tree lane a default
   evaluation actually takes has no recording site.
3. `forced` counts the DFA force-emit proof at the moment `_propose` ran. A
   position later remasked by the stream checker still appears in the step that
   committed it; the joint validator's reverts *are* excluded (the histogram is
   written after it runs), but a stream-check remask on the following step is
   not.
4. **`forced_tokens` is not this histogram.** It stayed 0 in every run while
   `semantic_singleton_bypasses` was 18–27, because the bypasses landed in
   constrained LTR repair, which increments `forced_row_tokens_without_forward`
   only. Flagged for S4's `forced_token_fraction`, not changed here.
