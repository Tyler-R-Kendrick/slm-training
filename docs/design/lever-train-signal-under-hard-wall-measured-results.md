# Train-signal levers under hard 30s wall — NOT SHIP

**Honesty:** fixture/scratch smoke n=3. **Not ship.**

## Hypothesis
After decode knobs failed to lift meaningful inside the hard wall, train-side
levers (more steps, curriculum, higher lr) on `lever_exposure12_v1` raise
`meaningful_program_rate` vs hard-wall baseline (median **0.33**).

## Fixed eval protocol
ASAP · grammar constrained · `decode_timeout_seconds=30` (hard wall) · `--seed 47` · multi-rep

## Results

| arm | train | reps | meanful median | meanful vals | parse mean | empty mean | max_lat mean | wall_ok |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| baseline_s16 | s16 lr1e-3 no-curr (hardwall baseline) loss=7.189 | 3 | 0.333 | [0.3333333333333333, 0.3333333333333333, 0.3333333333333333] | 0.889 | 0.33 | 30984 | True |
| s32_train | s32 lr1e-3 train, hardwall eval loss=4.899 | 2 | 0.333 | [0.3333333333333333, 0.3333333333333333] | 1.000 | 0.00 | 30178 | True |
| curriculum_s16 | s16 lr1e-3 --curriculum loss=15.593 | 2 | 0.333 | [0.3333333333333333, 0.3333333333333333] | 1.000 | 0.00 | 30107 | True |
| lr3e3_s16 | s16 lr=3e-3 loss=10.215 | 2 | 0.000 | [0.0, 0.0] | 1.000 | 0.00 | 23812 | True |

## Decision
**REJECT** — no train-side arm beats baseline meanful_median=0.333 under hard wall; best s32_train=0.3333333333333333

### Decode screen (companion)
See [`lever-faster-decode-under-hard-wall-measured-results.md`](lever-faster-decode-under-hard-wall-measured-results.md):
ltr_max / max_attempts / gen_steps also **reject** under hard wall (no meanful lift).

### Combined implication
Under honest hard-wall scoreboards, micro fixture trains plateau at meanful ≈ 1/3.
Next: synthesizer/harness work for train-loadable richer data **and** multi-rep hard-wall
gates as default for lever acceptance — not more smoke hyperparam mills.

Captured: 2026-07-27T18:35:19.685795+00:00
