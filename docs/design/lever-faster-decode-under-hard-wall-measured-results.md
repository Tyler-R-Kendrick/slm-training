# Faster decode under hard 30s wall — NOT SHIP

**Honesty:** fixture/scratch smoke n=3. **Not ship.**

## Hypothesis

Under the hard `decode_timeout_seconds=30` wall, faster legal-decode knobs(`grammar-ltr-max-tokens`, `max-attempts`, `gen-steps`) let the hero sample finishinside budget and raise `meaningful_program_rate` above the hard-wall baseline(median **0.33** on exposure12 seed47 ckpt).

## Fixed recipe

- checkpoint: `exp_lever_data_exposure12_s16_lr1e3_bs2_sb15_seed47`
- ASAP + grammar constrained + `--seed 47` + t30 hard wall
- multi-rep medians (baseline n=3; treatments n=2 screen)

## Results

| arm | flags | reps | meanful median | meanful vals | parse mean | empty mean | max_lat mean | wall_ok |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| baseline_hardwall | `defaults (ltr_max~256, attempts=3, gen=8)` | 3 | 0.333 | [0.3333333333333333, 0.3333333333333333, 0.3333333333333333] | 0.889 | 0.33 | 30984 | True |
| ltr64 | `--grammar-ltr-max-tokens 64` | 2 | 0.167 | [0.3333333333333333, 0.0] | 0.667 | 1.00 | 30014 | True |
| ltr32 | `--grammar-ltr-max-tokens 32` | 2 | 0.333 | [0.3333333333333333, 0.3333333333333333] | 1.000 | 0.00 | 30058 | True |
| att1 | `--max-attempts 1` | 2 | 0.333 | [0.3333333333333333, 0.3333333333333333] | 0.833 | 0.50 | 30208 | True |
| gen4 | `--gen-steps 4` | 2 | 0.333 | [0.3333333333333333, 0.3333333333333333] | 0.833 | 0.50 | 30011 | True |
| ltr48_att1 | `--grammar-ltr-max-tokens 48 --max-attempts 1` | 2 | 0.333 | [0.3333333333333333, 0.3333333333333333] | 1.000 | 0.00 | 30161 | True |

## Decision

**REJECT** — no arm beats baseline meanful_median=0.333; best was ltr32 at 0.333 vals=[0.3333333333333333, 0.3333333333333333]

Decode-speed knobs alone do not lift meaningful inside the hard wall on thischeckpoint. Next lever: **train-side** signal on exposure12 (or loadablenon-fixture data) scored with hard wall + seed multi-rep, not more decode canvas cuts.

Captured: 2026-07-27T18:23:57.579912+00:00
