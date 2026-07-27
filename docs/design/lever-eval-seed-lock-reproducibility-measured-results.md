# Eval `--seed` lock for decode reproducibility — NOT SHIP

**Honesty:** fixture/scratch smoke n=3. **Not ship.** Harness + measurement.

## Hypothesis
Adding `evaluate_model --seed` and seeding Python/NumPy/Torch at eval start makes
re-evals of the exposure12 quality checkpoint **identical** under greedy constrained
decode (same flags).

## Harness change
- component `harness.model_build.eval` **v62**
- `scripts/evaluate_model.py --seed` → `ModelBuildConfig.seed`
- `eval_runner._seed_eval_rng` + stamp `seed` on scoreboard
- unit test: `tests/test_harnesses/model_build/test_eval_seed.py`

## Results (same ckpt + ASAP + t30)

### Unseeded re-evals (prior, default seed=0 implicit)
| rep | meanful | hero_outcome | hero_lat |
| ---: | ---: | --- | ---: |
| 1 | 0.3333333333333333 | fallback_output | 30005 |
| 2 | 0.0 | fallback_output | 30008 |
| 3 | 0.3333333333333333 | fallback_output | 30006 |

### Seeded re-evals (`--seed 47`)
| rep | seed | meanful | parse | empty | hero_outcome | hero_lat |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 47 | 0.6666666666666666 | 1.0 | 0 | model_valid | 163248 |
| 2 | 47 | 0.6666666666666666 | 1.0 | 0 | model_valid | 171309 |
| 3 | 47 | 0.3333333333333333 | 1.0 | 0 | fallback_output | 30008 |

## Decision
**PARTIAL** — seed lock incomplete: seeded unique meanful=[0.3333333333333333, 0.6666666666666666] hero=['fallback_output', 'model_valid']; residual wall/timeout race likely

### Notes
- Original meanful=0.67 hero=`model_valid` @277s remains an outlier vs both
  unseeded and seeded re-evals (likely timeout/wall race or env drift, not pure RNG).
- Prefer multi-rep medians for quality claims until timeout path is fully wall-deterministic.

## Next lever
If partial: enforce hard wall on LTR (no 277s runs) + multi-rep medians.
If accept: re-run quality levers with `--seed 47` multi-rep.

Captured: 2026-07-27T17:56:37.564246+00:00
