# Hard LTR decode timeout wall — NOT SHIP

**Honesty:** fixture/scratch smoke n=3. **Not ship.**

## Hypothesis
A **cooperative monotonic deadline** (checked each LTR/MaskGIT step) plus
**interval SIGALRM** re-fire enforces `decode_timeout_seconds` so decode cannot
run for minutes past budget (previous: hero 160–277s). Seeded multi-rep evals
should then be more stable.

## Harness change
| component | version | change |
| --- | --- | --- |
| `harness.model_build.eval` | **v63** | `set_decode_deadline` / interval SIGALRM in `eval_runner` |
| `model.twotower` | **v261** | `check_decode_deadline()` in LTR + MaskGIT loops |
| tests | | `tests/test_harnesses/model_build/test_decode_deadline.py` |

## Results (same quality ckpt, ASAP, t30, `--seed 47`)

### Pre-wall seeded (v62 seed lock only)
| rep | meanful | max_lat_ms | hero_outcome | hero_lat |
| ---: | ---: | ---: | --- | ---: |
| 1 | 0.6666666666666666 | 163248 | model_valid | 163248.21 |
| 2 | 0.6666666666666666 | 171309 | model_valid | 171309.39 |
| 3 | 0.3333333333333333 | 193361 | fallback_output | 30007.72 |

### Hard-wall seeded (v63)
| rep | meanful | parse | empty | timeouts | max_lat_ms | sum_lat | hero_outcome | hero_lat | hero_stop |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 1 | 0.3333333333333333 | 1.0 | 0 | 0 | 30033 | 90042 | fallback_output | 30007.02 | completed |
| 2 | 0.3333333333333333 | 1.0 | 0 | 0 | 30007 | 90013 | fallback_output | 30007.47 | completed |
| 3 | 0.3333333333333333 | 0.6666666666666666 | 1 | 1 | 32912 | 92928 | runtime_timeout | 32912.25 | decode_timeout |

## Decision
**ACCEPT** — hard wall holds: max_lat per sample ≤35000ms; seeded meanful stable at 0.3333333333333333

### Implications
- Quality single-run meanful=0.67 that required 160s+ hero is **outside the hard wall**.
- Authoritative micro metrics under t30 must use multi-rep medians with `--seed` and the wall.
- Next quality work: improve model so hero finishes **inside** budget, not by relaxing the wall.

Captured: 2026-07-27T18:06:11.390196+00:00
