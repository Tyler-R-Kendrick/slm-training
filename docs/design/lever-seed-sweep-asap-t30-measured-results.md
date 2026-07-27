# Multi-seed recipe sweep: s30/s36 + ASAP + t30

**Honesty:** fixture_or_scratch / smoke n=3. **Not ship.**

```
# Multi-seed recipe sweep (ASAP + decode_timeout=30) — NOT SHIP
captured=2026-07-27T14:18:23.823239+00:00
seed steps             stop     loss  parse  meanf  reward empty tout       lat
  42    36            steps   12.026  1.000  0.333   0.844     0    0    8994.7
  43    30            steps   20.599  0.000  0.000   0.000     3    3   30000.8
  44    36            steps    7.112  0.667  0.333   0.463     1    1   13988.6
  45    30            steps    9.836  0.000  0.000   0.000     3    3   30005.0
  46    30            steps   10.848  0.000  0.000   0.000     3    3   30002.1
  47    30            steps   22.679  1.000  0.333   0.765     0    0   17282.5

success_seeds(parse>=0.5 & empty=0): []
hard_fail_seeds(empty=3): [43, 45, 46]
success_rate: 0/6
parse_rate: mean=0.444 median=0.333
meaningful_program_rate: mean=0.167 median=0.167
reward_score: mean=0.345 median=0.232

## Lever conclusions (deterministic micro-recipe)
1. **decode_timeout=30** is necessary for non-empty constrained decode on good seeds.
2. **ASAP** preserves quality and cuts latency ~2x when quality is already good.
3. **SFT.steps** helps on good seeds (8→36) but does not guarantee learning for all seeds.
4. **Seed variance is the dominant remaining failure mode** on wf_smoke_v2 micro-train: ~half of seeds hang in decode until timeout even with t=30/45/60.
5. gen_steps↑ and continue-SFT initialize-from rejected for this setup.

## Next lever (evidence-backed)
Target seed brittleness: longer train wall / larger data / curriculum — not more smoke iters.

```

Captured: 2026-07-27T14:18:23.824243+00:00
