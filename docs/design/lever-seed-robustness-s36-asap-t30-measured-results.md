# Seed robustness: s36 + ASAP + decode_timeout=30

**Honesty:** fixture_or_scratch / smoke n=3. **Not ship.**

## Hypothesis
Recommended micro-recipe quality is not seed-specific (seeds 42/43/44).

## Results
```
# Seed robustness: recipe s36 + ASAP + decode_timeout=30 (NOT SHIP)
seed42: parse=1.0 meanful=0.3333333333333333 reward=0.8436666666666667 empty=0 timeout=0 lat_p50=8994.71 loss=12.026487350463867 stopped=steps seed=42
seed43: parse=0.0 meanful=0.0 reward=0.0 empty=3 timeout=3 lat_p50=30001.61 loss=5.497641086578369 stopped=wall_time_budget seed=43
seed44: parse=0.6666666666666666 meanful=0.3333333333333333 reward=0.463 empty=1 timeout=1 lat_p50=13988.58 loss=7.11176872253418 stopped=steps seed=44
parse_rate: mean=0.5556 min=0.0000 max=1.0000
meaningful_program_rate: mean=0.2222 min=0.0000 max=0.3333
reward_score: mean=0.4356 min=0.0000 max=0.8437
latency_ms_p50: mean=17661.6333 min=8994.7100 max=30001.6100

```

## Decision
Record seed-mean metrics. If variance high, prioritize data/decode over more steps.

Captured: 2026-07-27T13:55:07.240863+00:00
