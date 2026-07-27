# Lever campaign summary: SFT steps × decode timeout × ASAP (seed=42)

**Honesty:** `fixture_or_scratch` / smoke `n=3`. **Not a ship claim.**

## Goal shift
Stopped pure smoke-mill iteration. Ran deterministic lever experiments via `slm sft train` / `slm eval model`.

## Scoreboard
```
# Campaign scoreboard (fixture smoke n=3, seed=42) — NOT SHIP
captured=2026-07-27T13:47:35.783119+00:00
arm                  parse_rate meaningful_program_rate reward_score decode_timeout_count empty_prediction_count placeholder_fidelity latency_ms_p50
s8_t12                   0.0000     0.0000     0.0000          3          3     0.0000 12007.1600
s48_t12                  0.3333     0.0000     0.3023          2          2     0.3333 12000.4400
s48_t30                  1.0000     0.3333     0.8437          0          0     0.7222 19192.2600
asap_s48_t30             1.0000     0.3333     0.8437          0          0     0.7222 10306.8800
s36_t30                  1.0000     0.3333     0.8437          0          0     0.7222 12998.8300
s36_asap_t30             1.0000     0.3333     0.8437          0          0     0.7222  8994.7100
continue_plus24_t30      0.0000     0.0000     0.0000          3          3     0.0000 30001.8400

champion=s36_asap_t30 metrics={'parse_rate': 1.0, 'meaningful_program_rate': 0.3333333333333333, 'reward_score': 0.8436666666666667, 'decode_timeout_count': 0, 'empty_prediction_count': 0, 'placeholder_fidelity': 0.7222222222222222, 'latency_ms_p50': 8994.71}

## Accepted levers
- SFT.steps↑ (partial): 8→~36 under t12 improves parse/timeouts
- decode_timeout 12→30: primary quality unlock (timeouts→0, parse→1.0 on s48)
- ASAP (fixed_asap): same quality as s48_t30, ~2× lower latency

## Rejected levers
- gen_steps 8→16 on s48@t12: no change
- continue SFT initialize-from +24: quality collapse despite lower loss
- s24 from-scratch @t30: worse than s8@t30 on this seed (non-monotonic micro-n)

## Recommended micro-recipe (not ship)
steps≈36, seed=42, twotower/scratch/cpu, eval decode_timeout=30, ASAP routing for latency

```

## Artifacts
- Runs under `outputs/runs/exp_lever_*`
- Scratch metrics: `/tmp/grok-goal-492cedee507b/implementer/autotrain_experiment_metrics.{txt,json}`
- Prior cycle notes: `docs/design/lever-sft-steps-8-vs-48-seed42-measured-results.md`, `docs/design/lever-decode-timeout-gen-steps-on-s48-measured-results.md`, `docs/design/lever-asap-and-continue-sft-seed42-measured-results.md`

## Champion
**s36_asap_t30** — grammar constrained retained.

## Next (until user stop)
Seed-robustness (seeds 43/44) of champion ASAP+t30 recipe; then larger data / HF context when micro-recipe plateaus.

Captured: 2026-07-27T13:47:35.784992+00:00
