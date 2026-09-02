# P8: measured cold-start steps/s prior for the screening steps fitter

Date: 2026-09-02
Status: implemented; scratch profile evidence only
Scope: `scripts/run_autotrain_continuous.py` (`_COLD_START_STEPS_PER_SEC`,
`_COLD_START_STEPS_PER_SEC_EVIDENCE`, `_fit_screening_steps`,
`_ensure_climb_champion`, `_park_champion_epochs_if_needed`,
`_warm_start_policy`), `src/slm_training/autoresearch/hillclimb.py`
(`baseline_seed` champion status, manifest-driven epoch cap). JSON mirror:
[`p8-screening-cold-start-steps-prior-20260902.json`](p8-screening-cold-start-steps-prior-20260902.json).
Honesty: **scratch profile, not a capability, promotion, or ship claim.** A
warm-started `baseline_seed` champion is never a ship or promotion claim.

## Why

RC5 of the
[hill-climb recovery directive](autotrain-hillclimb-recovery-directive.md):
each screening arm trained ~20-22 steps from random init every cycle. The
steps fitter used an unmeasured `5.0` steps/s cold-start prior, and the
champion seeder only accepted `confirmed | climb_accepted | promoted` queue
rows, which never existed (`docs/MODEL_CARD.md`), so
`initialized_weight_count` was `0` every cycle.

## Measurement (steps/s, this box)

Host: 4 CPU, no GPU, `torch 2.5.1+cpu`, sibling agents sharing the CPUs
(1-minute load average at launch recorded per run). Trainer:
`python -m scripts.train_model` at the exact screening launch shape emitted by
`autoresearch/engine.py`:

```
--context-backend scratch --device cpu --lr 3e-4 --seed 0 \
--train-version wf_smoke_v2 --local-files-only --no-sync-checkpoints \
--no-full-state-checkpoint [--batch-size 2|3]
```

`trainer_default_arch` is the trainer's default architecture (d_model 128,
4 heads, 2+4 layers, 1.6M trainable) that screening arms actually launch.
`e53_scratch_arch` adds the E53 recipe shape from
`harness_core/lineage/tracks.py` (`--d-model 192 --n-heads 6
--context-layers 3 --denoiser-layers 6 --mask-pattern mixed`, 5.1M
trainable); the frozen SmolLM2 `hf` context tower is not installable here
(`transformers` absent), so it is a scratch-context number. Every run
finished on `stopped_on == "steps"` (`steps_per_sec = steps /
elapsed_wall_seconds` from `train_summary.json`; process wall includes
interpreter start-up and is not part of the fit).

| run | arch | batch | steps | train wall s | process wall s | steps/s | load 1m | trainable |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `initial_b2_s200_r1` | trainer_default_arch | 2 | 200 | 68.82 | 79.02 | **2.91** | 4.17 | 1,608,962 |
| `initial_b2_s200_r2` | trainer_default_arch | 2 | 200 | 10.47 | 16.06 | **19.11** | 10.83 | 1,608,962 |
| `initial_b2_s200_r3` | trainer_default_arch | 2 | 200 | 11.15 | 16.75 | **17.94** | 8.25 | 1,608,962 |
| `initial_b3_s200_r1` | trainer_default_arch | 3 | 200 | 12.68 | 18.38 | **15.77** | 7.18 | 1,608,962 |
| `initial_b3_s200_r2` | trainer_default_arch | 3 | 200 | 12.82 | 18.55 | **15.61** | 5.87 | 1,608,962 |
| `initial_b3_s200_r3` | trainer_default_arch | 3 | 200 | 12.18 | 17.93 | **16.42** | 5.09 | 1,608,962 |
| `e53_b2_s200_r1` | e53_scratch_arch | 2 | 200 | 19.92 | 25.33 | **10.04** | 3.24 | 5,125,058 |
| `e53_b2_s200_r2` | e53_scratch_arch | 2 | 200 | 19.93 | 25.95 | **10.04** | 3.42 | 5,125,058 |
| `e53_b2_s200_r3` | e53_scratch_arch | 2 | 200 | 18.52 | 24.55 | **10.80** | 3.41 | 5,125,058 |
| `final_b2_s200_r1` | trainer_default_arch | 2 | 200 | 10.19 | 15.81 | **19.63** | 2.32 | 1,608,962 |
| `final_b2_s200_r2` | trainer_default_arch | 2 | 200 | 10.64 | 16.18 | **18.80** | 2.33 | 1,608,962 |
| `final_b2_s200_r3` | trainer_default_arch | 2 | 200 | 10.87 | 16.64 | **18.40** | 2.49 | 1,608,962 |

Summary over 12 complete runs: min **2.906**,
median **16.091**, max **19.633**
steps/s. The spread is CPU contention from concurrent agents, not recipe
variance (same seed, data, and knobs).

## Decision

- `_COLD_START_STEPS_PER_SEC = 2.9` — the slowest
  measured run rounded down to 0.1, so a cold fit never overshoots the train
  floor under contention (an overshoot truncates on the wall budget, and a
  truncated run is never evidence). Telemetry from
  `*/runs/*/train_summary.json` replaces the prior after the first arm.
- Expected screening steps at a 100 s train floor
  (`steps = clamp(floor * sps * 0.9, 1, 400)`): cold prior →
  **261** steps; measured median
  telemetry → **400** steps (cap
  400). At the old 20 s floor and 5.0 prior the same fitter gave 90 steps;
  the grown floor handed over by the decode fit is what buys steps, never a
  larger model.
- The fitter reports `steps_per_sec_source` (`cold_start_prior` vs
  `train_telemetry`) and embeds this evidence block in `steps_fit`.

## Champion warm start (`baseline_seed`)

- `_ensure_climb_champion` still seeds a `confirmed` champion from
  `confirmed | climb_accepted | promoted` rows. When none exists and
  `measurement.warm_start.seed_from_baseline_control` (default on) is set,
  the loop's first complete control run (`stopped_on == "steps"`,
  `steps > 0`, `checkpoints/last.pt` present; wall/token truncations never
  qualify) seeds a sidecar with `status: "baseline_seed"`
  (`climb_champion/v2`).
- `baseline_seed` is excluded from every confirmed / promotion consumer:
  `CLIMB_CHAMPION_ADVANCE_STATUSES`, `CLIMB_BASELINE_STATUSES`,
  `_PROMOTE_AUTHORITY_STATUSES`, `_CHAMPION_STATUSES`,
  `_RETRYABLE_PROMOTE_STATUSES`, `_is_decisive_causal_terminal`,
  `_should_enqueue_champion`. It only enables paired warm start: both arms
  launch with `initialize_from` = the champion checkpoint and
  `assert_warm_start_launch` still rejects unequal K / data / seed /
  checkpoint (directive §2.6).
- `maybe_advance_climb_champion` keeps confirmed semantics: a `baseline_seed`
  champion never advances on a screening win (explicit no-advance); only a
  confirmed result replaces it, and the replacement is `status: "confirmed"`.
- The cumulative-epoch cap (`max_cumulative_epochs`, default 50) is
  recomputed at park time as `cumulative_steps / record_count` with
  `record_count` read from the sidecar's train corpus manifest
  (`<train_dir>/manifest.json`); the accumulated sidecar value is the
  fallback when the manifest is unresolvable, so a missing manifest never
  lifts the cap.

## Tests

`tests/test_scripts/test_run_autotrain_continuous.py` (P8 block) and
`tests/test_autoresearch/test_climb_champion.py`: fitter provenance and the
100 s-floor expectations above; `baseline_seed` created after a complete
control and rejected by every promotion consumer; no-advance on screening
wins; manifest-driven epoch cap; unequal-pair launch assertion; and a
unit-level two-cycle simulation in which cycle 2 completes its control and
the next cycle launches both arms with `initialize_from` set to the seeded
champion checkpoint.

## Non-goals

No model, gate, or metric change; no promotion; no claim about the `hf`
context backend's speed. Version stamp:
`harness.autoresearch.experiment_campaign` v272.
