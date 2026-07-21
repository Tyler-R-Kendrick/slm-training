# SLM-240 (LRS0-01): learning-rate schedule gap probe (slm240-lrs0-01-lr-schedule-gap-20260721)

**Matrix set:** `slm240_lr_schedule_gap`
**Version:** `lrs0-01-v1`
**Status:** fixture
**Claim class:** wiring

## Hypothesis

The real model_build train loop (harnesses/model_build/train_loop.py::train) applies a bit-identical, constant learning rate to every optimizer parameter group for the entire duration of a run, for both supported optimizers (adamw, muon_hybrid), with no warmup ramp-up and no decay -- and the per-step metrics.jsonl row never records the applied learning rate, so no existing telemetry could detect a schedule even if one existed.

## Falsifier

Any optimizer.step() call recorded during a >=10-step training run reports a parameter-group lr that differs from the group's first recorded lr (i.e. some schedule already governs it in this code path), or a first recorded lr does not match the configured lr/muon_lr/adamw_lr value, or any metrics.jsonl row for the run contains an 'lr' or 'learning_rate' key.

## Honest caveats

- Fixture/wiring evidence only: a tiny scratch-backend TwoTower model training on 4 synthetic records for 20 optimizer steps says nothing about whether a schedule *should* exist or what it would do to real convergence -- this probe only asks whether one exists today.
- Instrumentation is a monkeypatch spy on torch.optim.AdamW.step and MuonHybrid.step, scoped to a try/finally around a single train() call and always delegating to the original method; no production file is edited and no optimizer behavior is altered, but a spy is still an external observation mechanism rather than a first-class API, so it would need to be re-verified if either optimizer's step() signature changes.
- This finding is about the current absence of a mechanism, not a claim that a schedule would help or hurt convergence at any scale; it also does not cover the causal-LM/HF Trainer track (models/causal_lm_openui.py), which does configure lr_scheduler_type and warmup_ratio through the standard HF Trainer.
- Only two optimizer configurations (adamw, muon_hybrid) and three seeds were probed; this does not rule out a schedule being applied only under some other config combination this probe did not exercise.

## Probe

- optimizer steps per arm: 20
- records: 4
- optimizers: ['adamw', 'muon_hybrid']
- seeds: [0, 1, 2]

## Per-arm results

| optimizer | seed | steps recorded | configured lrs | lr constant? | lr matches config? | metrics logs lr? | finite? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adamw | 0 | 20 | adamw=0.0003 | True | True | False | True |
| adamw | 1 | 20 | adamw=0.0003 | True | True | False | True |
| adamw | 2 | 20 | adamw=0.0003 | True | True | False | True |
| muon_hybrid | 0 | 20 | adamw=0.0001, muon=0.0005 | True | True | False | True |
| muon_hybrid | 1 | 20 | adamw=0.0001, muon=0.0005 | True | True | False | True |
| muon_hybrid | 2 | 20 | adamw=0.0001, muon=0.0005 | True | True | False | True |

## Summary

- all_lr_constant: True
- all_lr_matches_config: True
- any_metrics_log_lr: False
- all_finite: True

## Disposition

**gap_confirmed**

Every arm's live optimizer.step() lr was bit-identical across all recorded steps and matched the configured lr/muon_lr/adamw_lr value, for both adamw and muon_hybrid, across all seeds -- the model_build train loop applies no learning-rate schedule (no warmup ramp, no decay) today. No metrics.jsonl row in any arm logged an lr/learning_rate field, so this would be invisible to existing per-step telemetry even if a bug caused the lr to drift.

## Go / no-go decision

**No-go for any 'schedule already works' claim; honest gap confirmation.** This is wiring/fixture evidence over a tiny scratch-backend model and synthetic overfit data, not a quality or ship claim. A `gap_confirmed` disposition means the model_build TwoTower train loop has no learning-rate warmup or decay mechanism today, and no per-step telemetry field would surface one if it existed -- both real gaps for a future SLM to close, not something this probe fixes.

## Reproducibility

```bash
python -m scripts.run_slm240_lr_schedule_gap --mode plan-only
python -m scripts.run_slm240_lr_schedule_gap --mode fixture
```

