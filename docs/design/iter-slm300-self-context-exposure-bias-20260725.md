# SLM-300 AP-015 — Self-context exposure-bias curriculum (wiring)

## What
Preregistered self-context curriculum manifest and exact `policy_origin_mixture`
math for AP-015. The experiment tests whether train-decode state mismatch
(exposure bias) explains binder/reference recovery failures, and whether
mixing model-generated ("self-context") partial states into the corruption
schedule -- instead of exclusively gold-label-derived corruption -- improves
recovery without validity regression or a hidden training-budget increase.

## Matrix registration
- `matrix_set`: `self-context-exposure-bias`
- `matrix_version`: `ap015-v1`
- `curriculum_id`: `self-context-scheduled-corruption`

## Arms

| Arm | Self-context rate | Purpose |
| --- | --- | --- |
| A_control | 0%  | clean/legacy control (mixture-zero invariant) |
| SC10      | 10% | low self-context mixture |
| SC25      | 25% | medium self-context mixture |
| SC50      | 50% | high self-context mixture |

Within each non-zero arm, the self-context mass is split between the current
policy and a lagged/EMA policy checkpoint by `lagged_policy_share` (default
50/50) — this is the "current-versus-lagged policy origin" provenance
required by the issue, and it damps the instability of purely on-policy
self-distillation.

## Policy origin mixture

`policy_origin_mixture(self_context_rate, lagged_policy_share)` is a pure,
deterministic function (no training required):

```text
gold          = 1 - self_context_rate
model_lagged  = self_context_rate * lagged_policy_share
model_current = self_context_rate - model_lagged
```

At `self_context_rate == 0.0` the mixture is always
`{gold: 1.0, model_current: 0.0, model_lagged: 0.0}` for every
`lagged_policy_share` — this is the exact, tested proof that the mixture-zero
arm reproduces legacy (gold-only) corruption behavior, satisfying the issue's
"Mixture zero reproduces legacy behavior" acceptance criterion at the wiring
level.

## Frozen base recipe
The base recipe mirrors the SLM-120 corruption-curriculum recipe shape
(CPU device, matched steps/batch/seed fields) so both curricula remain
directly comparable at matched exposure. Its SHA-256 is stored in the
manifest.

## Files added
- `src/slm_training/harnesses/experiments/self_context_curriculum.py`
- `scripts/run_self_context_curriculum.py`
- `tests/test_harnesses/experiments/test_self_context_curriculum.py`
- `tests/test_scripts/test_self_context_curriculum.py`
- `docs/design/iter-slm300-self-context-exposure-bias-20260725.md`
- `docs/design/iter-slm300-self-context-exposure-bias-20260725.json`

## Commands

```bash
# Plan only (CPU, no model load)
python -m scripts.run_self_context_curriculum --mode plan-only \
  --output-dir outputs/runs/slm300_self_context_curriculum

# Fixture wiring check
python -m scripts.run_self_context_curriculum --mode fixture \
  --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/certified-baseline/ref.json \
  --output-dir outputs/runs/slm300_self_context_fixture
```

## Verification
- `pytest tests/test_harnesses/experiments/test_self_context_curriculum.py -q` → 19 passed
- `pytest tests/test_scripts/test_self_context_curriculum.py -q` → 4 passed
- `python -m scripts.verify_version_stamps --check` → ok

## Honest caveats
This is **wiring evidence only**. The policy-origin-mixture math is exact and
unit-tested (including the mixture-zero legacy-equivalence invariant), but
the actual multi-seed self-context/scheduled-corruption training arms —
collecting model-generated partial states from the certified baseline,
training matched variants at self-context rates 0/0.10/0.25/0.50, and
evaluating recovery by binder/reference mutation class and denoising round —
require a GPU host, the certified baseline checkpoint, and durable HF bucket
sync per SLM-103. The `frontier` mode emits a fixture plan and raises a clear
stderr message. No recovery or ship-gate claim is made from this artifact;
`binder_reference_recovery` and `denoising_round_recovery` fields remain
`null` until a real training run populates them.
