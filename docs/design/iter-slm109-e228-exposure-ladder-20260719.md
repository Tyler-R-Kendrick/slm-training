# SLM-109 — E228 ≥100× training-exposure checkpoint ladder (wiring)

SLM-109 tests whether the E228 legal-candidate-margin recipe is simply
underexposed or whether the representation/objective/recipe itself is
mis-specified. The experiment freezes every recipe detail from the original E228
run and scales cumulative target-token exposure from the original ~6.4k tokens
to at least 100× (~640k+ tokens), evaluating a durable checkpoint ladder at 1×,
4×, 16×, 64×, and 128×.

This iteration lands the harness, recipe-freeze manifest, and plan/fixture
wiring. No actual training was run.

## Frozen E228 recipe

The recipe is taken directly from
[`iter-e228-candidate-margin-alignment-20260716.json`](iter-e228-candidate-margin-alignment-20260716.json).

| Field | Value |
| --- | --- |
| Model | TwoTower, HF local SmolLM2 context |
| Output tokenizer | `lexer` |
| Batch / LR / seed | 4 / 0.0003 / 0 |
| Context backend | `hf_local_files_only` |
| `compiler_alignment_loss_weight` | 1.0 |
| `compiler_alignment_margin` | 1.0 |
| `compiler_alignment_stratified` | true |
| `compiler_alignment_semantic_exhaustive` | true |
| `compiler_decode_mode` | `tree` |
| Schema/slot contract | in context, constrained decode, honest contract |
| `allow_unconstrained_fallback` | false |
| Mixture sampling | `quota_capacity_aware` |
| Base target tokens (T0) | 6401 |
| Recipe SHA-256 | `cb77a897141ebf438c1ff5c6a1a3d70628df175d32138a9ae6b30fba29ebd8de` |

Any hash mismatch between ladder checkpoints is a failed experiment, not a new
arm.

## Exposure ladder

| Multiplier | Target tokens | Purpose |
| --- | --- | --- |
| 1× | 6,401 | Reproduction / bit-exact resume baseline |
| 4× | 25,604 | First ladder checkpoint |
| 16× | 102,416 | Mid-ladder |
| 64× | 409,664 | Late-ladder |
| 128× | 819,328 | Satisfies ≥100× threshold |

Seeds: 0, 1, 2 for the decisive endpoints; seed 0 drives the full diagnostic
curve. A claim that reverses prior conclusions requires five seeds.

## Commands

```bash
# Plan only (no model load)
python -m scripts.run_e228_exposure_ladder --mode plan-only \
  --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
  --output-dir outputs/runs/slm109_e228_ladder

# Fixture wiring check
python -m scripts.run_e228_exposure_ladder --mode fixture \
  --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
  --output-dir outputs/runs/slm109_e228_fixture
```

Frontier dispatch (GPU + durable checkpoint required):

```bash
python -m scripts.hf_jobs_train \
  --run-id e228-ladder-m4-s0 \
  --steps 3200 \
  --extra-train-args "--resume-from outputs/runs/e228-candidate-margin-matched/checkpoints/last_full_state.pt --target-token-budget 25604"
```

## Results

| Metric | Fixture value |
| --- | --- |
| Status | fixture |
| Points planned | 15 (5 multipliers × 3 seeds) |
| Points run | 0 |
| Ship gates | not claimed |

## Honest caveats

- This is a wiring and planning iteration only.
- No GPU training was performed; all ladder points are `fixture_planned`.
- The 1× reproduction requires the original E228 checkpoint or a verified
  bit-exact retrain.
- Durable checkpoint sync and provenance follow SLM-103.
- Ship gates are intentionally not claimed.

## Next step

Dispatch the frontier ladder on a GPU host from the verified E228 checkpoint,
sync checkpoints to `hf://buckets/TKendrick/OpenUI`, and publish the full
semantic/AgentV/loss/cost curves with paired confidence intervals.

Machine-readable evidence is in
[the SLM-109 JSON](iter-slm109-e228-exposure-ladder-20260719.json).
