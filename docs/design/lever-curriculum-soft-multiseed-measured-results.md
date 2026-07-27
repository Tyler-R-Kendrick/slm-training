# Soft curriculum multi-seed lever (champion recipe) — NOT SHIP

**Honesty:** `fixture_or_scratch` / smoke suite `n=3`. **Not a ship claim.**

## Hypothesis

Soft `--curriculum` (A→B→C mix) improves multi-seed success rate and/or
`meaningful_program_rate` vs no-curriculum on the champion micro-recipe
(s16 · lr=1e-3 · bs=2 · structural_bias=1.5 · ASAP · decode_timeout=30 ·
`wf_smoke_v2` · scratch/cpu).

## Recipe

```bash
python -m scripts.train_model \
  --train-dir outputs/data/train/wf_smoke_v2 \
  --model twotower --context-backend scratch --steps 16 --batch-size 2 \
  --lr 0.001 --structural-bias 1.5 --seed <N> --device cpu --asap-decode \
  --run-id exp_lever_{no,}curr_s16_lr1e3_bs2_sb15_seed<N> \
  --no-sync-checkpoints [--curriculum]

python -m scripts.evaluate_model \
  --test-dir outputs/data/eval/wf_smoke_v2 --suite smoke \
  --train-dir outputs/data/train/wf_smoke_v2 --model twotower --device cpu \
  --run-id <run>_eval_asap_t30 --checkpoint outputs/runs/<run>/checkpoints/last.pt \
  --grammar-constrained --decode-timeout-seconds 30 \
  --constraint-debt-routing-mode fixed_asap --run-class scratch_matrix --eval-limit 3
```

Baseline seed47 no-curriculum reuses prior champion
`exp_lever_sb15_s16_lr1e3_bs2_seed47` (iso recipe).

## Results

| arm | seed | last_loss | parse | meaningful | reward | empty | timeout | lat_p50_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nocurr | 42 | 8.459 | 1.0 | 0.3333333333333333 | 0.7653333333333334 | 0 | 0 | 20276.82 |
| nocurr | 47 | 7.452 | 1.0 | 0.3333333333333333 | 0.7653333333333334 | 0 | 0 | 1469.53 |
| nocurr | 51 | 7.418 | 1.0 | 0.3333333333333333 | 0.8236666666666667 | 0 | 0 | 24268.74 |
| curr | 42 | 11.433 | 1.0 | 0.3333333333333333 | 0.7653333333333334 | 0 | 0 | 30004.48 |
| curr | 47 | 13.086 | 1.0 | 0.3333333333333333 | 0.7653333333333334 | 0 | 0 | 21680.36 |
| curr | 51 | 14.609 | 1.0 | 0.3333333333333333 | 0.7653333333333334 | 0 | 0 | 19942.79 |

### Multi-seed summary

- no-curriculum success (parse=1 & empty=0): **3/3**
- curriculum success: **3/3**
- mean parse: nocurr=1.000 → curr=1.000
- mean meaningful: nocurr=0.333 → curr=0.333
- mean reward: nocurr=0.785 → curr=0.765

## Decision

**REJECT** curriculum — iso quality (parse/meaningful/empty), **latency regression**
on the champion seed (seed47 p50 ~1.47s → ~21.7s) and higher train loss under curriculum.

Keep **no-curriculum** as the default sampling mode on this micro-recipe.

### Side finding (not the lever under test)

On the **champion recipe itself** (bs=2 · sb=1.5 · lr=1e-3 · s16 · ASAP · t30),
seeds **42 and 51** both reach parse=1 / empty=0 / meaningful≈0.33 — previously
brittle under the older multi-seed recipe (bs=4, default structural bias). That is
a multi-seed robustness win for the champion hyperparameters, **not** for curriculum.

Meaningful-program remains capped at **1/3 on n=3** for every arm. Soft curriculum
does not lift that ceiling on `wf_smoke_v2`.

## Next lever (evidence-backed)

Prefer **richer train data** (non-smoke corpus / mixture / real train dir) to lift
`meaningful_program_rate`, not more sampling-order tweaks on the smoke fixture.

Captured: 2026-07-27T16:04:08.023710+00:00

