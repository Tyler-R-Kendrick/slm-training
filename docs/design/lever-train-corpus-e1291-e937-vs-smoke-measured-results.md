# Train-corpus lever: e1291 / e937 vs wf_smoke_v2 — NOT SHIP

**Honesty:** `fixture_or_scratch` / smoke suite `n=3`. **Not a ship claim.**

## Hypothesis

At the frozen champion micro-recipe (s16 · lr=1e-3 · bs=2 · structural_bias=1.5 ·
seed=47 · ASAP · t30 · scratch/cpu), training on **richer contract-valid corpora**
(e1291 document-only n=350, e937 role-safe n=524) lifts `meaningful_program_rate`
on the smoke eval vs the `wf_smoke_v2` baseline (n≈103).

## Recipe

```bash
python -m scripts.train_model \
  --train-dir <corpus> --model twotower --context-backend scratch \
  --steps 16 --batch-size 2 --lr 0.001 --structural-bias 1.5 --seed 47 \
  --device cpu --asap-decode --run-id exp_lever_data_<tag>_... --no-sync-checkpoints

python -m scripts.evaluate_model \
  --test-dir outputs/data/eval/wf_smoke_v2 --suite smoke \
  --checkpoint outputs/runs/<run>/checkpoints/last.pt \
  --grammar-constrained --decode-timeout-seconds 30 \
  --constraint-debt-routing-mode fixed_asap --run-class scratch_matrix --eval-limit 3
```

Eval suite is held fixed (smoke) so the only arm change is train corpus.

## Results

| arm | corpus_n | last_loss | parse | meaningful | reward | empty | timeout | lat_p50_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wf_smoke_v2 | 103 | 7.452 | 1.0 | 0.3333333333333333 | 0.7653333333333334 | 0 | 0 | 1469.53 |
| e1291_document_only | 350 | 22.439 | 1.0 | 0.3333333333333333 | 0.7653333333333334 | 0 | 0 | 23356.02 |
| e937_role_safe | 524 | 16.569 | 0.6666666666666666 | 0.0 | 0.5296666666666666 | 1 | 1 | 27683.62 |

### Blocked candidates (current contracts)

- `e230_diverse_judged_roots_v2` — symbol_only/v2 free-form strings
- `e826_target_slots_only_v4` — placeholder in non-content property
- `scope_graded_v1` — symbol_only/v2 free-form strings

## Decision

**REJECT** — no corpus beat wf_smoke_v2 baseline on meaningful/parse under champion micro-recipe; e937_role_safe regressed quality (parse=0.6666666666666666, empty=1)

Larger corpus size alone (350/524 vs 103) does **not** lift the meaningful-program
ceiling on this smoke scoreboard under a 16-step micro recipe. Keep `wf_smoke_v2`
as the micro-train default until a corpus improves metrics or a longer honest train
is run with matching eval suite.

## Next lever (evidence-backed)

1. **Build/refresh a contract-valid larger train set** via `slm data build-train` (current
   curated larger sets fail symbol-only / role-safe gates), then re-test corpus lever.
2. Or **increase steps with wall budget** on a corpus that already matches eval domain
   once meaningful headroom is measured on a larger suite than n=3.

Captured: 2026-07-27T16:12:17.255969+00:00

