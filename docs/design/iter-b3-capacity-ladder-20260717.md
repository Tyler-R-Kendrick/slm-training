# B3 — Capacity ladder: quality-vs-`d_model`, surface tokens vs choice codec (2026-07-17)

Registration + wiring run, not a ship run. Evidence:
[iter-b3-capacity-ladder-20260717.json](iter-b3-capacity-ladder-20260717.json).
Harness: [`src/slm_training/harnesses/experiments/ladder.py`](../../src/slm_training/harnesses/experiments/ladder.py),
runner [`scripts/run_scaling_ladder.py`](../../scripts/run_scaling_ladder.py). Linear SLM-23.
Feeds the E1 bits-per-semantic-decision study
([iter-e249-semantic-bits-20260716.md](iter-e249-semantic-bits-20260716.md)); pairs the
representation arms of the quality matrix (E255 lexer / E262 choice) across a `d_model` sweep.

## Question

Does removing non-lexical symbols (externalizing the grammar via the B1 choice codec)
let a *smaller* model learn the grammar? E1 measured the corpus-side prediction
(externalizing nearly halves the bits a model must reproduce). B3 sets up the direct
empirical test: quality-vs-`d_model` for surface-token targets vs the choice-sequence
codec, at matched recipe, so `params_per_bit` at matched quality can be compared per
representation.

## What was registered

The B3 capacity ladder reuses the existing from-scratch ladder point construction
(`scratch_ladder_default`: `budget ∝ d_model²`) and pairs two otherwise-identical
arms that differ **only** in `output_tokenizer`. `capacity_ladder_arms()` returns one
`ScalingLadder` per arm; `run_scaling_ladder.py --capacity-arm {lexer,choice}` runs one arm.

Matched recipe (both arms): `mask_pattern=diffusion`, `grammar_ltr_primary=False`,
`context_backend=scratch`, identical widths / depths / token budgets / seed / steps.

| Arm | `output_tokenizer` | ladder id | matched matrix control | rungs (`point_id`) |
| --- | --- | --- | --- | --- |
| A (surface control) | `lexer` | `capacity_lexer_v1` | E255 | `d64_…_t12500`, `d128_…_t50000`, `d192_…_t112500` |
| B (choice codec) | `choice` | `capacity_choice_v1` | E262 | `d64_…_t12500`, `d128_…_t50000`, `d192_…_t112500` |

Row set = **3 `d_model` rungs (64 / 128 / 192) × 2 arms = 6 registered rows**, matched
`point_id` across arms. Pinned by
[`tests/test_harnesses/experiments/test_capacity_ladder.py`](../../tests/test_harnesses/experiments/test_capacity_ladder.py)
(asserts identical rung set and identical non-tokenizer `ModelBuildConfig` fields).

## Semantic-density headline (meaningful at fixture scale)

The honest, scale-independent result is the per-arm target-stream information content —
how many bits each representation asks the model to reproduce over the same programs
(corpus `resources/{train,test}_seeds.jsonl`, n=36, reproduces E1/B1 exactly):

| Target stream (arm) | bits/decision (H) | n decisions | total bits | vs surface |
| --- | ---: | ---: | ---: | ---: |
| surface (arm A, `lexer`) | 5.45 | 1535 | 8368 | 1.00× |
| production | 4.31 | 1019 | 4392 | 1.90× fewer |
| choice (arm B, `choice`) | 4.41 | 842 | 3713 | **2.25× fewer** |

- **surface→choice total-bit ratio = 2.25×** — the choice arm's target stream carries
  2.25× fewer bits than the surface/lexer arm for the same programs.
- **choice decision reduction = 1.82×** — 1.82× fewer tokens per program.

This is a *representation* property (the denominator of `params_per_bit`); it predicts,
but does not prove, that a smaller model suffices — exactly what the trained ladder tests.

## Wiring smoke (wiring-evidence-only)

Both arms execute end-to-end through the new runner path
(`run_scaling_ladder.py --capacity-arm …`), CPU, scratch backend, 24-record train slice,
`--widths 64 --steps 4`:

| Arm | ladder id | weighted NLL | cost (s) |
| --- | --- | --- | ---: |
| choice | `capacity_choice_v1` | inf (untrained, 4 steps) | 7.09 |
| lexer | `capacity_lexer_v1` | inf (untrained, 4 steps) | 2.17 |

Non-finite NLL is the honest untrained result; this proves the ladder registers and
runs both arms, nothing about quality. Parse is a meaningful primary for the choice arm
(the detokenizer is fail-closed) but is not scored at this budget.

## Run metadata

- device: CPU; backend: scratch; matrix set: `capacity_v1`; suite size n=36 (density),
  24-record slice (wiring); honesty mode: wiring-only; ship-gate outcome: **n/a** (wiring).

## Honesty

- No checkpoint created or promoted; **MODEL_CARD not updated** (nothing to record).
- `params_per_bit` at matched quality (the B3 quality headline) is **unrun-at-scale** — it
  needs trained models at each rung on a production budget. The per-arm `total_bits` above
  are its denominators. **No capacity-vs-quality curve is claimed** from the wiring smoke;
  fabricating one from untrained noise is prohibited.
- The semantic-density separation (2.25× fewer target bits for choice) is real and
  scale-independent; it is the only quality-relevant claim this iteration supports.
