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

## Five-minute matched run — attempt 1 failed

The first bounded lexer control used the committed 480-record
`e218_schema_normalized_judge_v5` corpus, width 64, seed 0, batch 2, a
5,000-target-token budget, a 200-step ceiling, and all five remediated eval
suites (n=19). It reached step 20 / 1,944 target tokens with logged loss
33.297, then failed while publishing the first loss-suite AgentV bundle.

Root cause: the Python process wrote the AgentEvals JSONL relative to its
isolated worktree, but the pinned AgentV SDK ran from the Git common checkout
and resolved the same relative path there. The JSONL exists; the SDK result,
train telemetry summary, checkpoint, and evaluation scoreboard do not. This is
a failed infrastructure attempt, not model evidence.

The shared AgentV publisher now resolves its artifact root to an absolute path
before invoking the SDK. The matched run must restart from scratch after a new
latest-main preflight; no missing checkpoint is inferred or reconstructed.

Live machine-readable ledger:
[iter-b3-capacity-ladder-5m-results-20260717.json](iter-b3-capacity-ladder-5m-results-20260717.json).

## Five-minute matched run — lexer control complete

The restarted lexer control completed the matched 5,000-target-token recipe in
53 steps (5,004 tokens) and 165.28 seconds, below the configured five-minute
whole-arm cap. It used CPU scratch context, width 64, seed 0, batch 2, the 480
committed judge-approved E218 records, and all five remediated suites.

| suite | n | parse | meaningful | fidelity | structure | reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0 | 0 | 0 | 0.0125 | 0 |
| held_out | 5 | 0 | 0 | 0 | 0.1166 | 0 |
| adversarial | 4 | 0 | 0 | 0 | 0.0346 | 0 |
| ood | 4 | 0 | 0 | 0 | 0.0833 | 0 |
| rico_held | 3 | 0 | 0 | 0 | 0.2528 | 0 |

Weighted NLL fell to 13.1800, but it is not a substitute for the zero primary
quality metrics. AgentV passed 0/5 rows. The local scratch checkpoint is
therefore **not promotable and not ship**.

The run also exposed a fail-open promotion policy: a single-rung ladder with
only a passing integrity check was labeled promotable. The falsely named
`promoted.pt` / `promoted.json` outputs were removed. Promotion now requires at
least one quality, rank-stability, efficiency, or ship-gate evidence channel;
integrity alone cannot promote.

Telemetry identifies evaluation, not training, as the immediate throughput
bottleneck: the final five-suite evaluation consumed 92.50% of measured time,
loss suites 5.21%, and forward plus backward only 1.02%. This does not justify
reducing suite coverage in the matched pair; both arms retain the same honest
five-suite evaluation.

## Five-minute matched run — choice arm complete

The choice arm completed the same 5,000-target-token recipe in 107 steps
(5,022 tokens) and 19.94 seconds. It has 308,554 trainable parameters versus
294,666 for lexer because the output vocabularies differ. The corpus choice
stream contains 37,802 bits across 8,795 decisions, versus 84,904 bits across
15,780 lexer decisions: **2.246× fewer total bits** and **1.794× fewer
decisions** for the same 480 programs.

All five choice suites nevertheless scored parse, meaningful-program,
fidelity, structure, and reward at 0. AgentV passed 0/5 rows. Every one of the
19 predictions was the empty string and failed with `parser produced no root
element`. The final weighted NLL was 7.0985, but its category inventory was
incomplete (`binding` absent), and NLL values in different tokenizer spaces
are not directly comparable.

| matched result | lexer | choice |
| --- | ---: | ---: |
| target tokens consumed | 5,004 | 5,022 |
| train steps | 53 | 107 |
| whole-arm wall time | 165.28 s | 19.94 s |
| parse / meaningful / fidelity | 0 / 0 / 0 | 0 / 0 / 0 |
| AgentV | 0/5 | 0/5 |
| promoted | no | no |

**B3 verdict: no quality winner.** Both primary outcomes are tied at zero.
Choice is 8.29× faster in this single matched run and has the predicted
information-density advantage, but neither fact establishes quality.

The empty choice predictions are a deterministic-layer defect signal, not an
undertraining diagnosis. Choice decoding currently bypasses the surface DFA
and relies only on a fail-closed final detokenizer; an early EOS therefore
produces an empty program. The next iteration must add a choice-native legal
decision state derived from the production codec, including forced selection
when only one token is legal, then reevaluate this frozen checkpoint without
retraining.
