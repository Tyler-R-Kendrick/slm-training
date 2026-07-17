# B3 — Capacity-ladder representation axis (wiring + corpus bits) (2026-07-17)

Harness wiring + measurement, not a train/ship run. Evidence:
[iter-b3-capacity-ladder-wiring-20260717.json](iter-b3-capacity-ladder-wiring-20260717.json).
Code: [`harnesses/experiments/ladder.py`](../../src/slm_training/harnesses/experiments/ladder.py),
[`evals/semantic_bits.py`](../../src/slm_training/evals/semantic_bits.py),
[`scripts/run_scaling_ladder.py`](../../scripts/run_scaling_ladder.py).
Linear SLM-23.

## What landed

B3's empirical question — does removing non-lexical syntax let a smaller model
learn the grammar — needs matched quality-vs-`d_model` curves per output
representation. This iteration wires the axis; it does not run the study.

- **Ladder representation axis.** `LadderPoint.representation`
  (`compositional | lexer | choice`, default `compositional` so legacy point
  ids and behavior are unchanged; non-default ids gain an `_r{repr}` suffix).
  `capacity_ladder_pair()` builds matched arms — same widths, budgets, and
  frozen decode, differing only in representation. The point's representation
  is threaded into `ModelBuildConfig.output_tokenizer`.
- **Fail-closed for the untrainable arm.** The trainer supports
  `compositional`/`lexer` today. A `choice` point is constructible for
  planning and bit accounting, but `model_build_config_for_point` raises
  (`TRAINABLE_REPRESENTATIONS`) rather than silently training the wrong
  representation — `output_tokenizer="choice"` would otherwise fall through
  the twotower's lexer check into the compositional path. The blocker is a
  production-token output head in the twotower (follow-on to B1, SLM-42).
- **`params_per_bit` in the ladder summary** (E1's headline for B3):
  `run_scaling_ladder.py --representation …` computes the E1 corpus bits for
  the arm's stream and reports `trainable_params / total_bits` per trained row.
- **`choice` stream in `semantic_bits`** (production minus grammar-forced
  framing, via B1's `to_choice_stream`), and `compare_representations` now
  reports all three streams.

## Fixture run (wiring evidence, not a result)

`--track scratch --representation lexer --widths 64 --steps 50` on the
published `e230_diverse_judged_roots_v2` corpus (126 records), CPU: one row,
254,466 trainable params, `params_per_bit = 9.32` against the corpus's 27,310
surface bits. The loss (26.05 at 50 steps) is meaningless as quality — this run
exists to prove the axis flows end-to-end (`point_id …_rlexer`,
`output_tokenizer=lexer` in the run's track metadata, populated
`params_per_bit_rows`).

## Honest findings from the bit accounting

- **The choice representation cannot encode 61/126 of the e230 corpus** — the
  rootless single-statement `lc_*` language-contract records (P2 family) have
  no `root` binding, which the document-level production codec requires.
  `semantic_bits` now degrades per-record like the surface stream (skip visible
  via `n_scored_programs`, never silent), and `compare_representations` nulls
  cross-stream ratios when scored subsets differ (`scored_programs_mismatch`)
  instead of comparing different corpora. Before the choice arm can train, B1
  needs a rootless/fragment encoding path for this family — a concrete,
  previously invisible prerequisite this wiring surfaced.
- **On the eval corpus (19/19 parse in all streams):** surface→choice
  total-bit ratio **2.23×** vs E249's surface→production **1.88×** — the B1
  framing elision removes a further ~15% of total bits. Per-decision entropy
  moves the other way (4.44 vs 4.32 bits): the elided framing tokens were
  low-entropy, so the total shrinks while the per-token average rises. Both
  numbers matter: total bits is what capacity must cover; bits-per-decision is
  the per-step prediction difficulty.

## Deferred

The actual capacity study — quality-vs-`d_model` curves per representation with
meaningful parse primary, per `running-experiment-matrices` — needs (1) the
twotower production-token output head so the choice arm trains, and (2) GPU-
scale budgets. The `lexer` arm can run today via
`run_scaling_ladder.py --representation lexer`; the harness will refuse the
choice arm until its trainer support lands.
