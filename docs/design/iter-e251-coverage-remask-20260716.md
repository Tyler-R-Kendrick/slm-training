# E251 — A3 coverage-energy guided remasking (2026-07-16)

Decode-lever wiring, not a train/ship run. Code:
[`models/parallel_decode.py`](../../src/slm_training/models/parallel_decode.py)
(`select_remask_coverage_indices`),
[`models/twotower.py`](../../src/slm_training/models/twotower.py)
(`remask_policy="coverage"`, `_coverage_deficit`). Linear SLM-39.

## What and why

A4 (E250) makes the empty program *illegal* at the constraint layer. A3 is the
soft sibling at the sampling layer: instead of forbidding emptiness, it biases
**which committed positions get remasked** toward filler regions when the layout
is content-sparse, giving the denoiser another pass to place missing inventory
content. Softer levers avoid A4's hallucination risk when the model genuinely has
no signal; the two are compared on the same v9 rows.

## Mechanism

- `select_remask_coverage_indices(conf, known, *, remask_ratio, protect_bos,
  coverage_deficit, grammar_positions)` mirrors `select_remask_core_indices`:
  fill grammar positions first, then spend the remask budget on the highest
  per-position `coverage_deficit` (tiebreak: lowest confidence), falling back to
  confidence remasking when no deficit tensor is supplied.
- `TwoTowerModel._coverage_deficit(ids, known)` computes the deficit self-
  containedly from the tokenizer's compiler-derived symbol spaces
  (`component`/`sym`/`bind` via `kind_ids`) — no slot contract needed. A
  non-content position scores `1 - content_fraction` for its row; content
  positions score 0. Content-sparse rows push filler positions high; dense rows
  drive the deficit to ~0 and the policy degrades to ordinary confidence
  remasking. Tokenizers without `kind_ids` (the compositional default) yield a
  zero deficit and the same graceful fallback.

Selected with the existing `remask_policy` knob (`remask_policy="coverage"`) — no
new config field; needs no extra model forward, so it works even when successor
speculation forbids them.

## Verification

- Pure-function tests (`tests/test_harnesses/model_build/test_a3_coverage_remask.py`):
  prefers high-deficit positions, confidence fallback when deficit is None, empty
  at ratio 0, budget + BOS protection, highest-deficit-first ordering.
- Model integration: `_coverage_deficit` scores the component (content) position
  0 and structural filler positions > 0 on a lexer-tokenizer model.
- Regression: `test_v6_remask` + `test_v7_speculative` + `test_compiler_decode`
  green (58 passed).

## Honesty

Wiring + unit evidence only. No checkpoint, no suite scoreboard, no ship claim.
Whether coverage remasking lifts meaningful parse (and how it trades off against
A4's hard contract) is the measurement the v9 E-rows enable on a GPU host with a
compiler-tree checkpoint — not claimed here.
