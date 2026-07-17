# E250 — A4 minimum-content decode contract (2026-07-16)

Decode-lever wiring, not a train/ship run. Code:
[`dsl/grammar/fastpath/compiler_draft.py`](../../src/slm_training/dsl/grammar/fastpath/compiler_draft.py)
(`build_completion_forest(..., min_content=)`, `emitted_component_count`),
[`models/twotower.py`](../../src/slm_training/models/twotower.py)
(`decode_min_content`, `_effective_min_content`). Linear SLM-40.

## What and why

A1 (E248) framed the valid-but-empty wall as partly a decode-time preference for
the trivial program. A4 closes the loophole at the constraint layer: the
compiler completion forest already proves *legality*; this extends it to prove
*minimum content*. When the prompt names components, an empty/underfull but
grammatically complete layout is no longer a legal completion.

## Mechanism

`build_completion_forest` admits EOS only when
`"$END" in terminals and ast_complete and references_resolved`. A4 adds a
`min_content` gate at exactly that chokepoint:

- `emitted_component_count(tokenizer, prefix_ids)` counts component
  instantiations via the tokenizer's compiler-derived `component` symbol space
  (no AST parse, no Node bridge);
- when `min_content > 0` and the count is below it, EOS is withheld **only if a
  non-EOS continuation remains** — the gate forces the decoder to add content
  and never creates a dead end that would spuriously trigger a fallback.

Config knob `decode_min_content` (compiler-tree decode only):
`0` off · `>0` fixed floor · `-1` auto = distinct slot-contract roots
(`_effective_min_content`). Threaded `TwoTowerConfig` → `ModelBuildConfig` →
`factory._twotower_config_from_build` → the forest call site, so any matrix row
can set it.

## Risk (recorded)

A hard content floor can force a hallucinated component when the model has no
signal. This is why the floor is conservative (auto = distinct prompt slots, not
a fixed large N) and why the softer A3 coverage-energy remask is a sibling lever:
A4 makes empty *illegal*, A3 makes content *preferred*. The E-row comparison (vs
the A5 V9 baseline, `--matrix v9`) will show whether A4 lifts meaningful parse or
merely trades emptiness for hallucination — that is the measurement A4 enables,
not a result claimed here.

## Verification

- Forest-level unit tests (`tests/test_models/test_compiler_decode.py`):
  `emitted_component_count` == 1 for a one-component document; EOS admitted at
  floor 0 and 1, withheld at floor 2 with a non-EOS continuation still present.
- `_effective_min_content` resolves auto/-1 from distinct slot roots, fixed >0,
  and 0-off.
- End-to-end config threading verified: `ModelBuildConfig(decode_min_content=-1)`
  → `TwoTowerConfig.decode_min_content == -1`.
- Full `test_compiler_decode.py` + `test_lattice_search.py` green (38 passed).

## Honesty

Wiring + unit evidence only. No checkpoint, no suite scoreboard, no ship claim.
The behavioral effect on meaningful parse requires a compiler-tree checkpoint run
with `decode_min_content` set, on a GPU host with the frontier checkpoints.
