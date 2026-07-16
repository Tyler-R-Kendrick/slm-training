# E195–E199 — stratified grammar-decision alignment

Status: **diagnostic only; no checkpoint promoted; ship gates not run**.

## Hypothesis

E191 sampled one arbitrary compiler branch per row and regressed root selection.
E195–E196 instead classify gold decision positions from the tokenizer and Lark
completion forest, then sample one position per decision kind. This guarantees
that component and binder supervision are present without matching prompt text,
specific components, or known failing outputs.

## Recipe and invalid control

Both trains used CPU, 32 steps, batch 4, seed 0, frozen SmolLM2-135M context,
lexer output, schema/slot context, no DESIGN.md context, alignment weight 1.0,
and no checkpoint sync. E195 is **not a valid comparison**: `--train-version`
selected the committed corpus but silently left the online mixture unset. It
therefore consumed a different sample (23,381 prompt / 6,050 target tokens).
This found and fixed a pipeline defect: published versions now automatically
load their canonical `mixture.json` unless an experiment explicitly supplies
another committed manifest.

E196 reran with the E181 balanced manifest explicitly selected. Its corpus hash,
mixture hash, 17,848 prompt tokens, and 5,928 target tokens match E191.

| Train | Valid control | Last loss | Wall s | Aligned rows | Component | Binder | Structural | Symbol | Literal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E195 | No — missing mixture | 17.0750 | 174.22 | 597 | 128 | 128 | 128 | 128 | 85 |
| E196 | Yes | 7.8562 | 110.93 | 631 | 128 | 128 | 128 | 128 | 119 |

E196 alignment loss fell from 72.1903 to 2.4683 (mean 14.1408). The expanded
canvases are evaluated in one denoiser batch, so stratification does not add a
separate forward for each concrete kind.

## Evaluations

All evaluations are strict one-example smoke diagnostics with no unconstrained
fallback. Each wrote AgentEvals JSONL and an AgentV SDK bundle. Full paths,
checkpoint hashes, and recipes are in
[the result JSON](iter-e195-e199-stratified-alignment-20260716.json).

| Experiment | Generalized change | Syntax | Meaningful parse | Structure | Component recall | p50 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E197 | Stratified E196 checkpoint | 0.0 | 0.0 | 0.1917 | 0.0 | 3643.38 |
| E198 | EOS also requires resolved binder graph | 0.0 | 0.0 | 0.1917 | 0.0 | 3572.64 |
| E199 | Enum restriction follows parser slot progress | 1.0 | 0.0 | 0.1917 | 0.0 | 4304.95 |

E197 already ranked `Stack` and the first forward binder correctly, but stopped
inside the root call. E198 showed the stop happened before EOS eligibility. The
generated enum policy remained active after its string value had been consumed,
removing the grammar-valid closing delimiter. E199 ends enum restriction when
the parser reports that the slot has started. Syntax returns to 1.0 with zero
compiler fallbacks.

The E199 program is still trivial: it assigns the referenced child binder a
legal primitive string instead of an element. Therefore meaningful parse,
component recall, and reward remain zero. This is a negative ship result, not a
promotion.

## Next hypothesis

Propagate the value role attached to each generated binder reference into the
binder declaration. A reference used in an element-valued schema/AST position
must constrain its declaration to a component expression. The implementation
must derive that role from parser/schema state and typed binder identity; it must
not enumerate prompt strings, component arrangements, or observed failures.
