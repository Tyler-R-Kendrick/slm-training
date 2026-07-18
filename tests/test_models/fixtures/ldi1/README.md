# LDI1-01 fixture-grade causal decision trace

Deterministic, torch-free evidence for exact causal decision-state capture
(SLM-119 / LDI1-01). Regenerate with the recipe in
`docs/design/iter-ldi1-01-causal-capture-20260718.md`; `test_causal_trace_fixture.py`
loads and checks it.

- `traces.jsonl` — four captured causal decisions (a constraint shadow, a forced
  deduction, an ordinary decision, and the EOS stop) through the shared `TraceStore`.
  Integer prefix ids are the state authority; per-step raw/legal distribution telemetry
  is recorded.
- `causal_trace_manifest.json` — model/tokenizer/adapter/decode identities and state
  counts (bytes/state and duplicate-set reuse). No timestamps, so it is reproducible.
- `forced_counterfactual_outcome.json` — one forced legal action replay outcome handed to
  the counterfactual owner (no judge, no label). `canonical_program` is `null`:
  canonicalization is the strict validator's job and the component catalog is
  environment/pack-dependent, so the fixture does not assert the raw program is valid.

This artifact carries **no semantic label and makes no model-quality claim**; the
constraint-shadow evidence is legality-only and non-trainable by construction.
