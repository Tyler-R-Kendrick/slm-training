# E83 learned contract decode with 64-token cap — 2026-07-15

E83 disabled the contract-template fast path and evaluated the E82 checkpoint
through learned constrained decoding with `grammar_ltr_max_tokens=64`.

The bounded smoke probe did not parse. It reached 0.75 exact placeholder
fidelity, 0.75 normalized fidelity, and contract precision/recall 0.60/0.75,
but structural similarity was 0.21 and reward 0.0. Latency was 8,158.29 ms
with zero timeout events; the output remained semantically invalid.

Decision: reject cap 64 as a learned-quality setting. The model can emit some
contract slots, but learned constrained decoding still fails structure; the
certified fast path must remain clearly separated from model-quality claims.

This is a bounded scratch diagnostic, not a ship claim.
