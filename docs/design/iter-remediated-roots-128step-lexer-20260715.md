# Lexer-tokenizer root-target comparison — 2026-07-15

This matched comparison trains the source-controlled `remediated_roots` corpus
with the lexer output tokenizer rather than the default compositional tokenizer.

- TwoTower, scratch context, CPU
- 108 records / 94 unique targets
- 128 steps, batch 8, LTR loss weight 2.0
- 49,042 target tokens; telemetry total 20.49 seconds
- Best held-out weighted NLL: 6.719041 at step 128
- Best held-out broad mean NLL: 4.977218

The best checkpoint was evaluated with explicit grammar-constrained decoding,
LTR-primary repair, 64-token LTR cap, one attempt, and smoke `n=3`. Parse rate,
raw syntax validity, structural similarity, component recall, and reward were
all 0.0. Decode timeouts were 0 and p50 latency was 7,590 ms.

The checkpoint is rejected. Lower lexer loss does not translate to valid output;
the next iteration should inspect emitted token/prefix failure evidence and
repair the output objective or decoder contract.
