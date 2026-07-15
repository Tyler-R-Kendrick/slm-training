# Iteration: direct DFA probe benchmark (2026-07-15)

Direct checkpoint-tokenizer profiling measured 64 incremental Lark
`probe_chunk` calls at **15.7–17.7 ms** after prefixes of length 0, 31, and
57. The tokenizer vocabulary is 699 tokens. A full-vocabulary candidate scan
per decode position is therefore expensive enough to explain the constrained
evaluation timeout.

The standalone evaluator now exposes `--grammar-top-k`; a one-record,
one-step, one-attempt probe with `grammar_top_k=1`, stream probes skipped, and
structural trust still exceeded the execution window. Candidate breadth is a
real cost, but not the only remaining path. No quality or ship claim is made.
