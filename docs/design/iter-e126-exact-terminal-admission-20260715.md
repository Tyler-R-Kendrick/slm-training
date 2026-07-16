# E126 exact-terminal admission — 2026-07-15

E126 changed candidate selection so an exact DFA terminal set is treated as a
complete legal inventory. Top-k logits are still added for broad/compositional
states, but are no longer added to exact sets where they only create redundant
admission probes.

The profile improved from **12.34s to 6.16s** per generation. The bounded
smoke feedback remained negative: parse 0.0, raw syntax validity 0.0,
structural similarity 0.3833, reward 0.0, and 13.91s p50. The constrained
path still fell back, so this is a decoder-cost improvement, not a quality or
ship result. AgentEvals JSONL and the scoreboard are persisted with the run.
