# E124 decoder isolation — 2026-07-15

E124 holds the E123 checkpoint and prompt fixed while comparing constrained and
unconstrained generation.

| Variant | Parse | Structural similarity | Reward | p50 latency | Fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constrained | 0.0 | 0.1917 | 0.0 | 24,902 ms | 1.0 |
| Unconstrained | 0.0 | 0.25 | 0.0 | 836 ms | 0.0 |

The unconstrained path is much faster but still produces invalid OpenUI, so
grammar is not the root cause of the quality failure. The constrained path
adds roughly 24 seconds, emits 115 tokens, and performs 2,107 DFA sync/probe
operations before falling back. This identifies independent workstreams:
improve the learned distribution/training data, and reduce candidate-admission
cost without relaxing legality.

Both runs emitted AgentEvals JSONL and scoreboards. This is a one-example
diagnostic, not a ship evaluation.
