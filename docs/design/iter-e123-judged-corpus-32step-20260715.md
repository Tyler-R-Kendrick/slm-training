# E123 judged-corpus 32-step iteration — 2026-07-15

E123 tests whether E121's zero-quality smoke result was primarily caused by
under-training. It trained the same 405 independently judged records for 32
CPU scratch steps and ran the same bounded one-record smoke feedback.

| Suite | n | Parse | Structural similarity | Reward | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 1 | 0.0 | 0.1917 | 0.0 | 26,750 ms |

Training loss reached **10.97** (E121: 92.66), but generation did not improve.
The run emitted complete training telemetry and an AgentEvals JSONL bundle. It
used one unconstrained retry, reached the 256-token canvas cap, and recorded
`constrained_fallback_rate=1.0`. This is a negative diagnostic result, not a
ship evaluation. More scratch steps alone are not currently justified; the
next iteration should isolate generation-recipe/canvas-cap behavior before
changing the data or decoder.
