# E138 HF context seed-1 8-step control — 2026-07-15

E138 repeats E135's best observed 8-step HF recipe with seed 1. The
checkpoint completed with loss `32.5158`; training telemetry and the local
checkpoint hash are persisted in the companion JSON.

| Checkpoint | Seed | Steps | Parse | Placeholder validity | Structural similarity | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E135 | 0 | 8 | 0.0 | 0.3167 | 0.2422 | 17,943 ms |
| E138 | 1 | 8 | 0.0 | 0.0000 | 0.1683 | 12,491 ms |

The seed change worsened both diagnostic signals; neither run parses or earns
reward. This is evidence against selecting a checkpoint from a single seed,
not evidence that the judged corpus or loss weights should be changed yet.
The next controlled step is a small multi-seed checkpoint-selection run.

Feedback used the canonical remediated test manifest, three smoke records,
local-only HF weights, constrained decode, one attempt, and a 20-second
per-record timeout. It produced an AgentEvals JSONL bundle with zero decode
timeouts.
