# E135 HF context control — 2026-07-15

E135 is the first valid HF-context control after installing the repository's
Transformers extra and downloading the pinned SmolLM2-135M assets. It trains
the 405 judged records for 8 CPU steps with the context backbone frozen.

| Control | Parse | Placeholder validity | Structural similarity | p50 latency |
| --- | ---: | ---: | ---: | ---: |
| Scratch, 3-prompt control | 0.0 | 0.0 | 0.1742 | 3,815 ms |
| HF context, 3-prompt control | 0.0 | 0.3167 | 0.2422 | 17,943 ms |

The HF control improves representation-sensitive signals but is still not
parseable and has one 20-second decode timeout. Training telemetry and a
complete AgentEvals bundle are persisted. A later 32-step attempt stopped at
step 19 without a checkpoint and is intentionally excluded from the result.
This is diagnostic evidence, not a ship or promotion result.
