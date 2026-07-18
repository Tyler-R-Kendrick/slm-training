# E334 full local RICO validation — 2026-07-17

E334 builds the canonical full local evaluation set from the cached Hugging
Face Rico test split using E316's train manifest for leakage filtering. The
builder keeps 1,500 RICO records, rejects 158 overlaps, and reports zero
conversion errors. Manifest SHA:
`6b02e7544782bf51f5d64dc1073c9d920446dd04f97fe25a23e9f5728c7c6983`;
RICO records SHA:
`55e58123b988574a0fbc8804ad3ff8feaa7f78b05a7495c28f5d8208708747cf`.

The unchanged E333 checkpoint is evaluated under the same honest compiler-tree
policy with no suite cap.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | 3 | 1.0 | 1.0000 | 0.6281 | 0.6667 | 0.5000 | 0.6407 | Pass |
| held_out | 5 | 1.0 | 1.0000 | 0.5443 | 0.4000 | 0.3000 | 0.3868 | Pass |
| adversarial | 4 | 1.0 | 1.0000 | 0.6874 | 1.0000 | 0.7500 | 0.9700 | Pass |
| ood | 4 | 1.0 | 1.0000 | 0.6662 | 0.7500 | 0.5625 | 0.7425 | Pass |
| full `rico_held` | 1,500 | 1.0 | 0.6490 | 0.4582 | 0.9327 | 0.5148 | 0.8271 | Pass |

Full-RICO latency is 448.69 ms p50 / 948.52 ms p95. AgentV passes 5/5 with no
execution errors and all current gates pass.

**Verdict:** E333 clears the full local RICO bar and remains the scratch
champion. Do not claim production ship: its context backend is scratch, the
checkpoint is local with explicit no-sync, and no full HF-context train has
uploaded a durable checkpoint to the bucket.
