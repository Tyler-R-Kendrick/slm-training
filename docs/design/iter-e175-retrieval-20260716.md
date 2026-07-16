# E175 retrieval-conditioned semantic control (2026-07-16)

E175 kept the HF context tower frozen and schema context enabled, adding
retrieval of four nearby training examples. The first invocation stopped at
the checkpoint-bucket guard because no HF auth was present; the local scratch
rerun explicitly used `--no-sync-checkpoints` and completed with telemetry.

| Metric | E173 | E175 |
| --- | ---: | ---: |
| train steps | 32 | 8 |
| final loss | 11.0876 | 27.9708 |
| retrieval k | 0 | 4 |
| bounded syntax parse | 1.0000 | 0.0000 |
| bounded meaningful parse | 0.0000 | 0.0000 |
| structural similarity | 0.1542 | 0.3163 |
| p50 latency (ms) | 3896.5 | 7915.64 |

Retrieval did not recover semantic component selection and worsened syntax on
the bounded probe. It is rejected. The bucket-auth failure is retained as
evidence that local scratch runs must declare `--no-sync-checkpoints`; full HF
runs still require authenticated checkpoint persistence.

Evidence: [result JSON](iter-e175-retrieval-20260716.json), [train summary](../../outputs/runs/e175-retrieval-8step/train_summary.json), [probe eval](../../outputs/runs/e175-retrieval-probe/eval_smoke.json), and the AgentV JSONL path recorded in the result.
