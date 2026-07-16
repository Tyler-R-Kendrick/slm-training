# E176 broad-corpus semantic control (2026-07-16)

E176 trained on the broader 1,417-record `v2_prompt_contract` artifact instead
of the 498-record judged root corpus. The decoder and compiler settings stayed
matched to E175.

| Metric | E175 judged corpus | E176 broad corpus |
| --- | ---: | ---: |
| train records | 498 | 1417 |
| train steps | 8 | 8 |
| final loss | 27.9708 | 34.0464 |
| bounded syntax parse | 0.0000 | 0.0000 |
| bounded meaningful parse | 0.0000 | 0.0000 |
| structural similarity | 0.3163 | 0.1187 |
| p50 latency (ms) | 7915.64 | 21276.93 |

The larger corpus does not improve semantic selection and harms structure and
latency in this matched short control. Retain the judged corpus as the base;
the next data change should be a targeted, judge-gated semantic-role variant,
not an unfiltered corpus expansion.

Evidence: [result JSON](iter-e176-broad-corpus-20260716.json), [train summary](../../outputs/runs/e176-broad-corpus-8step/train_summary.json), [probe eval](../../outputs/runs/e176-broad-corpus-probe/eval_smoke.json), and the AgentV JSONL path recorded in the result.
