# Remediated cap-3 corpus feedback — 2026-07-15

This candidate derives from the published `remediated` corpus and caps exposure at three records per root parent. It is persisted as `remediated_cap3` for reproducible future runs and appears in the web app's Training Data inventory. It contains 324 records and manifest fingerprint `d94d09bef3b6bfbffd5f8bc34e78c4f25b558bdda7dc56e09b3efb68188b2bcc`.

## Recipe

- scratch context, compositional output tokenizer
- 64 steps, batch size 8, seed 0
- random masking, LTR loss weight 4.0, fidelity loss weight 0.5
- honest smoke feedback, n=3, AgentEvals bundle emitted

## Result

| Candidate | NLL step 64 | Parse | Structural | Component recall | Placeholder validity | Reward | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `iter-published-remediated-cap3-64step-ltr4-20260715` | 8.1539 | 0.0000 | 0.1597 | 0.2500 | 0.0889 | 0.0000 | Reject |

All three smoke predictions failed to parse. The cap-3 data exposure change and the ltr4 loss weight did not improve the failure mode; the published `remediated` control remains the current training candidate. The full train telemetry, smoke scoreboard, and AgentEvals JSONL remain under the corresponding `outputs/runs/` directories locally.

The cap-3 corpus is retained as a rejected, source-controlled future-run candidate—not promoted as a model or ship result.
