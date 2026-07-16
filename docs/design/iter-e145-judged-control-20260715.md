# E145 — Judged-corpus control train (2026-07-15)

## Question

Does a fresh short control train on the persisted `remediated_roots_judged` corpus change the constrained-decoder failure observed in E142–E144?

## Recipe

The run used 405 source-controlled records, the frozen HF context model `HuggingFaceTB/SmolLM2-135M`, CPU, compositional output tokens, batch size 4, seed 0, and 8 steps. It kept LTR primary/repair, schema context, slot-contract context/decoding, and no DESIGN.md context. Checkpoint and training telemetry are under `outputs/runs/iter-e145-judged-control-20260715/e145-judged-control/`.

## Results

| Measurement | Result |
| --- | ---: |
| last training loss | 39.0902 |
| seen prompt / target tokens | 5,457 / 1,880 |
| trainable parameters | 1,262,466 |
| training telemetry | 12,483.4 ms |
| context encode share | 46.64% |
| forward share | 48.78% |
| smoke n | 1 |
| parse rate | 0.0 |
| structural similarity | 0.0 |
| placeholder validity | 0.0 |
| timeout count | 1 |
| p50 latency | 20,000.5 ms |
| decoder tokens captured | 234 |
| bounded selection events | 64 |
| AgentEvals | 0/5 passed |

The short control does not improve the failure. The evaluator still times out and produces an unparsable result, but partial decoder telemetry and the AgentEvals bundle are durable. The first selection remains an unmeasured early return (`legal_candidates=-1`), not evidence of a singleton legal set.

## Decision

Do not promote this checkpoint or claim a data-quality improvement. The next data iteration should add semantic pairing checks only where the source metadata provides an expected transformation (edit/repair), then rebuild and compare against this control with the same policy.
