# E498 — restore E396 slot-component learning on current main

E498 reconciles the learned slot-to-component auxiliary head from the durable
E396 checkpoint with current `main`. It restores the head configuration,
checkpoint state, training loss, corpus-derived class/lexeme/span priors, model
builder and CLI wiring, and legal decode bias. No training or checkpoint write
occurred.

## Reproducibility

- Frozen checkpoint: `e396-balanced-type-head-continuation-r1`
- SHA-256: `feefa0564490bd1db42f79ff710143ad8ed07ab9e4e324f2744a30f8c2f2eee0`
- Bucket: `hf://buckets/TKendrick/OpenUI/checkpoints/e396-balanced-type-head-continuation-r1/`
- Device/backend: CPU / frozen local HF context
- Evaluation: SDE0-01 Stage A, `smoke`, `n=3`, choice codec
- Process envelope: interrupt at 170 seconds, force-kill ten seconds later
- AgentV: each arm emitted AgentEvals JSONL and ran through the pinned SDK; the
  baseline bundles are linked in the adjacent JSON record.

## Measured result

| Current-main condition | Parse | Meaningful | Structure | Component recall | Fidelity | Reward | Head applications / changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Head loads, structural choices misclassified | 1.0 | 0.0 | 0.17197 | 0.0 | 0.11111 | 0.0 | 0 / 0 |
| Structural bound choices receive learned bias | 1.0 | 0.0 | 0.27057 | 0.0 | 0.0 | 0.0 | 20 / 20 |

The trace found the remaining compatibility defect: choice-tokenizer
`structural` states were classified as `component_root_or_bound`, while the
slot head intentionally scores only legal `component_bound` choices. Treating
both `v05` and `structural` states as bound activates the frozen learned head.
Structural similarity improves by `+0.09860`, and all 20 applications changed
the selected component.

This is still a negative ship result. Meaningful-program rate, component recall,
and reward remain zero, placeholder fidelity regresses to zero, only smoke ran,
and AgentV remains red. E396 is load-compatible current-main diagnostic evidence,
not a champion, promotion, or production model. Stage B is not recommended.

Canonical metrics: [JSON](iter-e498-current-main-slot-component-restore-20260718.json).
