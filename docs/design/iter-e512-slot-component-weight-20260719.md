# E512 — slot-to-component decode weight

E512 adds a canonical `--slot-component-decode-weight` evaluation override and
tests weight 8 against E510's checkpoint weight 4. Every other OOD setting is
matched: component-plan weight 4, honest slot-contract context and constraint,
length-safe 160-token canvas, default generation controls, no DESIGN context,
and no fallback.

The evaluation completed under its external 170-second cap and emitted
AgentEvals plus pinned AgentV evidence without execution errors.

| Slot weight | n | Meaningful | Fidelity | Structure | Recall | Reward | AST node F1 | AST edge F1 | Spam | Role mismatch | AgentV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 (E510) | 4 | 0.50 | 0.6583 | 0.3446 | 0.3958 | 0.8405 | 0.4679 | 0.1625 | 3 | 4 | 0/1 |
| 8 (E512) | 4 | 0.25 | 0.3417 | 0.2869 | 0.3333 | 0.7245 | 0.3817 | 0.1000 | 1 | 4 | 0/1 |

Weight 8 changes every slot-component-biased choice it touches. It suppresses
placeholder spam but increases missing components and does not change
semantic-role mismatch prevalence. The net result is a regression on every
major quality metric.

## Decision

Reject weight 8 and retain the checkpoint's weight 4. The next improvement must
come from better slot-role supervision and anti-spam calibration during
training, not stronger inference bias. No checkpoint was created or synced.

Exact metrics:
[machine-readable record](iter-e512-slot-component-weight-20260719.json).
