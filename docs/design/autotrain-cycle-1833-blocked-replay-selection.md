# Autotrain c1833: blocked replay selection

**Verdict:** no model result. c1833 stopped before hypothesis compilation or arm
execution because frozen-replay discovery selected c1832's evidence-bound
`blocked` retry action even though predecessor gating had already accepted the
receipt and advanced.

| Stage | Status | Signal |
| --- | --- | --- |
| Latest / merge | pass | upstream `9dcfa7e6...5bd0` was an ancestor of integration `3eae7be4...2bec` |
| Campaign init / research | pass | evidence snapshot captured |
| Blocked-action filtering | fail | invalid c1832 promotion-endpoint manifest selected for automatic replay |
| Train / eval / AgentV | not run | no model or metric evidence |
| Lean / formal | `not_applicable:pre_execution` | promotion was not reached |

The contradiction was between two readers of the same append-only receipt
ledger: prerequisite gating treated the blocked action as handled, while
execution discovery considered only `completed` receipts terminal. Campaign
harness v130 makes evidence-bound `blocked` receipts terminal for steering
discovery, while prerequisite repair/document gates still require `completed`.
This preserves the reason and evidence for the blocked action without executing
an invalid frozen manifest.

Next: reopen c1832's incomplete champion and run a new-seed smoke confirmation
under the v129 confirmation boundary. Machine evidence:
[`autotrain-cycle-1833-blocked-replay-selection.json`](autotrain-cycle-1833-blocked-replay-selection.json).
