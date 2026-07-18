# E442 E396 full five-suite ship gates — 2026-07-18

E442 combines the policy-identical E398 bounded suites with E441's canonical
full-RICO aggregate for checkpoint
`e396-balanced-type-head-continuation-r1` (SHA
`feefa0564490bd1db42f79ff710143ad8ed07ab9e4e324f2744a30f8c2f2eee0`).
All source evaluations are complete, non-diagnostic suites. The merge command
completed normally in 1.8 seconds under the external 290-second cap.

Recipe: CPU frozen SmolLM2 context; local files only; 320-token grammar LTR;
component-plan decode weight 2; slot-component weight 8; honest constrained
slot contract; eight generation steps; three attempts; no DESIGN.md context.
The checkpoint was trained locally for 427 cumulative steps / 22,044 target
tokens and carries an explicit no-sync scratch reason.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5600 | 0.5000 | 0.9770 |
| held_out | 5 | 1.0 | 0.6000 | 1.0 | 0.5933 | 0.4833 | 0.5916 |
| adversarial | 4 | 1.0 | 0.7500 | 1.0 | 0.6762 | 0.7500 | 0.7268 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.5511 | 0.7292 | 0.9827 |
| rico_held | 1500 | 1.0 | 0.9847 | 0.9993 | 0.6390 | 0.8652 | 0.9827 |

The authoritative `--ship-gates` result passes every required threshold with
no failures. AgentV passes 5/5 with zero execution errors. The canonical
artifacts are under `outputs/runs/e442-e396-full-ship-gates-r1/`.

**Verdict:** promote E396 as the local ship-gate champion. This is not yet a
production HF ship: the checkpoint remains local and must be synced to the
OpenUI HF bucket, then registered with its durable URI, before making that
claim.
