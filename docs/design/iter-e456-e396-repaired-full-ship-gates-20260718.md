# E456 E396 repaired full five-suite ship gates — 2026-07-18

E456 combines E455's four complete bounded suites with E454's canonical
repaired full-RICO aggregate for unchanged checkpoint E396 (SHA
`feefa0564490bd1db42f79ff710143ad8ed07ab9e4e324f2744a30f8c2f2eee0`).
All source evaluations are complete and non-diagnostic. The fail-closed merger
verified identical checkpoint and evaluation policy and completed normally in
1.8 seconds under the external 290-second cap.

Recipe: CPU, local HF context, 320-token grammar LTR, automatic content floor,
component-plan weight 2, slot-component weight 8, prompt-role constrained
decode, honest constrained slot contracts, eight generation steps, three
attempts, and no unconstrained fallback.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5600 | 0.5000 | 0.9770 |
| held_out | 5 | 1.0 | 0.6000 | 1.0 | 0.6400 | 0.5048 | 0.5922 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.7661 | 1.0 | 0.9760 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.5835 | 0.8125 | 0.9835 |
| rico_held | 1500 | 1.0 | 1.0 | 1.0 | 0.8683 | 0.9960 | 0.9940 |

The authoritative `--ship-gates` result passes every required threshold with
no failures. AgentV passes 5/5 with zero execution errors. Canonical artifacts
are under `outputs/runs/e456-e396-repaired-full-ship-gates-r1/`.

**Verdict:** retain E396 as the local ship-gate champion with stronger
repaired-corpus evidence. This is not a production HF ship: the checkpoint
remains local and requires durable bucket sync and URI registration first.
