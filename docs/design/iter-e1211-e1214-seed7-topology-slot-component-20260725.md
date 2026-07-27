# E1211-E1214 — seed-7 topology dose and slot-component replication

E1211 restores the E1181 seed-7 CPU-scratch TwoTower/lexer/tree recipe and
changes only `binder_topology_loss_weight` from 0 to 0.25. Topology decode
weight remains 0, so grammar/compiler legality and learned decode authority do
not change. E1212 evaluates the completed checkpoint with strict compiler-tree
decoding, the honest slot contract, and no unconstrained fallback.

## E1211-E1212 result — neutral/rejected

E1211 completed 395 steps (1,580 draws) on CPU with final loss `9.5528`.
E1212 is strict held-out `n=5`; it emitted AgentEvals JSONL and a pinned AgentV
bundle. Its entire headline scorecard exactly matches E1182/E1200:

| Metric | E1212 | E1182/E1200 |
| --- | ---: | ---: |
| Parse / meaningful | .4 / .4 | .4 / .4 |
| Strict-v2 | .2 | .2 |
| Placeholder fidelity / structural similarity | .28 / .2852 | .28 / .2852 |
| Component recall / reward | .3333 / .3388 | .3333 / .3388 |
| Timeout-empty rows / constrained fallback | 3 / 0 | 3 / 0 |

The unit topology-loss arm E1209 regressed this scorecard; the quarter-strength
dose is neutral. Binder-topology supervision is therefore rejected as a seed-7
stabilizer. This local scratch checkpoint is unsynced, non-promotable,
unservable, and ineligible as parent evidence. `n=5` is not shipping evidence.

Checkpoint SHA-256: `8c9694060c669703b599173d58a480f0a20f98c027836c2cc2c49e88274ebf11`.
AgentEvals JSONL:
`outputs/runs/e1212_v273_e1211_full_held_out/agentv/openui-model-ship-gates-2026-07-25t12-10-28-903275-00-00.eval.jsonl`.
Version stamp: train `harness.model_build.train=v24`; eval
`harness.model_build.eval=v58`, `evals.meaningful_program=2.13.0`,
`evals.scoring=v15`, `model.twotower=v273`, `config.levers=v41`.

## E1213-E1214 preregistration

E1213 is the next unused single-lever replication: exact E1181 seed-7 training
with only prompt-conditioned `slot_component_loss_weight` changed from 0 to 1;
its decode weight remains 0. E1166's seed-5 signal did not replicate on seed 3
(E1168), so E1214's locked strict held `n=5` AgentEvals/AgentV endpoint must
materially improve E1182 without a timeout regression. Otherwise this auxiliary
is rejected as a seed-7 stabilizer. It remains local scratch only and never
creates a ship claim.

Machine-readable record:
[iter-e1211-e1214-seed7-topology-slot-component-20260725.json](iter-e1211-e1214-seed7-topology-slot-component-20260725.json).
