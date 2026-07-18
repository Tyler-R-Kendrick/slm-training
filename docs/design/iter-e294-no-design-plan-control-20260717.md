# E294 no-DESIGN component-plan control (2026-07-17)

## Question

E293's no-DESIGN plan arm lacked a separately trained no-plan control. E294
runs that missing arm to isolate plan training from the context-policy repair.

## Matched recipe

`e294-choice-no-design-control-r1` matches E293 `r3` except both component-plan
weights are zero: CPU scratch, choice codec, d64/h2, seed 0, batch 2, diffusion
corruption, no DESIGN.md context, 107 steps / 5,022 target tokens, and no
checkpoint sync. The corrected train summary reports `no-design-md-context`,
matching the serialized checkpoint config.

| Metric | E294 control | E293 plan |
| --- | ---: | ---: |
| weighted NLL | **7.4977** | 7.5550 |
| binding NLL | **8.0988** | 8.2260 |
| structural NLL | 5.8927 | **5.8901** |
| repair NLL | **8.3503** | 8.4051 |
| schema-OOD NLL | **7.3295** | 7.3557 |
| broad NLL | **8.2535** | 8.3028 |

Checkpoint SHA-256:
`df30ca03f8f2bc3313b1b8afff9c40b7ab18c4fd2b0e8ae1b3888ba780d9add0`.
The loss-suite AgentV record passes 1/1 with zero execution errors.

## Frozen honest evaluation

The control uses scratch context, grammar constraints, prompt-derived honest
slot contracts, no DESIGN.md context, and no unconstrained fallback.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Component recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.0 | 0.3333 | 0.3500 | 0.0 | 0.0 |
| held_out | 5 | 1.0 | 0.0 | 0.0000 | 0.2514 | 0.0 | 0.0 |
| adversarial | 4 | 1.0 | 0.0 | 0.2500 | 0.2363 | 0.0 | 0.0 |
| ood | 4 | 1.0 | 0.0 | 0.0000 | 0.2369 | 0.0 | 0.0 |
| rico_held | 3 | 1.0 | 0.0 | 0.0000 | 0.0901 | 0.0 | 0.0 |

These metrics are exactly identical to E293's same-checkpoint decode-bias-off
ablation, even though plan training changes 69/73 shared non-head tensors.
AgentV is 0/5, zero execution errors, and 17 frozen gates fail.

## Verdict

E294 isolates E293's effect. At this budget, plan training does not improve the
base generator's discrete outputs and slightly worsens weighted NLL. The learned
plan head is nevertheless active: enabling it in E293 changes 38 legal choices,
improves several fidelity/structure cells, and cuts failures 17→13. That is a
secondary-ranking gain, not a meaningful-program gain; neither checkpoint is
promotable or ship-ready.

Artifacts:

- `outputs/runs/e294-choice-no-design-control-r1/`
- `outputs/runs/e294-choice-no-design-control-honest-r1/`
- [machine-readable result](choice-plan-control-results-iter-e294-20260717.json)
