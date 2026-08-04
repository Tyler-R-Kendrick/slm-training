# Autotrain c1734: component-inventory screen

**Verdict:** component-inventory supervision is another exactly size-matched
quality null. Its 0.47% p50 gain is below policy v4's 5% efficiency floor, so
the arm is rejected and is not checkpoint, promotion, or ship evidence.

## Result matrix

| Arm | Params | n | Parse | Binder F1 | Meaningful | Structure | Recall | AST node / edge F1 | p50 | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,682,363 | 3 | 1.0 | .6333 | .3333 | .17417 | .25 | .26190 / 0 | 1,574.83 ms | complete fixture control; gates fail |
| component-inventory loss `1.0` | 1,682,363 | 3 | 1.0 | .6333 | .3333 | .17417 | .25 | .26190 / 0 | 1,567.47 ms | exact quality null; 0.47% below floor |

Both arms also match placeholder fidelity .5278, reward .7653, 11 neural
forwards, 31,118 unique completion states, 861 witness expansions, and 32,126
parser forks. Candidate training loss is worse (14.7713 vs 13.6692), and its
training wall is slower (2.721 vs 2.548 seconds). These are 22-step CPU scratch
checkpoints, seed 101734, batch 2, with the component-inventory head prebuilt in
both arms.

AgentV completed both arms without execution errors. Smoke `n=3` is below the
evidence floor, and held-out, adversarial, OOD, and `rico_held` were not run.
Lean is `not_applicable:screening`; promotion and RL remain locked.

## Flow result and next priority

The merged cross-cycle cooldown correctly excluded `binder-topology`,
`component-plan`, and `component-edge`, selected component-inventory, and now
advances to the distinct size-matched `component-structure` hypothesis. No new
harness failure was observed.

1. Run component-structure next under the same matched capacity and smoke
   primary.
2. Preserve deterministic-work counters and the 5% efficiency floor.
3. Do not interpret repeated fixture parse success as meaningful OpenUI quality;
   this cycle still misses the honest semantic gates.

Machine-readable evidence is in
[`autotrain-cycle-1734-component-inventory-screen.json`](autotrain-cycle-1734-component-inventory-screen.json).
