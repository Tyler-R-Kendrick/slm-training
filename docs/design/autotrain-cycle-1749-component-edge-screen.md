# Autotrain c1749: component-edge screen

**Verdict:** the exactly size-matched component-edge arm is an exact smoke
quality null. It preserves parse, meaning, structure, recall, and AST scores,
while its 0.79% p50 reduction is below the 5% efficiency floor. Candidate loss
is worse and training wall is 1.16 times control. Reject; this is neither
promotion nor ship evidence.

## Result matrix

| Arm | Params | n | Parse | Binder F1 | Meaningful | Structure | Recall | AST node / edge F1 | p50 | Train loss / wall | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,766,987 | 3 | 1.0 | .6333 | .3333 | .17417 | .25 | .26190 / 0 | 1,130.21 ms | 14.4206 / 2.895 s | complete fixture control; gates fail |
| component-edge loss `1.0` | 1,766,987 | 3 | 1.0 | .6333 | .3333 | .17417 | .25 | .26190 / 0 | 1,121.26 ms | 15.4778 / 3.353 s | exact quality null; efficiency below floor |

Both 22-step CPU scratch arms completed all three records with zero decode
timeouts and zero unconstrained fallbacks. AgentV completed without execution
errors. The checkpoints are local with explicit no-sync and are not reusable
champions.

## Signals and next hypothesis

Component-edge supervision did not move any decoded output metric. Along with
the repeated component-plan null, this suggests the auxiliary objectives may
not be changing the logits consumed by grammar-constrained decode at this
budget. The next registered distinct arm is component-inventory; it tests a
different target before escalating to a harness audit.

1. Run the exactly size-matched component-inventory arm next.
2. If it is also an exact quality null, inspect auxiliary-target prevalence,
   non-zero gradient flow, parameter updates, and whether decode consumes the
   trained head. Do not keep rotating weights without that causal evidence.
3. Keep RL locked. A three-record fixture cannot authorize RL or promotion.
4. Preserve the Lean gate: screening has no empirical optimum band, while an
   actual champion promotion remains blocked without locked LeverProof evidence.

## Honest gates and formal evidence

Ship gates fail: `n=3` is below the evidence floor, and meaningful rate,
structural similarity, component recall, AST BEq, and canonical BEq miss their
thresholds. Held-out, adversarial, OOD, and `rico_held` were not run. No
empirical optimum band or confirmed champion exists, so Lean promotion proof is
`not_applicable:screening`, not absent authority.

Machine-readable evidence is in
[`autotrain-cycle-1749-component-edge-screen.json`](autotrain-cycle-1749-component-edge-screen.json).
