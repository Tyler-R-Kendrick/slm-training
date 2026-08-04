# Autotrain c1752: coupled binder-topology screen

**Verdict:** the exactly size-matched coupled binder-topology arm is an exact
quality null on smoke and held-out. It is slower and has worse train loss.
Reject; this is fixture screening evidence, not promotion or ship evidence.

## Result matrix

| Arm | Params | Suite n | Parse | Binder F1 | Meaningful | Structure | Recall | p50 | Train loss / wall | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched topology-head control | 2,137,346 | smoke 3 / held 5 | 1 / 1 | .82222 / .70762 | .3333 / 0 | .51000 / .37006 | .25 / .16190 | 4,398.12 / 4,573.57 ms | 12.0579 / 3.000 s | complete fixture control; gates fail |
| topology loss `.25`, decode `1.0` | 2,137,346 | smoke 3 / held 5 | 1 / 1 | .82222 / .70762 | .3333 / 0 | .51000 / .37006 | .25 / .16190 | 4,663.12 / 4,614.14 ms | 12.1095 / 6.659 s | exact quality null; slower |

Both 22-step CPU scratch arms used the binder-topology head and
`compiler_decode_mode=tree`; only the candidate enabled its loss and legal-symbol
decode ranker. Both completed every record with zero decode timeouts and zero
unconstrained fallbacks. AgentV completed with no execution errors. The candidate
was 6.03% slower on smoke and 0.89% slower on held-out, while training took 2.22x
the control wall time.

## Signals and next hypothesis

The c1751 harness repair worked: compiler mode reached train and evaluation,
the capability validator accepted both recipes, the candidate's effective model
config retained topology loss `.25` and decode weight `1.0`, and both arms
finished inside their symmetric walls. The resulting exact output tie shows that
coupling this isolated head at these weights does not change the chosen programs.

The next preregistered experiment should test the distinct coupled
`component-structure` arm, then inspect per-branch legal-candidate score deltas if
another exact tie occurs. Keep the matched compiler path, head capacity, seed,
steps, and parameter count. Do not increase capacity or weaken grammar gates.

## Honest gates and formal evidence

Ship gates fail: smoke `n=3` and held-out `n=5` are below the evidence floor;
meaning, component recall, AST BEq, and canonical BEq miss thresholds; adversarial,
OOD, and full `rico_held` were not run. RL stays locked. This was a promotion-cadence
screen with no confirmed champion, so no promotion claim was attempted and Lean
proof is `not_applicable:no_champion`. Any real promotion remains fail-closed on a
fresh proved Lean certificate.

Machine-readable evidence is in
[`autotrain-cycle-1752-coupled-binder-topology-screen.json`](autotrain-cycle-1752-coupled-binder-topology-screen.json).
