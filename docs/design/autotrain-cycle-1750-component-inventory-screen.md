# Autotrain c1750: component-inventory screen

**Verdict:** the exactly size-matched component-inventory arm is an exact smoke
quality null. Its 3.50% p50 reduction is below the 5% efficiency floor, and its
loss is worse. Reject; this is neither promotion nor ship evidence.

## Result matrix

| Arm | Params | n | Parse | Binder F1 | Meaningful | Structure | Recall | AST node / edge F1 | p50 | Train loss / wall | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,682,363 | 3 | 1.0 | .6333 | .3333 | .17417 | .25 | .26190 / 0 | 1,164.70 ms | 10.5061 / 3.067 s | complete fixture control; gates fail |
| inventory loss `1.0`, decode weight `0.0` | 1,682,363 | 3 | 1.0 | .6333 | .3333 | .17417 | .25 | .26190 / 0 | 1,123.96 ms | 11.5443 / 2.859 s | exact quality null; efficiency below floor |

Both 23-step CPU scratch arms completed all three records with zero decode
timeouts and zero unconstrained fallbacks. AgentV completed without execution
errors. The checkpoints are local with explicit no-sync and are not reusable
champions.

## Training signal and harness feedback

The candidate's inventory targets are present and its auxiliary head is active:
late-step telemetry records 2.5–4.0 positive components per row and top-k recall
between .433 and .75. However, the campaign explicitly evaluates with
`component_inventory_decode_weight=0.0`. The auxiliary loss can shape shared
context features, but the learned inventory scores are not directly consumed by
decode. Component-plan and component-edge screens used the same loss-only
pattern. Three exact output-quality nulls therefore justify improving the
screening harness, not merely rotating to another detached scoring head.

The successor repair couples each structural auxiliary training loss to its
matching non-zero decode ranking weight while keeping grammar legality,
parameter count, seed, steps, and prebuilt head capacity matched. Confirmation
and promotion fingerprints must retain those decode knobs. The next run should
exercise the coupled binder-topology arm; if it remains null, inspect per-head
score deltas at legal branch points before further weight rotation.

## Honest gates and formal evidence

Ship gates fail: `n=3` is below the evidence floor, and meaningful rate,
structural similarity, component recall, AST BEq, and canonical BEq miss their
thresholds. Held-out, adversarial, OOD, and `rico_held` were not run. RL stays
locked. No empirical optimum band or confirmed champion exists, so Lean
promotion proof is `not_applicable:screening`; actual promotion remains
fail-closed on the mathlib-free LeverProof certificate.

Machine-readable evidence is in
[`autotrain-cycle-1750-component-inventory-screen.json`](autotrain-cycle-1750-component-inventory-screen.json).
