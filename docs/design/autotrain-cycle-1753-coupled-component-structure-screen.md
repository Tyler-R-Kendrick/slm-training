# Autotrain c1753: coupled component-structure screen

**Verdict:** reject. The exactly size-matched joint component-plan/edge arm
changes the decoded programs, but reduces smoke structural similarity from
`.05750` to `.04333`, leaves meaningful rate at zero, and is slower.

## Result matrix

| Arm | Params | n | Parse | Binder F1 | Meaningful | Structure | AST node / edge F1 | p50 | Train loss / wall | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched joint-head control | 1,913,789 | 3 | 1 | .82222 | 0 | .05750 | 0 / 0 | 1,424.68 ms | 13.7437 / 2.740 s | complete fixture control; gates fail |
| plan+edge loss/decode `1.0` | 1,913,789 | 3 | 1 | .82222 | 0 | .04333 | .01626 / 0 | 1,712.01 ms | 18.5889 / 7.346 s | quality regression; slower |

Both 23-step CPU scratch arms used the same joint head capacity and
`compiler_decode_mode=tree`; only the candidate enabled both plan and edge loss
and legal-symbol decode weights. Both completed all records with no timeout,
fallback, or AgentV execution error. The treatment lowers the primary by
`.01417` (24.64% relative), raises p50 20.17%, takes 2.68x training wall, and has
worse loss.

## Signals and next hypothesis

This run proves the coupled decode path is active: unlike the preceding isolated
topology null, it changes structural output and AST-node overlap. The direction
is harmful, so more weight on the same joint mechanism is not justified. The
quality-family bank is now exhausted on this integrated code. The next run is
the preregistered completion-bounds runtime diagnostic with a matched control;
it cannot become a quality or promotion claim.

Ship gates fail at fixture `n=3`; meaningful rate, structure, recall, AST BEq,
and canonical BEq miss thresholds. Held-out, adversarial, OOD, and full
`rico_held` were not run. RL stays locked. No champion exists, so Lean is
`not_applicable:screening`; a real promotion still requires a fresh proved Lean
certificate.

Machine-readable evidence is in
[`autotrain-cycle-1753-coupled-component-structure-screen.json`](autotrain-cycle-1753-coupled-component-structure-screen.json).
