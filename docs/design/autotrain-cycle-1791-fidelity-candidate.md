# Autotrain c1791: placeholder-fidelity fixture candidate

**Verdict:** queue exact fresh-seed confirmation. Increasing only the
placeholder-fidelity loss weight from 0.5 to 1.5 improved every measured
meaning/structure signal while preserving parse rate, reducing p50 latency,
and keeping trainable parameters exactly matched.

| Arm | Params | Loss / train wall | Smoke quality | p50 | Decision |
| --- | ---: | --- | --- | ---: | --- |
| fidelity 1.5 | 1,608,962 | 13.5652 / 2.52 s | parse 1; meaning .3333; structure .17417; binder F1 .6333; recall .25; fidelity .5278; reward .76533 | 1,088.81 ms | fixture candidate; confirm |
| matched control 0.5 | 1,608,962 | 8.7293 / 3.08 s | parse 1; meaning 0; structure .14593; binder F1 .5; recall 0; fidelity .4444; reward .53800 | 1,154.30 ms | matched baseline |

The candidate gained .02823 structural similarity, .13333 binder-reference
F1, .25 component recall, and .08333 placeholder fidelity; latency improved
5.67%. Its higher training loss is consistent with the stronger auxiliary
weight and is diagnostic rather than a promotion metric. Both arms completed
3/3 smoke documents with no decode timeouts and zero AgentV execution errors.

This is fixture evidence only (`n=3`). Both arms fail honest ship gates,
including evidence volume, meaningful-program rate, structural similarity,
component recall, AST BEQ, canonical BEQ, and missing production suites. The
candidate is not promoted, reusable, synced, or ship-ready. The exact recipes
must re-hold on a fresh seed before any promotion cycle can open.

Lean is `not_applicable:screening`: the candidate is queued for confirmation,
not confirmed, so there is no formal promotion target. If confirmation holds,
the next promotion still requires the normal Lean preflight and both formal CI
lanes.

Machine evidence:
[`autotrain-cycle-1791-fidelity-candidate.json`](autotrain-cycle-1791-fidelity-candidate.json).
