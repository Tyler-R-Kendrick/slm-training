# Autotrain c1754: completion-bounds diagnostic

**Verdict:** reject. Completion bounds is an exact smoke quality null and is
0.15% slower than the matched control.

| Arm | Params | n | Parse | Binder F1 | Meaning | Structure | Recall | p50 | Loss / train wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 3 | 1 | 1 | .6667 | .03573 | .3333 | 14,431.80 ms | 11.5170 / 2.817 s |
| completion bounds | 1,608,962 | 3 | 1 | 1 | .6667 | .03573 | .3333 | 14,453.65 ms | 11.5170 / 3.348 s |

Both 24-step CPU scratch arms completed without timeout, fallback, or AgentV
execution error. Bounds changes neither output quality nor loss, raises p50 by
21.85 ms, and takes 1.19x training wall. Ship gates fail at fixture `n=3` and
on structure, recall, AST BEq, and canonical BEq. RL stays locked. Lean is
`not_applicable:screening`; actual promotion still requires a fresh proof.

Next: the distinct compact-canvas runtime diagnostic under the same matched
controls. Machine evidence:
[`autotrain-cycle-1754-completion-bounds-diagnostic.json`](autotrain-cycle-1754-completion-bounds-diagnostic.json).
