# Autotrain c1757: steps efficiency candidate

**Verdict:** queue for confirmation; do not promote. Doubling training from 22
to 44 steps improves smoke p50 by 12.02% with parse and meaningful-program rate
held, but structural similarity drops 22.33% and component recall drops from
.25 to .1667.

| Arm | Params | Steps | n / complete / timeout | Parse | Meaning | Structure | Recall | p50 | Loss / train wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 22 | 3 / 3 / 0 | 1 | .3333 | .17417 | .25 | 1,346.77 ms | 12.7757 / 2.882 s |
| doubled steps | 1,608,962 | 44 | 3 / 3 / 0 | 1 | .3333 | .13527 | .1667 | 1,184.95 ms | 12.3065 / 4.952 s |

Both CPU scratch arms used seed 101757, batch size 2, and strict
compiler-tree constrained evaluation. The candidate improves MPR/ms by 13.66%
and loss by 3.67%, while spending 1.72x training wall. The climb policy marks
this a screening efficiency candidate because parse and MPR hold; that does not
erase the structural regression and is not a quality or ship clear.

AgentV completed without execution errors. `--ship-gates` fails at fixture
`n=3`, meaning, structure, recall, AST BEq, and canonical BEq; held-out,
adversarial, OOD, and `rico_held` were not run. Lean is
`not_applicable:screening`. The candidate checkpoint is retained only for a
fresh preregistered, counterbalanced confirmation; it is not promoted, synced,
served, or shipped.

Next: confirm across a distinct seed and both AB/BA orders, with structure and
recall treated as explicit rollback signals. Promotion remains fail closed
without held-out evidence and current Lean certificate authority.

Machine evidence:
[`autotrain-cycle-1757-steps-efficiency-candidate.json`](autotrain-cycle-1757-steps-efficiency-candidate.json).
