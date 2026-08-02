# Autotrain c1759: steps candidate rejected on confirmation

**Verdict:** reject the c1757 doubled-steps candidate. On a fresh matched seed,
44 steps reduce training loss by 77.00% but regress every measured quality signal
that differs: meaningful-program rate falls from .3333 to 0, structural
similarity falls 66.99%, component recall falls from .25 to 0, and p50 latency
increases 134.51%.

| Arm | Params | Steps | n / complete / timeout | Parse | Meaning | Structure | Recall | p50 | Loss / train wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source control | 1,608,962 | 22 | 3 / 3 / 0 | 1 | .3333 | .17417 | .25 | 1,362.47 ms | 20.6566 / 3.141 s |
| doubled steps confirmation | 1,608,962 | 44 | 3 / 3 / 0 | 1 | 0 | .05750 | 0 | 3,195.06 ms | 4.7515 / 4.678 s |

Both CPU scratch arms used seed 101759, batch size 2, and strict
compiler-tree constrained evaluation. The repaired champion queue reconstructed
the exact c1757 source recipes (22-step control versus 44-step candidate),
reclaimed the c1758 attempt that failed before `experiment_started`, and then
executed both c1759 arms. This closes the c1757 efficiency hypothesis: lower
token loss and one-seed p50 improvement do not transfer to quality or latency.

AgentV completed without execution errors. `--ship-gates` fails at fixture
`n=3`, meaning, structure, recall, AST BEq, and canonical BEq; held-out,
adversarial, OOD, and `rico_held` were not run. Lean is
`not_applicable:confirmation` because the candidate is rejected before any
promotion claim. Neither checkpoint is synced, reusable, served, promoted, or
ship evidence.

Next: treat the doubled-steps fingerprint as exhausted. Prioritize a distinct,
size-matched objective that directly supervises structural/meaningful quality;
use training loss only as a diagnostic, and keep runtime-only arms labeled as
such. The handoff must derive successor priorities after observing the outcome
instead of repeating the now-falsified confirmation hypothesis.

Machine evidence:
[`autotrain-cycle-1759-steps-confirmation.json`](autotrain-cycle-1759-steps-confirmation.json).
