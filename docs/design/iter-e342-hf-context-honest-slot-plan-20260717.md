# E342 bounded honest-slot plus component plan — 2026-07-17

E342 composes E341's honest visible-slot contract with the E337 checkpoint's
trained component-plan decode weight of 1. The four-suite evaluation completed
in 50.7s under the hard 300-second cap.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.1944 | 0.4044 | 0.0 | 0.0 | 0.0 |
| held_out | 5 | 0.8 | 0.1333 | 0.2034 | 0.0 | 0.0 | 0.0 |
| adversarial | 4 | 0.75 | 0.1458 | 0.1524 | 0.0 | 0.0 | 0.0 |
| ood | 4 | 0.75 | 0.0833 | 0.1895 | 0.0 | 0.0 | 0.0 |

AgentV passes 0/4 with no execution errors. Relative to E341, plan bias cuts
fidelity, regresses adversarial/OOD parse, and removes OOD's recovered
meaningful/recall/reward signal. RICO was intentionally omitted.

**Verdict:** reject E342. The old plan bias is incompatible with the honest
visible-slot path. The evidence supports training with visible-slot
conditioning rather than adding the existing decoder head. No checkpoint was
written.

