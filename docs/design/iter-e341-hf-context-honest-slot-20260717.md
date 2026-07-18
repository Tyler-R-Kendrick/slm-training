# E341 bounded honest visible-slot contract — 2026-07-17

E341 evaluates the unchanged E337 checkpoint with component-plan bias off,
visible slot-contract context enabled, constrained slot decoding enabled, and
`honest_slot_contract=true`. Hidden gold placeholder inventory remains
forbidden. The four-suite run completed in 36.8s under the hard 300-second cap.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.6389 | 0.3386 | 0.0 | 0.0 | 0.0 |
| held_out | 5 | 0.8 | 0.2700 | 0.1619 | 0.0 | 0.0 | 0.0 |
| adversarial | 4 | 1.0 | 0.1458 | 0.2899 | 0.0 | 0.0 | 0.0 |
| ood | 4 | 1.0 | 0.2167 | 0.1884 | 0.5 | 0.25 | 0.3435 |

AgentV passes 0/4 with no execution errors, and RICO was intentionally
omitted. This is nevertheless the first bounded HF-context arm to recover
nonzero semantic quality: OOD meaningful/recall/reward become
0.50/0.25/0.3435 and smoke fidelity reaches 0.6389.

**Verdict:** retain honest visible-slot conditioning as a causal lever, but do
not promote or claim ship. Held/adversarial semantics remain zero and every
AgentV row fails. No checkpoint was written.

