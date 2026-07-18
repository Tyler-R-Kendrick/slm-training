# E327 frozen lexeme-prior decode sweep — 2026-07-17

E327 evaluates the unchanged E326 checkpoint at slot-component decode weights
1.25 and 2.0 under the same honest five-suite policy.

| Weight | AgentV | Smoke recall | Held structure | RICO structure |
| ---: | ---: | ---: | ---: | ---: |
| 1.00 | 4/5 | 0.3333 | 0.5458 | 0.4826 |
| 1.25 | 4/5 | 0.3333 | 0.5458 | 0.5523 |
| 2.00 | 4/5 | 0.3333 | 0.5408 | 0.6153 |

Both arms preserve parse, fidelity, meaningful rate, component recall, reward,
and all four passing suite gates. Weight 2 improves limited-RICO structure but
slightly lowers held structure. Neither arm changes the smoke predictions.

**Verdict:** reject weight scaling as the final-gate lever. Keep E326's
checkpoint default at 1.0; the remaining error needs joint multi-slot owner
evidence, not a larger global bias.
