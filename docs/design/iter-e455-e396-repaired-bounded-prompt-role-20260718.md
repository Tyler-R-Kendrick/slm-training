# E455 E396 repaired bounded prompt-role evaluation — 2026-07-18

E455 evaluates all four complete bounded suites from E451's repaired corpus
with E454's exact prompt-role policy and the unchanged E396 checkpoint (SHA
`feefa0564490bd1db42f79ff710143ad8ed07ab9e4e324f2744a30f8c2f2eee0`).

Recipe: CPU, local HF context, 320-token grammar LTR, automatic content floor,
component-plan weight 2, slot-component weight 8, prompt-role constrained
decode, honest constrained slot contracts, eight generation steps, three
attempts, and no unconstrained fallback. The process completed normally in
26 seconds under the external 290-second cap.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5600 | 0.5000 | 0.9770 |
| held_out | 5 | 1.0 | 0.6000 | 1.0 | 0.6400 | 0.5048 | 0.5922 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.7661 | 1.0 | 0.9760 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.5835 | 0.8125 | 0.9835 |

Fallback and decode timeout counts are zero for every suite. AgentV passes
4/4 with zero execution errors.

**Verdict:** all bounded suites clear their local gates. Relative to E398,
smoke is unchanged; held-out structure and recall improve; adversarial
meaningful rate rises 0.75→1.0, structure 0.6762→0.7661, and recall
0.75→1.0; OOD structure and recall also improve. This is bounded-suite
evidence, not a five-suite, promotion, or production HF claim.
