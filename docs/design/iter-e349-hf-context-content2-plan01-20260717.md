# E349 bounded two-content floor plus weak plan bias — 2026-07-17

E349 adds a 0.1 component-plan decode weight to E347's two-content floor,
best-weighted-NLL checkpoint, and honest visible-slot policy. The four-suite
evaluation completed in 20.7 seconds, under the hard 300-second cap.

The plan bias was applied 32 times across 16 examples but changed only one
component choice. Held-out component recall remains 0.20, so AgentV remains
3/4 with no execution errors. Against E347, smoke/adversarial/OOD aggregate
quality is unchanged; held-out fidelity falls from 0.6867 to 0.6467 while
structure rises from 0.4375 to 0.4541. RICO was omitted.

**Verdict:** reject E349 in favor of E347. Weak component-plan bias does not
repair the remaining held-out type-selection failure.
