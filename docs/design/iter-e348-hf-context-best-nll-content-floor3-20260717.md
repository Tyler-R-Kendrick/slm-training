# E348 bounded best-NLL three-content floor — 2026-07-17

E348 raises E347's minimum content-component floor from two to three while
keeping the same best-weighted-NLL checkpoint and honest visible-slot policy.
The four-suite evaluation completed in 28.3 seconds, under the hard
300-second cap.

The local unsynced checkpoint SHA is
`6f0ecf7cce2ebc7c61f133c13456ac91bcd4861bd3e2f4f70a3a72473c211985`.
Every example parses. Smoke/held/adversarial/OOD meaningful rate remains
0.6667/0.40/0.75/0.75 and component recall remains 0.50/0.20/0.50/0.50.
Fidelity and reward improve in every suite, but structural similarity falls
in smoke, held-out, and adversarial. AgentV remains 3/4 with no execution
errors; held-out still misses only its 0.30 component-recall gate at 0.20.
RICO was omitted.

**Verdict:** reject E348 in favor of E347. The extra forced component does not
clear another gate and trades structural similarity for surface fidelity.
Stop the monotonic floor sweep at two components.
