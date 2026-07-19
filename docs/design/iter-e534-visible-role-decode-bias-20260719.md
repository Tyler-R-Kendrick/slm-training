# E534 — honest visible-role decode bias

E534 tests whether E533 failed because the E531 model ignored the visible
semantic-role contract or because that contract carried no useful causal
signal. It holds the E531 checkpoint and E533 OOD n=4 recipe fixed, then adds
one weight-4 bias to legal bound-component choices. Candidate types come only
from official component names already present in prompt prose and their
schema-compatible visible slots. The decoder never reads gold component types
or the gold reference graph and fails closed unless honest visible-role context
is enabled.

The CPU diagnostic completed under the 170-second process cap from clean commit
`bdb994a` and emitted AgentEvals JSONL plus the pinned AgentV SDK bundle. A
preceding `r1` attempt lacked the AgentV runtime and is not evidence; all numbers
below are from completed run `e534-e531-ood160-visible-role-bias4-r2`.

| Metric | E533 contract only | E534 direct bias | Delta |
| --- | ---: | ---: | ---: |
| Syntax parse rate | 1.0000 | 1.0000 | 0.0000 |
| Meaningful program rate v1 | 0.0000 | 0.2500 | +0.2500 |
| Placeholder fidelity | 0.3833 | 1.0000 | +0.6167 |
| Placeholder validity | 0.5300 | 1.0000 | +0.4700 |
| Structural similarity | 0.1159 | 0.1959 | +0.0801 |
| Component type recall | 0.2292 | 0.5417 | +0.3125 |
| Reward | 0.3685 | 0.7402 | +0.3717 |
| AST node F1 | 0.1627 | 0.1627 | 0.0000 |
| AST edge F1 | 0.0417 | 0.0417 | 0.0000 |
| Strict binding-aware meaning | 0.0000 | 0.0000 | 0.0000 |
| AgentV | 0 / 1 | 0 / 1 | unchanged |

The bias applied at 18 component decisions and changed all 18 choices. That
establishes useful causal information in the visible contract and isolates
E533's failure to model uptake. It does not establish model readiness:
reference-graph validity is still 0/4, strict meaning is 0/4, AST overlap is
unchanged, and AgentV fails.

Retain E534 as an opt-in honest constrained-inference lever. Do not promote the
checkpoint or claim ship readiness from this diagnostic subset. The next
bounded lever must target reference construction/topology using visible
authority; another semantic-role prompt-conditioning train is not justified.
Machine-readable evidence is in
[the E534 JSON](iter-e534-visible-role-decode-bias-20260719.json).
