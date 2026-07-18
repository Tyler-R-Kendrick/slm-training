# E319 distinct slot consumption — 2026-07-17

E319 fixes a generalized choice-decoder defect exposed by E318. Required
string arguments were selected from `bound_component_count`, so every string
inside one composite component reused the same slot. The canonical constrained
choice path now selects the first symbol not actually present in the emitted
prefix. A multi-string component therefore consumes `@0`, then `@1`, while the
accepted E315 distinct-slot content floor still requires all bound components
before root closure.

The unchanged E318 r2 checkpoint (SHA
`b4e5a87b158e9c2b184f3d850d45948c76ac613f6d2034c92e5787f126f534d9`)
was rerun under the frozen honest policy.

| Suite | n | Parse | Fidelity | Structure | Meaningful | Component recall | Reward | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke | 3 | 1.0 | 1.0 | 0.5464 | 0.6667 | 0.3333 | 0.6407 | Fail: recall needs 0.35 |
| held_out | 5 | 1.0 | 1.0 | 0.4431 | 0.4000 | 0.2000 | 0.3916 | Fail: recall needs 0.30 |
| adversarial | 4 | 1.0 | 1.0 | 0.5970 | 0.5000 | 0.3750 | 0.4805 | Pass |
| ood | 4 | 1.0 | 1.0 | 0.4304 | 0.5000 | 0.2500 | 0.4992 | Pass |
| limited `rico_held` | 3 | 1.0 | 1.0 | 0.4215 | 1.0000 | 0.5556 | 1.0000 | Pass |

Against E318, limited-RICO fidelity recovers 0.4167→1.0, structure
0.2468→0.4215, and reward 0.791→1.0. Every other suite metric is unchanged.
AgentV remains 3/5 with the same two metric failures. No fallback was used.

**Verdict:** accept the distinct-slot decoder correction. Do not promote the
unchanged E318 checkpoint or claim ship: semantic component selection still
misses smoke/held recall and remains worse than E316 on OOD. The next mechanism
must score a candidate against all slots its schema consumes, not only the next
slot.
