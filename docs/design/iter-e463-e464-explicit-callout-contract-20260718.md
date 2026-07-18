# E463–E464 explicit callout contract — 2026-07-18

E463 tests a missing generalized prompt contract: natural prompts that
explicitly request a callout were not represented in
`prompt_role_component_counts`, so prompt-role constrained decode could not
require the named component. The decoder now recognizes visibly counted or
article-prefixed `Callout` requests and treats only arguments 1 and 2 as its
semantic placeholder slots. It does not infer unmentioned wrappers such as
`Buttons`, `CardHeader`, or `Separator`.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, prompt-role constrained decode, honest constrained
slot contracts, eight generation steps, three attempts, and no fallback.
Both processes completed normally under the external 290-second cap.

E463's complete three-row smoke diagnostic changes only
`smoke_callout_01`. That row's component recall rises 0.5→1.0 and structure
0.3833→0.7500. Aggregate smoke structure/recall improve
0.5600/0.5000→0.6822/0.6667, while reward changes 0.9770→0.9730. Parse,
meaningful output, and fidelity remain 1.0 with zero failures, fallback, or
decode timeouts. The single-suite AgentV envelope reports 1/5 because four
required suites are intentionally absent; E464 supplies the complete bounded
evidence.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.6822 | 0.6667 | 0.9730 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.8023 | 0.9048 | 0.9862 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.5835 | 0.8125 | 0.9835 |

E464 completes all four bounded suites in about 27 seconds. Held-out,
adversarial, and OOD are metric-identical to E461. Every suite has zero
failures, fallback, and decode timeouts; AgentV passes 4/4 with zero execution
errors.

**Verdict:** accept the explicit-callout decoder correction for bounded
evaluation. The material smoke structure/recall gains outweigh the 0.004
reward reduction, and no other bounded suite changes. The subsequent audit
finds zero matching E451 RICO prompts, permitting exact E460 reuse in E465's
current-policy five-suite result.
