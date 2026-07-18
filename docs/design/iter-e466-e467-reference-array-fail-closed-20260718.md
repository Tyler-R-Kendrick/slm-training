# E466–E467 reference-array fail-closed decode — 2026-07-18

E466 repairs a schema-general fail-open in semantic-density decode. When a
component's child array accepts component references only, the decoder
restricts choices to prior compatible references. With no prior references,
however, it previously fell back to every legal token, allowing a raw
placeholder to masquerade as a child component. It now closes the empty array
when legal; remaining placeholders must be materialized by actual content
components.

The correction is derived from the pinned component schema and applies to all
reference-only component arrays. It does not special-case Modal or any fixture.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, current prompt-role constrained decode, honest
constrained slot contracts, eight generation steps, three attempts, and no
fallback. Both processes completed normally under the external 290-second
cap.

E466's complete four-row OOD diagnostic changes `ood_modal_01`: the raw body
placeholder leaves `Modal.children`, and a real `TextContent` is emitted.
Modal structure/recall improve 0.6075/0.5000→0.7850/0.7500. Aggregate OOD
structure/recall/reward improve
0.5835/0.8125/0.9835→0.6279/0.8750/0.9865. Parse, meaningful output, and
fidelity remain 1.0 with zero failures, fallback, or decode timeouts.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.6822 | 0.6667 | 0.9730 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.7838 | 0.9048 | 0.9868 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.6279 | 0.8750 | 0.9865 |

E467 completes all four bounded suites in about 22 seconds. Relative to E464,
smoke and adversarial are unchanged. Held-out recall is unchanged and reward
rises 0.0006, while structure falls 0.0185. The OOD gains are larger:
structure +0.0444, recall +0.0625, and reward +0.0030. Every suite has zero
failures, fallback, and decode timeouts; AgentV passes 4/4 with zero execution
errors.

**Verdict:** accept for bounded evaluation. The fix removes schema-invalid
child content, materially improves OOD, and preserves every bounded gate.
The subsequent prediction-level audit finds no affected E460 RICO rows,
permitting exact reuse in E468's current-policy five-suite result.
