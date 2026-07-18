# E476 schema-array item bounded evaluation — 2026-07-18

E476 evaluates the generalized E475 array-item constraint across all four
bounded E451 suites. The decoder carries pinned `items` schemas through nested
choice-state list frames and rejects primitive, reference, component, union, or
nested-array elements of the wrong type.

Recipe: unchanged E396 checkpoint and E451 corpus, CPU, local HF context,
320-token grammar LTR, automatic content floor, component-plan weight 2,
slot-component weight 8, E474 policy, honest constrained slot contracts, eight
generation steps, three attempts, and no fallback. The complete process
finished normally in 28.5 seconds under the external 290-second cap.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.6822 | 0.6667 | 0.9730 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.7838 | 0.9048 | 0.9868 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 1.0 | 0.9767 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.6343 | 0.8750 | 0.9865 |

Smoke, held-out, and adversarial are metric-identical to E474. OOD structure
improves 0.6279→0.6343 while all gate metrics and reward are unchanged. The
affected gallery row now emits `ImageGallery([])` rather than populating its
object-item array with a component. AgentV passes 4/4 with zero execution
errors; every suite has zero failures, fallback, and decode timeouts.

**Verdict:** accept for bounded evaluation. Fresh full-RICO evidence is
required because the E475 replay audit found broad impact. E474 remains the
authoritative five-suite result meanwhile.
