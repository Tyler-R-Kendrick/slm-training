# E491–E494 semantic optional-slot decode — 2026-07-18

These runs use the unchanged E396 checkpoint and E451 corpus on CPU with local
HF context. The generalized decoder fix stops after satisfying visible
prompt-role counts and fills schema-optional semantic slots such as
`ImageBlock.alt` before closing a component.

On the targeted `rico_hf_test_293`, structure progressed from 0.35 (E491) to
0.4889 (E492) and then 1.0 (E493), without a record-specific rule.

| Suite | n | Parse | Fidelity | Structure | Δ vs E489 | Type recall | Reward | Δ reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 0.7656 | +0.0833 | 0.6667 | 0.9690 | -0.0040 |
| held_out | 5 | 1.0 | 1.0 | 0.8083 | +0.0245 | 0.9048 | 0.9862 | -0.0006 |
| adversarial | 4 | 1.0 | 1.0 | 0.8061 | 0.0000 | 1.0 | 0.9767 | 0.0000 |
| ood | 4 | 1.0 | 1.0 | 0.6546 | +0.0203 | 0.8750 | 0.9865 | 0.0000 |

AgentV passes 4/4 with zero execution errors, failures, fallbacks, or timeouts.
The small reward decreases remain negative evidence. A later 48-row RICO
attempt was externally interrupted at 170 seconds and contributes no metrics.

**Verdict:** accept bounded evidence only; this is not a new checkpoint or
production ship claim.
