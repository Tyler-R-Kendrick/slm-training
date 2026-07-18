# E491–E494 semantic optional-slot decode — 2026-07-18

These runs use the unchanged E396 checkpoint and E451 corpus on CPU with local
HF context. They retain E490's 320-token grammar LTR, component-plan weight 2,
slot-component weight 8, prompt-role and honest slot-contract constraints,
eight generation steps, three attempts, and no unconstrained fallback. Every
process completed normally under the hard three-minute policy.

The generalized failure was in constrained decode, not the checkpoint:

1. explicit prompt-role counts stopped constraining once their minimum count
   was reached, allowing unrelated components before the root;
2. `ImageBlock.alt` is schema-optional, so the component closed after its
   required source even when the visible slot contract supplied the alt.

E491's first diagnosis changed nothing on `rico_hf_test_293` (structure 0.35).
E492's root-stop rule raised it to 0.4889 but exposed the optional-alt close.
E493 filled schema-optional semantic slots before close and reached structure
1.0, with parse, meaningful rate, fidelity, and type recall all 1.0, zero
fallbacks/timeouts, and no exact-record special case.

## E494 bounded results

| Suite | n | Parse | Meaningful | Fidelity | Structure | Δ vs E489 | Type recall | Reward | Δ reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 1.0 | 1.0 | 0.7656 | +0.0833 | 0.6667 | 0.9690 | -0.0040 |
| held_out | 5 | 1.0 | 1.0 | 1.0 | 0.8083 | +0.0245 | 0.9048 | 0.9862 | -0.0006 |
| adversarial | 4 | 1.0 | 1.0 | 1.0 | 0.8061 | 0.0000 | 1.0 | 0.9767 | 0.0000 |
| ood | 4 | 1.0 | 1.0 | 1.0 | 0.6546 | +0.0203 | 0.8750 | 0.9865 | 0.0000 |

AgentV passes 4/4 with zero execution errors. There are no decode failures,
fallbacks, or timeouts. The small smoke and held-out reward decreases are
retained as negative evidence rather than hidden.

**Verdict:** accept the generalized decoder fix for a larger RICO shard.
E493 is diagnostic-only and E494 is bounded evidence; neither is a new
checkpoint or production ship claim.

A subsequent 48-row `rico_held` attempt at offset 144 was externally
interrupted at 170 seconds (exit 124). It is not a run, contributes no metrics,
and is retained only as timeout evidence. The next RICO shard must be smaller.
