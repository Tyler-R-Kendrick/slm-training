# E713 — explicit outer-group topology

Date: 2026-07-21  
Status: completed historical scratch diagnostic; checkpoint invalidated

E713 recognizes explicit prompts that place a sibling group inside an outer
component. For “two cards around a separator, then put the group inside an outer
card,” the semantic plan now compiles the two Cards and Separator into an inner
Stack, nests that Stack in the outer Card, and attaches the outer Card to the
root. The correction is schema- and verifier-checked rather than record-specific.

The targeted adversarial record became parseable and structurally correct. A
metric false positive was also fixed: text-only routing is mechanical gaming only
when a requested component can directly own one of the required semantic roles;
a Separator cannot own content.

On the then-current frozen E620 checkpoint, the five-suite `n=19` replay raised
adversarial binding-aware strict meaningfulness 0.50→0.75, structure
0.88325→0.916675, node F1 0.9152→0.9318, and edge F1 0.8419→0.8611. The other
four suites matched E712 on quality. AgentV remained 0/5 and Rico still exceeded
the 160-symbol canvas budget with p95 190, so this was never ship evidence.

The subsequent symbol-only output contract intentionally invalidates the E620
checkpoint and every other pre-v2 checkpoint. E713 therefore remains historical
diagnostic evidence for the topology implementation only; it cannot justify a
current model or deployment claim. No checkpoint was created, synced, or
promoted.

Evidence: [JSON](iter-e713-outer-group-topology-20260721.json). Contract:
[symbol-only output](symbol-only-output-contract.md).

