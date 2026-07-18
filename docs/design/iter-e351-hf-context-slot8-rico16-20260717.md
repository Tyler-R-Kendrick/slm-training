# E351 bounded RICO-16 diagnostic — 2026-07-17

E351 applies E350's retained decode policy to the first 16 `rico_held`
examples. The diagnostic evaluation completed in 25.6 seconds, under the hard
300-second cap.

All 16 examples parse and are meaningful. Component recall is 0.5208 and
reward is 0.7326, so the diagnostic AgentV gate passes 1/1 with no execution
errors. Placeholder fidelity is 0.2388 and structural similarity is 0.2208;
these modest values remain important limits even though the current diagnostic
gate passes.

**Verdict:** retain as encouraging bounded RICO evidence for E350. This is
diagnostic-only (`eval_limit=16`) and cannot support checkpoint promotion or a
production ship claim. Full `rico_held` remains unrun because every command is
hard-capped at five minutes.
