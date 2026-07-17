# E296 25% DESIGN-dropout replication (2026-07-17)

E296 tests whether E295's one meaningful adversarial result persists at a lower
dropout rate without changing initialization or training recipe.

`e296-choice-design-dropout25-r1` uses the E295 CPU scratch choice recipe:
107 steps / 5,022 target tokens, seed 0, and no checkpoint sync. The stable hash
omits DESIGN for 127/480 records (26.46% realized) and retains it for 353.
Complete weighted NLL improves from E295's 7.3785 to 7.3503, between E292
all-DESIGN (7.2265) and E294 no-DESIGN (7.4977). Category NLLs are binding
8.1103, structural 5.7633, repair 7.8997, schema-OOD 7.1793, and broad 8.1960.
The loss-suite AgentV record passes 1/1 with no execution errors. Checkpoint
SHA-256 is
`b3c4df4cca25905d1101ed8006f430a772a7228f894530ef98cb8fd8cfc1a1ed`.

Frozen prompt-only ship evaluation exactly matches E294 on every headline
metric: parse 1.0 throughout, meaningful 0.0 throughout, AgentV 0/5, and 17
failed thresholds. E295's adversarial meaningful 0.25 therefore does not form a
smooth rate response at seed 0.

**Verdict:** E296 is not promotable or ship-ready. Keep the dropout mechanism,
but treat E295's one-example gain as unreplicated until a 50% second-seed arm
reproduces it. Do not combine dropout with the plan head yet.

Artifacts:

- `outputs/runs/e296-choice-design-dropout25-r1/`
- `outputs/runs/e296-choice-design-dropout25-honest-r1/`
- [machine-readable result](choice-design-dropout-results-iter-e296-20260717.json)
