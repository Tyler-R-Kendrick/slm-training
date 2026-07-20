# E559 — twofold rare slot-owner coverage

E559 changes only E558's rare-owner record multiplier from 4× to 2×. The
threshold remains at most 10 labels, selecting 75 of 244 records and expanding
the sampling pool to 319 records.

The clean run processed 1,296 target tokens in 31.14 seconds under
`max_wall_minutes=3` and wrote checkpoint SHA
`1d11926d6784cac62f6d65249030be9b392d0f185e59b1a4212d9f0ff9aac861`.

Against E555 on matched OOD `n=4`, fidelity improves 0.3000→0.4417 and
component recall 0.1250→0.2708. Structure is 0.1085, AST-node F1 0.2048, and
AST-edge F1 0.0648. Meaning-v1/v2 remain 0, reward falls 0.5453→0.1643, and
AgentV remains 0/1.

**Verdict:** reject the checkpoint. Retain the sampler and narrow eligibility
to truly scarce owner classes with at most four labels in E560. Evidence:
[JSON](iter-e559-owner-coverage2-20260720.json).
