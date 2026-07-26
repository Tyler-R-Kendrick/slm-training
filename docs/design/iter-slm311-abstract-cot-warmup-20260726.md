# SLM-311 Abstract-CoT warm-up fixture

This local, hermetic T=3 run uses the existing AP-020 collector and AP-019
segment-payload builder. It records immutable per-phase command, configuration,
input/output hashes, token counts, and result references. It is **wiring only**:
no model checkpoint was written, no locked test selected an iteration, and it
does not support a promotion or ship claim.

- Iterations: 3; policies: fixture://slm311-abstract-cot-warmup/iteration/1, fixture://slm311-abstract-cot-warmup/iteration/2, fixture://slm311-abstract-cot-warmup/iteration/3.
- Resume: 0 reused, 3 newly run in this invocation.
- Meaningful-parse: not measured (there was no model evaluation).
- AgentEvals/AgentV checks the T=3 lineage assertion in the adjacent artifact.
