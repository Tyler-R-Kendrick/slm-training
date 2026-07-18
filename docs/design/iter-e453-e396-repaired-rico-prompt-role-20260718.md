# E453 E396 repaired-RICO prompt-role shard — 2026-07-18

E453 repeats E452 on the same E451 rows 1344–1439, checkpoint SHA, CPU,
HF-local context, grammar-LTR settings, slot contract, and no-fallback policy.
The only policy change is `prompt_role_constrained_decode=true`.

| Run | Meaningful | Fidelity | Structure | Type recall | Reward | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E452 control | 0.9792 | 1.0000 | 0.6421 | 0.8681 | 0.9778 | 2 |
| E453 prompt roles | 1.0000 | 1.0000 | 0.8609 | 1.0000 | 0.9956 | 0 |
| Delta | +0.0208 | 0.0000 | +0.2189 | +0.1319 | +0.0179 | -2 |

All 96 IDs match. Ninety-four predictions change; structure improves on 88
rows, ties on eight, and regresses on none. Type recall improves on 26 rows and
regresses on none. The formerly invalid `rico_hf_test_3283` gold is now a
six-card Stack, and E453 scores structure 0.93, type recall 1.0, and fidelity
1.0 on it.

E453 completes normally in about 218 seconds under the external 290-second
cap. It has zero fallback and decode timeout. AgentEvals JSONL and an AgentV
bundle are present; AgentV is 0/5 with zero execution errors because this
remains a one-suite diagnostic shard.

**Verdict:** accept prompt-role constrained decode for broader evaluation. The
matched repaired-corpus shard has no per-row structure or type-recall
regression. This does not establish full-RICO or five-suite ship gates; expand
through capped shards and re-run every bounded suite before changing champion
claims.
