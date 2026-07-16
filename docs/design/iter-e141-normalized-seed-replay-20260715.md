# E141 normalized three-seed checkpoint replay — 2026-07-15

E141 replays the E135/E138/E139 8-step HF checkpoints under one declared
evaluation policy. This removes the prior comparison confound from different
attempt and chosen-token verification settings.

| Seed | Checkpoint | Parse | Placeholder validity | Structure | Reward | Timeouts | p50 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | E135 / E140 | 0.0 | 0.1333 | 0.1189 | 0.0 | 2 | 20,001 ms |
| 1 | E138 | 0.0 | 0.0000 | 0.0250 | 0.0 | 2 | 20,000 ms |
| 2 | E139 | 0.0 | 0.0000 | 0.0000 | 0.0 | 0 | 4,067 ms |

Seed 0 remains the best diagnostic checkpoint, but it does not parse or earn
reward. This is a checkpoint-selection signal only, not a ship signal. The
training corpus and loss weights remain unchanged.

The replay also exercised the new policy serializer with CLI passthrough
values (`grammar_top_k` and `grammar_verify_chosen_only` represented as null),
confirming that evaluation does not crash when a checkpoint supplies those
settings.

Next: instrument the constrained decoder's per-token dead-end and selection
trace around the invalid-output path before another expensive HF training run.
