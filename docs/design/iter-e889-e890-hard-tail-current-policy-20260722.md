# E889-E890: hard-tail continuation under current policy

E889 continued the recovered E886 scratch checkpoint for 20 CPU steps on the
committed E872 hard-tail corpus. The run completed in 2.91 seconds with final
loss 5.41076660, initialized all 1,681,282 weights, and produced RMS weight
drift 0.00116243. Checkpoint sync was explicitly disabled.

| Run | Parent / suite | n | parse | meaning-v1 | strict-v2 | fidelity | structure | recall | reward | timeout / fallback | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E888 | E886 / held_out | 5 | 0.8000 | 0.4000 | 0.2000 | 0.6400 | 0.2588 | 0.4190 | 0.7178 | 1 / 2 | 0/1 |
| E890 | E889 / held_out | 5 | 1.0000 | 0.4000 | 0.2000 | 0.6733 | 0.1889 | 0.5190 | 0.8570 | 0 / 3 | 0/1 |

The hard-tail continuation removes the timeout and improves parse by 0.20,
normalized fidelity by 0.0333, component recall by 0.10, and reward by 0.1392.
It does not improve either meaningful-program metric, increases fallback-marked
rows from two to three, and reduces structural similarity by 0.0700. Two rows
collapse to a single `TextContent`, while the form and settings rows remain
overgenerated.

Reject E889 as a replacement checkpoint and retain E886 as the local parent.
The next matched intervention should preserve the base distribution while
replaying hard-tail examples; pure hard-tail continuation trades topology for
surface validity. This is diagnostic held-out evidence, not a ship result:
AgentV fails, only five rows ran, and no production suite or ship gate passed.
