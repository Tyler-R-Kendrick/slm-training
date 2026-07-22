# E886-E888: recovered current-policy baseline

E886 reconstructed the lost E852 local scratch checkpoint from committed E851
data after the original `/tmp` worktree disappeared. The exact 600-step CPU
recipe completed in 53.30 seconds with final loss 4.07366085. Its checkpoint SHA
is `76cd2dc2…b09819`, bit-for-bit identical to E852, and checkpoint sync was
explicitly disabled.

| Run | Suite | n | parse | meaning-v1 | strict-v2 | fidelity | structure | recall | reward | timeout / fallback | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E887 | smoke | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6589 | 0.7500 | 0.9490 | 0 / 0 | 0/1 |
| E888 | held_out | 5 | 0.8000 | 0.4000 | 0.2000 | 0.6400 | 0.2588 | 0.4190 | 0.7178 | 1 / 2 | 0/1 |

This exactly restores the retained scratch baseline and gives it a current
five-row held-out board. It is not a ship result: AgentV fails, held-out strict
meaning and structure remain below policy, and only smoke plus held-out ran.
Retain E886 as the local parent for the next matched intervention; target the
held-out strict-meaning/fallback path rather than adding undirected steps.

