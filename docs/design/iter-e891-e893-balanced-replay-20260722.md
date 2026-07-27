# E891-E893: balanced hard-tail replay

E891 continued E886 for 20 CPU steps on the committed E851 base corpus while
replaying committed E872 hard-tail rows at 25%. The run completed in 3.06
seconds, sampled 60 base and 20 replay examples, initialized all 1,681,282
weights, and produced RMS weight drift 0.00109225. Sync was explicitly disabled.

| Run | Checkpoint / suite | n | parse | meaning-v1 | strict-v2 | fidelity | structure | recall | reward | timeout / fallback | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E887 | E886 / smoke | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6589 | 0.7500 | 0.9490 | 0 / 0 | 0/1 |
| E893 | E891 / smoke | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5783 | 0.6667 | 0.9450 | 0 / 0 | 0/1 |
| E888 | E886 / held_out | 5 | 0.8000 | 0.4000 | 0.2000 | 0.6400 | 0.2588 | 0.4190 | 0.7178 | 1 / 2 | 0/1 |
| E892 | E891 / held_out | 5 | 0.8000 | 0.8000 | 0.2000 | 0.8000 | 0.3298 | 0.7143 | 0.7844 | 1 / 0 | 0/1 |

Balanced replay doubles held-out meaning-v1, raises fidelity by 0.16, structure
by 0.0710, recall by 0.2952, and reward by 0.0666, while removing both
fallback-marked rows. Strict-v2 and the single timeout do not improve. Smoke
retains perfect parse/meaning/fidelity but loses 0.0806 structure and 0.0833
recall.

Retain E891 as the strongest held-out research candidate, but keep E886 as the
unqualified baseline because the smoke topology regression prevents a clean
replacement. The next matched arm should lower replay exposure and test whether
the held-out gain survives. This is not a ship result: both AgentV checks fail,
the boards are diagnostic subsets, and no production gate ran.
