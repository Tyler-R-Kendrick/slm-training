# E855-E857: structurally disjoint action group

E855 tested the valid alternative to E854: a two-action `Buttons` fixture whose
topology is disjoint from the held-out one-button smoke record. Derived from
E851 under unchanged strict gates, it admitted 352/352 rows with zero rejects,
warnings, sanitizer fallbacks, recommendations, or experiment candidates.

E856 trained locally on CPU with scratch context, lexer output, batch size 4,
AdamW, and 600 steps. It completed in 57.13 seconds under the 95-second harness
cap; final loss was 3.8642. Its explicit no-sync checkpoint SHA-256 is
`0c045aa81147df28e52ea2fd976d53f77a0ed5a0e0751ce577441055243d3e88`.

E857 used the unchanged E842 smoke suite and matched strict compiler-tree
decode recipe. Parse and fidelity stayed 1.0000, but strict meaningfulness fell
from E853's 1.0000 to 0.6667 and structure fell from 0.6589 to 0.5500. The
button prediction still omitted `Buttons`; component recall remained 0.7500.

| n | Parse | Meaning-v1 | Strict-v2 | Fidelity | Validity | Structure | Recall | Reward | p50 / p95 | Fallback / timeout | AgentV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1.0000 | 1.0000 | 0.6667 | 1.0000 | 0.6000 | 0.5500 | 0.7500 | 0.9610 | 3378.21 / 3597.25 ms | 0 / 0 | 0/1 |

Reject E856 and remove the ineffective producer fixture from future builds.
E851/E852 remains the stronger scratch baseline. No remote workflow, bucket
sync, deployment, promotion, or ship claim occurred.
