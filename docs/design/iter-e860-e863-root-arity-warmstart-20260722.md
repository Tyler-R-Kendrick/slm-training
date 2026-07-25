# E860-E863: root-arity warm start

E860 attempted the corrected full-head root-reference arity objective without
declaring the lexer compiler mode. The centralized capability validator rejected
the configuration before model construction; no checkpoint or training evidence
was produced.

E861 reran correctly with `compiler_decode_mode=tree`. It warm-started the E852
checkpoint on unchanged E851 data, enabled only root-reference arity loss weight
1, and completed 120 local CPU steps in 20.32 seconds under the 95-second
harness cap. Final loss was 7.5547 and initialized-weight RMS drift was 0.003642.
The explicit no-sync checkpoint SHA-256 is
`5eebfaddc49590ecd8773e93c822448bd7ed671c6e73e8616c4b6de2879df8e2`.

E862/E863 evaluate the same checkpoint and E842 smoke with root-arity decode
weight 1/0. All other decode settings match E853.

| Arm | Parse | Strict-v2 | Fidelity | Structure | Recall | Reward | p50 / p95 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E862 weight 1 | 1.0000 | 0.6667 | 0.8333 | 0.4850 | 0.6667 | 0.9110 | 3480.44 / 4314.79 ms | 0/1 |
| E863 weight 0 | 1.0000 | 0.6667 | 0.9167 | 0.4753 | 0.6667 | 0.9360 | 2970.79 / 3691.67 ms | 0/1 |

The trained head applied 13 times and changed one choice at weight 1, improving
structure by only 0.0097 while reducing fidelity by 0.0833. Both arms regress
heavily from E853's strict 1.0000, fidelity 1.0000, structure 0.6589, and recall
0.7500. The dominant failure is continuation training drift, not insufficient
root-arity decode authority. Reject E861; retain E851/E852. No remote workflow,
bucket sync, promotion, deployment, or ship claim occurred.
