# E41 judged-corpus training feedback — 2026-07-15

The E41 lexer-native recipe was rerun after replacing `remediated_roots` with
the independently judged corpus.

| corpus | records | fingerprint | smoke parse | structural similarity | reward | p50 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| judged roots | 405 | `b6d135be...a11ec1` | 0.000 | 0.000 | 0.000 | 28,815 ms |

The prior corpus admitted 61 under-specified language-contract rows; those are
now excluded. The model still produced malformed lexer-native binder streams,
so data quality was a real upstream defect but not the sole cause of the
decoder failure. This is the clean baseline for the next lexer training-signal
intervention.

The run used the judged source-controlled corpus and emitted telemetry,
AgentEvals, and the smoke scoreboard under
`outputs/runs/iter-e41-judged-20260715/`.
