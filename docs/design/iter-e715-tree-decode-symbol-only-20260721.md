# E715 tree decode on the symbol-only baseline

E715 tests one decode-only treatment on the E714 checkpoint before spending
another training cycle. The local CPU smoke evaluation changes only
`compiler_decode_mode` from `off` to `tree`; checkpoint, dataset, canvas, retry,
timeout, grammar, and honest slot-contract settings remain fixed.

| Arm | n | Parse | Strict v2 | Fidelity | Structure | Recall | Timeouts | p50 / p95 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| E714 control, compiler off | 3 | 0.3333 | 0.0 | 1.0 | 0.0880 | 0.4167 | 0 | 14.06s / 14.27s | 0/1 |
| E715 treatment, compiler tree | 3 | 0.0 | 0.0 | 0.6667 | 0.0619 | 0.1667 | 1 | 5.75s / 8.03s | 0/1 |

All three E715 predictions pass the `symbol_only_output` verifier. The compiler
tree treatment lowers median latency, but converts the one parseable control
prediction into a parse failure, reduces fidelity, structure, and component
recall, produces no eligible AST metrics, and adds a timeout. AgentV remains
0/1 with zero execution errors and mean score 0.5.

Decision: reject tree compiler decoding for this checkpoint and close this
decode arm. Do not extend it to the remaining suites and do not retrain it.
Investigate the required-schema value selection failure exposed by
`TextArea.name` before trying another broad decode policy.

Machine-readable evidence:
[iter-e715-tree-decode-symbol-only-20260721.json](iter-e715-tree-decode-symbol-only-20260721.json).
