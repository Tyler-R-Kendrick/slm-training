# E716 schema-typed string candidates

E716 fixes the concrete defect exposed by E715: the lexer-native `STRING`
terminal expanded the tokenizer's entire literal kind, which also contains
booleans, `null`, `LIT_NUM`, and `LIT_END`. The compiler therefore treated a
number marker as a legal first value for required string fields such as
`TextArea.name`.

The grammar-to-token map now admits only fixed string symbols (`STR:*`), visible
template symbols (`<SYM_n>`), and the legacy string opener when present. Raw
byte text and non-string literal markers are excluded. Model component version
`model.twotower` advances from v187 to v188, and the shared token-map path is now
watched by that component.

## Verification

- Full focused grammar/compiler suite: 188 passed in 11.31 seconds.
- Regression coverage proves `STRING` excludes `true`, `false`, `null`,
  `LIT_NUM`, `LIT_END`, and raw byte generation while retaining fixed string
  and template symbols.
- Version-stamp verification passes with one component touched.

## Local smoke replay

The clean-revision local CPU replay uses the E714 checkpoint and the rejected
E715 tree-decode recipe. It changes only the typed candidate mapping.

| Arm | Parse | Strict v2 | Fidelity | Structure | Recall | Timeouts | p50 / p95 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| E715 pre-fix tree | 0.0 | 0.0 | 0.6667 | 0.0619 | 0.1667 | 1 | 5.75s / 8.03s | 0/1 |
| E716 typed tree | 0.0 | 0.0 | 0.6667 | 0.0619 | 0.1667 | 1 | 5.25s / 8.00s | 0/1 |

All three predictions pass `symbol_only_output`. The previously chosen
`TextArea.name=LIT_NUM` edge is no longer legal; the model selects the fixed
grammar symbol `STR:email`, and the `null-required:TextArea.name` failure reason
disappears. Aggregate quality is otherwise neutral and remains below every
semantic gate.

Decision: retain the generalized type-safety correction because it removes an
invalid schema edge and enforces output contract v2. Do not promote the
checkpoint or extend tree decoding to other suites; the remaining failure is
model selection and document completion, not string-terminal typing.

Machine-readable evidence:
[iter-e716-schema-typed-string-candidates-20260721.json](iter-e716-schema-typed-string-candidates-20260721.json).
