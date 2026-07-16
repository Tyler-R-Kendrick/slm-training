# Remediated-roots independent judge gate — 2026-07-15

The existing root corpus was rebuilt through a prompt/output judge before
training admission. Syntax, schema, and lint checks were retained; the new
judge rejects records that are valid OpenUI but do not answer the prompt.

| stage | records |
| --- | ---: |
| collected | 480 |
| verifier rejected | 0 |
| judge/quality rejected | 61 |
| published | 405 |

The rejected set is dominated by 61 `language_contract` rows whose prompts are
only “Emit the OpenUI construct: the X component” while their outputs are
generic synthetic wrappers. The boolean-literal example was especially clear:
the prompt requested a lexical value, but the output added an unrelated Stack,
Separator, and TextContent layout.

Published judged corpus fingerprint: `b6d135be9806c708486f1f09efd5c993bafdbff99d029c1985488c57c0a11ec1`.
Mean retained quality score: 0.9663. The next training run must use
`src/slm_training/resources/train_data/remediated_roots_judged`, not the prior
`remediated_roots` directory.

This judge is deterministic and conservative, not an LLM claim. It is an
independent prompt/output contract gate that runs before records are written to
a training snapshot; later work can add a model or human review tier without
weakening this gate.
