# E879-E885: union-vocabulary warm-start and atomic matched evaluation

The warm-start loader now owns context-vocabulary compatibility. It constructs
an ordered union of the parent checkpoint vocabulary and the current filtered
corpus vocabulary, then remaps embedding rows by token identity. Parent tokens
cannot disappear when a continuation corpus is smaller, and current-only tokens
remain available. The model is not asked to translate markers or text.

E879 is a zero-step E852 → E872 compatibility proof. It retained all 712 parent
context tokens, preserved all 131 tensors exactly with zero RMS drift, and
completed in 1.52 seconds. Its serving checkpoint has the same SHA as E852;
the distinct full-state artifact records the continuation run metadata. This is
a local scratch diagnostic with checkpoint sync explicitly disabled.

E880 is invalid because it repeated the earlier harness omission: the command
used AST-plan weights 0/0. The canonical `strict_compiler_tree` policy now owns
weights 4/2 in the centralized lever registry. E885 omitted both CLI flags and
exactly reproduced the E881 matched control, proving that callers cannot create
that mismatch through omission.

| Run | Suite | n | parse | meaning-v1 | strict-v2 | fidelity | structure | recall | reward | timeouts | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E885 E879 zero-step control | smoke | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6589 | 0.7500 | 0.9490 | 0 | 0/1 |
| E882 E877 continued candidate | smoke | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.7633** | 0.7500 | 0.9490 | 0 | 0/1 |
| E883 E852 baseline | held_out | 3 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.1417 | 0.3333 | 0.3203 | 2 | 0/1 |
| E884 E877 candidate | held_out | 3 | 0.6667 | 0.3333 | 0.0000 | 0.5714 | 0.4161 | 0.4762 | 0.6041 | 1 | 0/1 |

E877 is retained only as a research candidate. It improves matched smoke
structure by 0.1044 without changing the other smoke headlines, and improves
most held-out diagnostics, but strict-v2 falls on the tiny timeout-confounded
held-out slice. Neither checkpoint passes AgentV, and no ship or promotion claim
is made. E852 remains the retained scratch baseline pending broader evidence.

Canonical evidence:
[`iter-e879-e885-vocab-union-matched-eval-20260722.json`](iter-e879-e885-vocab-union-matched-eval-20260722.json).
