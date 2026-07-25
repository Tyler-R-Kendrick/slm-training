# E858-E859: lexer structural decode diagnosis

E858 attempted to apply the previously retained choice-tokenizer schema-value
and semantic-plan-root-margin recipe to E852. The centralized lever validator
rejected both levers because this checkpoint uses the lexer/tree path. No
evaluation artifacts were produced and E858 is not evidence.

E859 instead changed only the lexer-compatible semantic-plan decode weight and
margin from E853's 4/2 to 8/4. It used the same E852 checkpoint, E842 smoke
suite, and strict compiler-tree policy.

| n | Parse | Meaning-v1 | Strict-v2 | Fidelity | Validity | Structure | Recall | Reward | p50 / p95 | Fallback / timeout | AgentV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.6000 | 0.6422 | 0.6667 | 0.9450 | 3595.31 / 3928.61 ms | 0 / 0 | 0/1 |

The stronger weights regress E853 structure (0.6589) and recall (0.7500), so
retain 4/2. Extra generic structure persists in the callout prediction, while
the button still omits `Buttons`. The next arm needs a trained structural signal
or generalized producer evidence, not a larger decode bias. No checkpoint,
remote workflow, bucket sync, promotion, deployment, or ship claim occurred.
