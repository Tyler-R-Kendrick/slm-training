# E803 reserved discrete-operator token baseline (SLM-382)

Date: 2026-07-23
Status: measured; rejected
Scope: bounded CPU symbolic baseline; no checkpoint or ship claim

## Decision

The versioned reserved target codec and compiler-membership boundary are retained
default-off, but the model-visible token hypothesis is rejected on this corpus.
The canonical symbolic question exposes state and legal-set identity without an
edit-intent channel, so multiple different transformations are gold for the same
visible input. Reserved tokens change choices but cannot resolve that ambiguity.

## Matched result

| Arm | Seed | Exact action | Operator ID | Result AST | False admissions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `RESULT_AST_ONLY` | 11 | 0.500 | 0.750 | 0.500 | 0 |
| `RESULT_AST_ONLY` | 29 | 0.500 | 0.750 | 0.500 | 0 |
| `RESULT_AST_ONLY` | 47 | 0.500 | 0.750 | 0.500 | 0 |
| `OPERATOR_ONLY` | 11 | 0.500 | 0.750 | 0.500 | 0 |
| `OPERATOR_ONLY` | 29 | 0.500 | 0.750 | 0.500 | 0 |
| `OPERATOR_ONLY` | 47 | 0.500 | 0.750 | 0.500 | 0 |
| `OPERATOR_PLUS_RESULT` | 11 | 0.500 | 0.750 | 0.500 | 0 |
| `OPERATOR_PLUS_RESULT` | 29 | 0.500 | 0.750 | 0.500 | 0 |
| `OPERATOR_PLUS_RESULT` | 47 | 0.500 | 0.750 | 0.500 | 0 |

All arms use the same train/held-out decisions, seeds, optimizer steps,
learning rate, and parameter count. The treatment arms use explicit
`<|openui_operator:v1|>` framing and typed canonical operator arguments.

## Causal choice changes

| Treatment | Seed | Changed | Rate | Correct | Wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| `OPERATOR_ONLY` | 11 | 0/4 | 0.000 | 0 | 0 |
| `OPERATOR_ONLY` | 29 | 0/4 | 0.000 | 0 | 0 |
| `OPERATOR_ONLY` | 47 | 2/4 | 0.500 | 1 | 1 |
| `OPERATOR_PLUS_RESULT` | 11 | 2/4 | 0.500 | 1 | 1 |
| `OPERATOR_PLUS_RESULT` | 29 | 0/4 | 0.000 | 0 | 0 |
| `OPERATOR_PLUS_RESULT` | 47 | 2/4 | 0.500 | 1 | 1 |

## Acceptance and honesty

- `causal_change_rate_at_least_0_05`: **fail**
- `correct_changes_exceed_wrong_changes`: **fail**
- `held_out_result_ast_improves_across_seeds`: **fail**
- `zero_false_legal_admissions`: **pass**

CAP0 is retained because the codec is default-off and the disabled path
defers unchanged. CAP1 retention is unavailable because CERT_CAP1/SLM-379
does not exist. No efficiency conclusion is drawn from this semantic run.

The run completed in 13.02s with peak process memory 345,001,984 bytes. AgentV passed 5/5 evidence cases with zero execution errors.

No checkpoint was created, so the model card and README checkpoint summary
do not change.
