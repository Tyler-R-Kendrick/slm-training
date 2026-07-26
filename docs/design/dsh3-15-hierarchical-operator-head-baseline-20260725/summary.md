# DSH3-15 hierarchical operator action head causal baseline (SLM-383)

Date: 2026-07-25
Status: measured; rejected
Scope: bounded CPU fixture causal baseline; no checkpoint or ship claim

## Decision

SLM-383 authorizes testing the encoder-side hierarchical operator action
head only after a token baseline is run. DSH3-14/SLM-382 (E803) rejected the
*decoder-visible* token hypothesis outright; per AGENTS.md IV.11 that left the
encoder-side/masked scoring mechanism untested. This run is that follow-up:
token baseline vs. weight-zero (frozen no-op capacity control) vs. enabled
(trained) arms over a shared fixture corpus whose gold operator depends only
on a typed candidate-group-size feature (never opaque ids or display names).

The enabled head strictly beats its own weight-zero capacity control -- it is
trainable and not a prediction-identical no-op -- but ties exactly with the
token baseline (both memorize the finite shape vocabulary to ceiling). DSH3-15's
acceptance bar requires causal improvement *beyond* the token baseline, which
does not hold here, so the stop rule applies and the head stays default-off.

## Matched result

| Arm | Seed | Operator accuracy | Parameters | Trainable | False admissions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `TOKEN_BASELINE` | 11 | 1.000 | 33857 | 33857 | 0 |
| `TOKEN_BASELINE` | 29 | 1.000 | 33857 | 33857 | 0 |
| `TOKEN_BASELINE` | 47 | 1.000 | 33857 | 33857 | 0 |
| `WEIGHT_ZERO` | 11 | 0.500 | 641 | 0 | 0 |
| `WEIGHT_ZERO` | 29 | 0.500 | 641 | 0 | 0 |
| `WEIGHT_ZERO` | 47 | 0.500 | 641 | 0 | 0 |
| `ENABLED` | 11 | 1.000 | 641 | 641 | 0 |
| `ENABLED` | 29 | 1.000 | 641 | 641 | 0 |
| `ENABLED` | 47 | 1.000 | 641 | 641 | 0 |

## Causal choice changes (vs. token-baseline control)

| Treatment | Seed | Changed | Rate | Correct | Wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| `WEIGHT_ZERO` | 11 | 6/12 | 0.500 | 0 | 6 |
| `WEIGHT_ZERO` | 29 | 6/12 | 0.500 | 0 | 6 |
| `WEIGHT_ZERO` | 47 | 6/12 | 0.500 | 0 | 6 |
| `ENABLED` | 11 | 0/12 | 0.000 | 0 | 0 |
| `ENABLED` | 29 | 0/12 | 0.000 | 0 | 0 |
| `ENABLED` | 47 | 0/12 | 0.000 | 0 | 0 |

## Acceptance and honesty

- `causal_change_rate_at_least_0_05_vs_weight_zero`: **pass**
- `correct_changes_exceed_wrong_changes_vs_weight_zero`: **fail**
- `enabled_beats_weight_zero_capacity_control`: **pass**
- `enabled_causally_improves_beyond_token_baseline`: **fail**
- `zero_false_legal_admissions`: **pass**

Stage-1 (operator-kind) decision only; stage-2 (typed-argument) selection has no distinguishing typed-feature signal under the current hash-scalar candidate features (SLM-398 follow-up) and is out of scope for this run.

CAP0 is retained because the codec is default-off and the disabled path
defers unchanged. CAP1 retention is unavailable because CERT_CAP1/SLM-379
does not exist. No efficiency conclusion is drawn from this semantic run.

The run completed in 37.97s with peak process memory 316,616,704 bytes. AgentV passed 6/6 evidence cases with zero execution errors.

No checkpoint was created, so the model card and README checkpoint summary
do not change.
