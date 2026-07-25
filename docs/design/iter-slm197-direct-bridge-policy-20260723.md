# SLM-197: direct legal-edit policy matrix

**Status:** measured fixture wiring; powered experiment blocked.
**Honest verdict:** `inconclusive_fixture_only`.
**Ship claim:** none.

## Decision

The direct-policy contract, multi-positive set-mass objective, matched
time encodings, checkpoint migration, and exact live-candidate decode are
implemented and exercised. The requested powered D0-D5 comparison cannot
run honestly because the committed SLM-196 corpus is non-publishable and
contains only four rows across two targets; its X22/local-corruption controls
are absent. D0 and D1 are therefore unavailable, not silently synthesized.

## Matrix

| Arm | Status | Seed runs | Dev top-1 positive range | Free-run target exact range |
| --- | --- | ---: | ---: | ---: |
| `D0` X22 control | unavailable: no hash-pinned X22 baseline manifest supplied | 0 | — | — |
| `D1` plain local corruption | unavailable: no hash-pinned local-corruption corpus supplied | 0 | — | — |
| `D2` plain full bridge | fixture measured | 5 | 1.000–1.000 | 0.000–0.000 |
| `D3-linear` time-conditioned full bridge; linear schedule | fixture measured | 5 | 1.000–1.000 | 0.000–0.000 |
| `D3-fourier` time-conditioned full bridge; Fourier schedule | fixture measured | 5 | 1.000–1.000 | 0.000–0.000 |
| `D4` one-hot planner negative control | fixture measured | 5 | 1.000–1.000 | 0.000–0.000 |
| `D5` multi-positive direct control reserved for flow comparison | fixture measured | 5 | 1.000–1.000 | 0.000–0.000 |

## Contract evidence

- Equal parameter capacity: `True` ([4769]).
- Time encodings use fixed-budget schedule progress; gold remaining distance
  and bridge length never enter the scorer.
- The likelihood is `logsumexp(all live) - logsumexp(certified positives)`.
- UNKNOWN remains separately masked and is never used by an explicit negative loss.
- Free-running decode re-enumerates exact candidates, verifies transition
  certificates, replays the edit, and logs every state/action.
- Plan conditioning is default-off; D4 alone receives the planner one-hot.

## Recipe

- Device/backend: `cpu` / `exact legal-edit candidates`
- Steps/seeds: `8` / `[0, 1, 2, 3, 4]`
- Train/dev rows: `2` / `2`
- Independent targets: `2`
- Wall time: `4.131s` (cap `2.8m`)
- AgentV: `{'total': 5, 'passed': 5, 'failed': 0, 'executionErrors': 0, 'durationMs': 24, 'meanScore': 1}`

No checkpoint was written or promoted, so MODEL_CARD.md and the README
checkpoint summary do not change.

## Confirmation firewall

- SLM-196 corpus manifest is non-publishable fixture evidence
- D0 X22 and D1 local-corruption controls are not hash-pinned
- only two independent targets and four bridge rows are available

`--confirm` fails closed until a publishable corpus and frozen confirmation
manifest are supplied. This fixture result does not freeze a production baseline.
