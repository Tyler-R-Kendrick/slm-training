# SLM-198: bridge curriculum and target-balance ablation

**Status:** measured fixture wiring; powered experiment blocked.
**Decision:** select no curriculum.
**Ship claim:** none.

## Result

The seven-arm schedule, target-first sampler, deterministic difficulty
features, exposure ledger, anti-curriculum, and resumable cursor are wired
against the frozen SLM-197 direct legal-edit scorer. The committed corpus
cannot distinguish the ablation: its train split has two rows from one target,
all bridges have length two, and dependency capsules are empty.

| Arm | Class | Seeds | Dev top-1 positive range | Free-run exact range |
| --- | --- | ---: | ---: | ---: |
| `uniform_rows` | measured_fixture | 5 | 1.000-1.000 | 0.000-0.000 |
| `uniform_targets` | measured_fixture | 5 | 1.000-1.000 | 0.000-0.000 |
| `length_curriculum` | measured_fixture | 5 | 1.000-1.000 | 0.000-0.000 |
| `entropy_curriculum` | measured_fixture | 5 | 1.000-1.000 | 0.000-0.000 |
| `dependency_curriculum` | measured_fixture | 5 | 1.000-1.000 | 0.000-0.000 |
| `anti_curriculum` | measured_fixture | 5 | 1.000-1.000 | 0.000-0.000 |
| `oracle_difficulty` | development_diagnostic | 5 | 1.000-1.000 | 0.000-0.000 |

## Exposure and policy contract

- Final target support equal: `True`.
- Balanced target exposure equal: `True`.
- Candidate-token totals equal in this matched fixture: `True`.
- Parameter capacity equal: `True` ([4769]).
- Train/dev target isolation: `True`.
- `oracle_difficulty` is development-only and cannot be selected.

## Recipe

- Device/backend: `cpu` / `SLM-197 direct legal-edit scorer`
- Epochs/seeds: `8` / `[0, 1, 2, 3, 4]`
- Train/dev rows: `2` / `2`
- Train/dev independent targets: `1` / `1`
- Wall: `7.005s` (cap `2.8m`)
- AgentV: `{'total': 5, 'passed': 5, 'failed': 0, 'executionErrors': 0, 'durationMs': 18, 'meanScore': 1}`

No checkpoint was written or promoted, so the model card and README
checkpoint summary do not change.

## Confirmation firewall

- the committed SLM-196 corpus is a non-publishable four-row fixture
- the train split contains only one target and two decision rows
- no powered long-bridge or deep-structure confirmation slice exists
- uniform-target balance and staged curricula are indistinguishable here

`--confirm` fails closed. A future confirmation must freeze a publishable,
multi-target/multi-length corpus and compare one preregistered selected arm
only against `uniform_targets`.
