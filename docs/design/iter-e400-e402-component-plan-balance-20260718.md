# E400–E402 component-plan balance — 2026-07-18

E396 showed that stronger component-plan supervision plus slot-owner balance
fixes the held-out aggregate recall gate, but `held_out_settings_01` still
collapsed to common `Button`/`TextContent` types. The component-plan objective
had no class-balance option: root cross-entropy and positive bound-count loss
weighted every type uniformly despite a highly imbalanced corpus.

E400 adds an opt-in `component_plan_class_balance_power`. It derives normalized
inverse-frequency weights from all training-layout component occurrences,
applies them to root cross-entropy and positive bound-count loss, and leaves
negative bound evidence uniform. Power zero preserves the prior objective.
Focused model/config tests cover weight derivation, finite loss, persistence,
and CLI wiring.

The E400 experiment resumes E396's full state on the unchanged 998-record E357
corpus with balance power 0.5. It reaches its 29,000-token budget after 106.8
seconds, 567 cumulative steps, and 29,066 target tokens. The local checkpoint
SHA is `e62f0859172072a1567e38876a93a6aa76b2dbeaaa1a59d726fe6b779a230aad`.
It has no fresh loss-suite result, inherits best weighted NLL 5.8091, is
explicitly not synced, and is not promoted.

E401's complete held-out suite shows a real but narrow change:
`held_out_settings_01` now emits `SwitchItem`, meaningful rate rises
0.6→0.8, and only the tabs row fails. Aggregate recall remains 0.4833, while
structure regresses 0.5933→0.5217.

E402 rejects the candidate on the complete bounded suite set. Smoke meaningful
rate and recall fall to 0.3333 and 0.1667, below 0.66 and 0.35. OOD also incurs
one parse/fidelity failure and p95 latency rises to 27.23s. AgentV passes 3/4
with zero execution errors.

| Suite | n | Parse | Meaningful | Fidelity | Structure | Type recall | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0 | 0.3333 | 1.0 | 0.5114 | 0.1667 | 0.3163 |
| held_out | 5 | 1.0 | 0.8 | 1.0 | 0.5217 | 0.4833 | 0.7814 |
| adversarial | 4 | 1.0 | 0.5 | 1.0 | 0.5137 | 0.3750 | 0.4865 |
| ood | 4 | 0.75 | 0.5 | 0.75 | 0.4652 | 0.4167 | 0.4835 |

Every command used an external 290-second interrupt plus a forced kill ten
seconds later. E400 additionally used the trainer's internal 4.5-minute wall
limit and stopped normally on its token budget. E401 and E402 completed
normally; no timed-out process contributes evidence.

**Verdict:** retain the default-off balance mechanism as a reusable research
lever, but reject the E400 checkpoint. E396 remains the bounded HF-context
candidate. Do not spend RICO evaluation on E400 and do not promote or claim
ship.
