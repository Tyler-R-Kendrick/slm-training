# Autotrain continuous-openui-local cycle 2: frozen replay confirms the harness fix, rejects on fixture n

**Verdict:** reject (fixture ship-gate, not a model regression). This cycle
replays the identical frozen control/candidate arm from cycle 1
(`frozen_manifest_sha256`
`7462dc61b1fd1023203cb6df61716a7a4136b39f4dbc52216fa9ceffc0d4c6dd`) now that
[PR #1354](https://github.com/Tyler-R-Kendrick/slm-training/pull/1354) is on
the branch. Both 21-step CPU scratch arms (1,608,962 trainable params,
`wf_smoke_v2`) trained and — for the first time in this loop — **completed a
full AgentEvals scoreboard with 0 AgentV execution errors**, confirming the
harness repair actually unblocked evaluation:

| Arm | Parse | Meaningful | Struct sim | Binder F1 | p50 ms | AgentV errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1.0 | 0 | .0575 | .63333 | 1,053.75 | 0 |
| bounds | 1.0 | 0 | .0575 | .63333 | 1,050.67 | 0 |

Both arms are matched-knob and produce identical metrics, so this is not an
attributable model comparison. Ship gates correctly reject: `smoke` ran only
`n=3` (the fixture screening suite), far under the `n>=20` volume floor, and
`held_out`/`adversarial`/`ood`/`rico_held` are absent (not built for this
screening cycle). Quality thresholds (`meaningful_program_rate`,
`structural_similarity`, `component_type_recall`, `ast_beq_rate`,
`canonical_beq_rate`, `reward_score`) also fail outright — expected for a
21-step CPU screening run with no real training signal, not evidence of a
regression. This is an honest fixture-demo result, not a production claim.

Neither checkpoint is reusable, promotable, synced, or ship evidence. Lean is
`not_applicable:retry_measurement`. Per SDLC Phase A this cycle is
non-positive (`fixture_insufficient_n_alone`), so no new stacked PR — local
commit and docs only.

The next queued hypothesis is the distinct, size-matched `component-plan`
quality lever (`c20260803-continuous-openui-local-8c0b60dd-c2-component-plan`).

Machine evidence:
[`autotrain-continuous-openui-local-c2-frozen-replay-rejected.json`](autotrain-continuous-openui-local-c2-frozen-replay-rejected.json).
