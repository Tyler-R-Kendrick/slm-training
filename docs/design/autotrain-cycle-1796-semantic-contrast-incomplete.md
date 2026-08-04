# Autotrain c1796: semantic-contrast measurement incomplete

**Verdict:** inconclusive infrastructure result. Both size-matched arms trained,
but neither completed the preregistered smoke + held-out measurement. Partial
quality aggregates are not comparable and do not establish whether the new
semantic-contrast loss helps or hurts.

| Arm | Params | Loss / train wall | Smoke complete | Held-out complete | Decision |
| --- | ---: | --- | ---: | ---: | --- |
| semantic contrast 0.25 | 1,608,962 | 12.1460 / 6.24 s | 1/3 (2 timeouts) | 4/5 (1 timeout) | incomplete |
| matched control 0.0 | 1,608,962 | 11.9107 / 7.01 s | 1/3 (2 timeouts) | 0/5 (5 timeouts) | incomplete |

Both CPU scratch arms used 22 steps, batch 2, seed 101796, the same 835
binding/content/contract contrast pairs, margin 1, and pair fraction .5. The
only treatment change was `semantic_contrast_loss_weight: 0→.25`; parameter
count is identical. The checkpoints differ, so the treatment reached training,
but the loss difference is not a quality verdict.

The evaluator recorded a 26.84–27.61 s cumulative wall for eight promotion
documents. Its fair-share allocator consequently reduced the nominal 24 s
per-record timeout to about 3.0–3.35 s. The result is typed and honest: the
control completed only one smoke document and no held-out documents; the
treatment completed one smoke and four held-out documents. Reported partial
values—control smoke structure .18 and treatment held-out structure .16535—are
conditioned on different completed subsets and must not be subtracted,
promoted, or used to reject the training objective.

Both pinned AgentV SDK bundles completed with zero execution errors, and both
ship-gate reports fail. In addition to the decode timeouts, this is fixture
volume and lacks adversarial, OOD, and RICO evidence. Neither checkpoint is
reusable, promoted, synced, or ship-ready. Lean is
`not_applicable:no_champion`: there is no confirmed optimum to prove.

The run also exposed a stage-ordering defect. Cycle 1796 landed on a promotion
cadence slot with no confirmed screening winner, yet the driver evaluated a
fresh rotating arm on held-out promotion suites. Campaign harness v94 makes a
promotion slot an opportunity rather than authority: when the queue has no
prior screening winner, the cycle falls back to diagnostic smoke-only
screening. Exact frozen replays remain exact, so this comparison still requires
one bounded replay after the repair; a repeat runtime failure is a terminal
runtime disposition, not evidence manufactured from partial rows.

Next priority: replay this exact semantic-contrast pair once. If it completes,
score the matched result. If it repeats the typed decode failure, close the
candidate as a runtime-incompatible approach and return to a fresh smoke-only
screening hypothesis. Promotion and Lean preflight stay closed until a fresh
screening result is confirmed.

Machine evidence:
[`autotrain-cycle-1796-semantic-contrast-incomplete.json`](autotrain-cycle-1796-semantic-contrast-incomplete.json).
