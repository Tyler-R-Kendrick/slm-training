# Autotrain c1736: completion-bounds promotion-cadence screen

**Verdict:** `grammar_completion_bounds` is an exactly size-matched quality
null and a runtime regression. It changes neither smoke nor held-out quality,
raises p50 by 4.77% on smoke and 3.91% on held-out, and takes 1.35 times the
training wall. The diagnostic arm is rejected and is not checkpoint,
promotion, or ship evidence.

## Result matrix

| Arm | Params | Suite n | Parse | Binder F1 | Meaningful | Structure | Recall | AST node / edge F1 | p50 | Disposition |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,608,962 | smoke 3 / held 5 | 1.0 / 1.0 | 0 / 0 | 0 / 0 | .0575 / .0894 | 0 / .10 | 0 / 0; .10 / 0 | 2,933.21 / 3,058.12 ms | complete fixture control; gates fail |
| completion bounds | 1,608,962 | smoke 3 / held 5 | 1.0 / 1.0 | 0 / 0 | 0 / 0 | .0575 / .0894 | 0 / .10 | 0 / 0; .10 / 0 | 3,072.99 / 3,177.65 ms | exact quality null; 4.77% / 3.91% slower |

The arms also have identical loss 10.1437. Both execute 30 smoke and 54 held-out
neural forwards; their completion state, witness-expansion, and parser-fork
counters are identical. Training wall is 2.889 seconds for control and 3.886
seconds for bounds. These are 24-step CPU scratch checkpoints, seed 101736,
batch 2, with no checkpoint sync.

AgentV completed both suites for both arms with zero execution errors. Smoke
`n=3` and held-out `n=5` are below the evidence floor; adversarial, OOD, and
`rico_held` are absent. Meaning is zero on both suites, all honest ship gates
fail, and RL remains locked. The campaign formal status is
`not_applicable:no_champion`: no promoted candidate exists for a Lean band
certificate. The repository's full Lean proof suite was green immediately
before this run.

## Next-run priority

The recent structural-quality families and completion-bounds diagnostic are
exhausted. The next distinct registered runtime hypothesis is
`compact_active_canvas`: compare it with the same size-matched control and test
whether it lowers smoke p50 without lowering parse rate. This remains a
diagnostic experiment, not a claim that the model-quality problem is solved.

Machine-readable evidence is in
[`autotrain-cycle-1736-bounds-promotion-screen.json`](autotrain-cycle-1736-bounds-promotion-screen.json).
