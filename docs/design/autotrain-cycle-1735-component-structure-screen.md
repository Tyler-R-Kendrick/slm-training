# Autotrain c1735: component-structure screen

**Verdict:** joint component-plan and component-edge supervision is an exactly
size-matched quality null and a cost regression. It changes no measured quality
or deterministic-work counter, raises p50 by 5.30%, and takes 3.04 times the
training wall. The arm is rejected and is not checkpoint, promotion, or ship
evidence.

## Result matrix

| Arm | Params | n | Parse | Binder F1 | Meaningful | Structure | Recall | AST node / edge F1 | p50 | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,913,789 | 3 | 1.0 | .6333 | .3333 | .17417 | .25 | .26190 / 0 | 1,509.10 ms | complete fixture control; gates fail |
| component-plan + edge loss `1.0` | 1,913,789 | 3 | 1.0 | .6333 | .3333 | .17417 | .25 | .26190 / 0 | 1,589.07 ms | exact quality null; 5.30% slower |

Both arms also match placeholder fidelity .5278, reward .7653, 11 neural
forwards, 31,118 unique completion states, 861 witness expansions, and 32,126
parser forks. Candidate loss is worse (24.2103 vs 18.6822), and training wall is
8.813 vs 2.904 seconds. These are 23-step CPU scratch checkpoints, seed 101735,
batch 2, with both structural heads prebuilt in both arms.

AgentV completed without execution errors. Smoke `n=3` is below the evidence
floor; held-out, adversarial, OOD, and `rico_held` were not run. Lean is
`not_applicable:screening`; promotion and RL remain locked.

## Harness signal and repaired next priority

The cross-cycle cooldown correctly exhausted all recent structural-quality
families and selected `bounds`, but the first terminal projection generically
called it a “quality hypothesis.” Bounds is a registered latency diagnostic.
The repair now derives the successor area and wording from the selected arm's
actual preregistered hypothesis. The terminal signal is therefore:

> Area `experiments`: run the `grammar_completion_bounds` latency diagnostic
> versus the matched control without lowering parse rate.

This is a diagnostic transition, not evidence that the model quality problem is
solved. Machine-readable evidence is in
[`autotrain-cycle-1735-component-structure-screen.json`](autotrain-cycle-1735-component-structure-screen.json).
