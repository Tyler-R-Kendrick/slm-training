# Autotrain c1787: binder-arity CLI harness failure

**Verdict:** inconclusive infrastructure failure. The governed matrix selected
the intended `binder-arity` quality arm, but both the size-matched control and
candidate stopped in `train_model` argument parsing before training began. No
model metric, checkpoint, AgentV bundle, or ship-gate result exists for this
cycle.

| Arm | Frozen recipe | Result | Interpretation |
| --- | --- | --- | --- |
| control | CPU scratch; seed 101787; 22 steps requested; `structural_aux_head_profile=binder-arity`; binder loss/decode weights 0 | exit 2; no scoreboard | harness failure, not baseline evidence |
| binder-arity | CPU scratch; seed 101787; 22 steps requested; size-matched profile; governed binder-arity train/decode weights | exit 2; no scoreboard | harness failure, not candidate evidence |

The exact error was that `scripts.train_model` rejected `binder-arity` as an
invalid `--structural-aux-head-profile` choice, even though the canonical
model-build config/factory and continuous matrix already implement that
profile. The same incomplete CLI enum would also have rejected the registered
`binder-component-plan` successor. The repair extends the shared CLI boundary
and adds external declarative cases proving that all three binder profiles
reach `ModelBuildConfig`.

The preregistered comparison remains frozen under aggregate manifest digest
`56e8c805a46903bf4b3f514e6ba73f1d099beed5fb52acdbc744442718ba4c98`.
The next cycle must replay this exact control/candidate pair after the repair;
it must not change the seed, steps, arms, primary metric, or stopping rule in
response to this failure. Priority remains binder-reference F1 and certified
structural similarity under parse and latency constraints.

Lean is `not_applicable:infrastructure_failure_no_checkpoint`: no model was
trained and there is no champion or formal promotion target. The repository's
Lean build and formal-contract audit remain required validation for the
harness repair.

Machine evidence:
[`autotrain-cycle-1787-binder-arity-cli-harness-failure.json`](autotrain-cycle-1787-binder-arity-cli-harness-failure.json).
