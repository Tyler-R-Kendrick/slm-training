# E238 — binder-reference arity (confounded)

Date: 2026-07-16
Status: completed; training comparison invalidated; checkpoint not promotable or ship

E238 adds grammar-derived reference-count supervision per binder declaration
and scores compiler completion paths only when alternatives differ between
emitting another legal binder reference and stopping. It contains no
component-name, fixture, or literal-layout cases.

Immediately before training, the isolated branch fetched and rebased onto
`origin/main` at `63856c0`, was clean, and was zero commits behind. The matched
126-row E230 recipe used CPU, 32 steps, batch 4, learning rate 0.0003, seed 0,
frozen local SmolLM2-135M, exhaustive compiler CE/margin 1.0, role-plan weight
1.0, arity loss/decode weight 1.0, all edge/topology/binder-component weights
0, capacity-aware sampling, honest train-only context, and no checkpoint sync.

Training took 158.83 s. Arity loss fell 4.4211 → 2.9895 and sampled-batch
accuracy rose 0 → 0.2647 over 17 → 34 declaration rows. Final total loss was
25.1978. Trace: `976c4bc6e0e8ed011cdd629649ef828f`; checkpoint SHA:
`2f9accc592787d93b0462e06b4ec8db9f40b83fe4e2292e0e3a60595fe3ff7a4`.

Strict evaluation regressed: syntax was 0.6667/0.6000/0.7500/0.5000/0.6667
and meaningful-program rate was 0 on smoke/held-out/adversarial/OOD/RICO.
Ten thresholds failed and AgentV passed 0/5. Arity scoring was applied 708
times and changed three choices; p95 latency was 41–43 seconds. Evaluation
trace: `787bf5856083ba125554cecf589caef2`.

The decode-off ablation had identical quality aggregates and gates. It proves
the three arity rerankings did not improve suite quality, but it does not rescue
the training comparison. Ablation trace: `711e8d0c83152c45072ecca4fe4ae98f`.

Post-run audit found a harness confound: optional heads consumed global Torch
RNG during initialization. Enabling an auxiliary head therefore changed later
masking/dropout draws even when its context and loss were isolated. A direct
state-hash reproduction differed with arity off versus on. Follow-up matched
runs also exposed coupling through combined backward traversal, optimizer
layout, and global gradient clipping. Auxiliary modules now initialize under
isolated stable seeds, backpropagate detached losses separately, use independent
optimizer groups, and clip each group independently. E238 is invalid evidence
about the arity training objective and must not be compared causally with prior
checkpoints. E239 confirms all 104 shared tensors are bit-exact after these
generalized corrections.

Machine-readable evidence:
[iter-e238-binder-arity-confounded-20260716.json](iter-e238-binder-arity-confounded-20260716.json).
