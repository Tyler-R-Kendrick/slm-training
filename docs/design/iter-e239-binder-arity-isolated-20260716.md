# E239 — isolated binder-reference arity

Date: 2026-07-16
Status: completed; hypothesis rejected; checkpoint not promotable or ship

E239 reruns the grammar-derived binder-reference arity objective after removing
four auxiliary-head confounds: module initialization uses isolated stable RNG,
the auxiliary loss has its own backward pass, auxiliary parameters have separate
optimizer groups, and each group is clipped independently. A matched no-arity
control proves the correction: all 104 shared checkpoint tensors are bit-exact;
only `binder_arity_head.weight` and `.bias` are candidate-only.

Three matched pairs were invalidated while establishing that invariant. E239
(`14136eb54ac5b303fa2691cf6db65b47` / `4568cc930806e400053767ca3ae41da9`)
confirmed that initialization isolation alone left optimizer/backward coupling. E239b
(`c70a75997036ab74c989e514db38297a` / `7a1d4befb6c354498589808164caffd0`)
retained shared-tensor drift after optimizer grouping, exposing combined
backward coupling. E239c
(`5638677e326fbd1c266caa5fbd53fa40` / `cbf3e5d4562f0ac900d10ff86f9f6b49`)
isolated backward and optimizer layout but still differed in 100/104 shared
tensors because global clipping included the auxiliary-head norm. These runs
are harness evidence only and must not be used as model comparisons.

Immediately before each training command, the branch fetched and rebased onto
`origin/main`, was clean, and was zero commits behind. Both runs used the
committed 126-row `e230_diverse_judged_roots_v2` corpus, CPU, 32 steps, batch 4,
learning rate 0.0003, seed 0, frozen local SmolLM2-135M, lexer output, exhaustive
compiler CE/margin 1.0, role-plan weight 1.0, capacity-aware sampling, honest
train-only context, and no checkpoint sync.

The candidate ran for 224.64 s. Arity loss fell 4.0988 → 2.4903 and sampled-batch
accuracy rose 0 → 0.4706 over 17 → 34 declaration rows. Final reported total
loss was 23.7655. Candidate trace: `03c38dcbbc9f967d58c6c03435e8253d`;
checkpoint SHA: `677e80efb4ae1585334ba9bb6741472d3780aa0d80f7dc1adc471aeeed86774d`.
The matched control trace was `48c9ab5f8c66ea1134f5cf861aed16c5`; its
checkpoint SHA was `456638b0e031dca4d1bbcb96f5785e287684a32982152aa885f09751110588bd`.

Strict evaluation used all five committed policy suites, honest slot contracts,
compiler-tree decode, no unconstrained fallback, and AgentV. Arity scoring was
applied 1,606 times and changed 29 choices. It improved smoke syntax from 0 to
0.3333 and smoke structure from 0.1591 to 0.2591, but meaningful-program rate
remained 0 on every suite. Both settings failed the same 11 thresholds and
AgentV passed 0/5. Placeholder fidelity and structure moved in mixed directions.

| Suite | n | Syntax on/off | Meaningful on/off | Fidelity on/off | Structure on/off | p95 ms on/off |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 0.3333 / 0 | 0 / 0 | 0.5833 / 0.8056 | 0.2591 / 0.1591 | 80,285 / 105,959 |
| held_out | 5 | 0.2000 / 0 | 0 / 0 | 0.5667 / 0.6000 | 0.1338 / 0.1394 | 85,493 / 86,670 |
| adversarial | 4 | 0 / 0 | 0 / 0 | 0.8333 / 0.8333 | 0.1912 / 0.1912 | 81,590 / 56,908 |
| ood | 4 | 0 / 0 | 0 / 0 | 0.5250 / 0.6083 | 0.1775 / 0.2066 | 83,893 / 48,105 |
| rico_held | 3 | 0 / 0 | 0 / 0 | 0.3750 / 0.3750 | 0.0971 / 0.0956 | 82,055 / 55,950 |

This rejects arity reranking as a sufficient semantic-quality lever. Retain the
grammar-derived mechanism and the generalized auxiliary-isolation invariants,
but do not promote the checkpoint. The next compiler experiment should address
the pathological long invalid trajectories and use a corrected greedy-tree
control before adding wider search.

Machine-readable evidence:
[iter-e239-binder-arity-isolated-20260716.json](iter-e239-binder-arity-isolated-20260716.json).
