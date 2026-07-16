# E240 — corrected greedy compiler-tree control

Date: 2026-07-16
Status: completed; control established; quality gates failed; no checkpoint created

E240 establishes the V9 control before enabling lattice rollback or trajectory
search. It evaluates the unchanged E228 parent with the compiler completion tree
as the authoritative legal-token layer and greedy model-score ordering. This row
is evaluation-only: it did not train, copy, modify, create, sync, or promote a
checkpoint.

The immutable parent was
`outputs/autoresearch/e228-candidate-margin-alignment/runs/e228-candidate-margin-matched/checkpoints/last.pt`
(SHA-256
`7a9be4a665e216d7f7e73883ad74ad972bbf30846896d0c29188d6482f5b093a`).
The committed train manifest was `e230_diverse_judged_roots_v2`, fingerprint
`9f72d85b6cc7118e0f69e010d0debdd2b40ede514e03178dded8e164daaae9bb`;
it was recorded for lineage but not consumed by this eval-only row. Evaluation
used CPU, local frozen HF context, seed 0, all five committed remediated suites,
honest slot contracts, schema and slot-contract context, compiler-tree decode,
greedy width 1, eight generation steps, final validation, and no unconstrained
fallback.

| Suite | n | Syntax | Meaningful | Fidelity | Structure | Reward | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.5278 | 0.4642 | 0.8073 | 3,692 | 6,136 |
| held_out | 5 | 1.0000 | 0 | 0.2800 | 0.3369 | 0.7330 | 3,485 | 4,318 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.5417 | 0.4744 | 0.8115 | 3,154 | 3,664 |
| ood | 4 | 1.0000 | 0 | 0.2583 | 0.3750 | 0.7265 | 3,507 | 4,237 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.1250 | 0.1628 | 0.6865 | 3,945 | 3,974 |

Four gates failed: smoke meaningful-program rate (0.3333 < 0.66), held-out
meaningful-program rate (0 < 0.40), OOD meaningful-program rate (0 < 0.25),
and RICO structural similarity (0.1628 < 0.20). AgentV passed 1/5 suite cases,
with zero execution errors and mean score 0.60.

The deterministic layer produced syntax-valid output on all 19 examples with
zero compiler fallbacks, unconstrained fallbacks, constrained dead ends, decode
timeouts, lattice bottoms, rollbacks, or abstentions. Aggregate
forwards/tokens/candidates were 73/113/1,105 (smoke), 115/185/2,068
(held-out), 99/151/1,447 (adversarial), 98/150/1,531 (OOD), and
69/111/1,275 (RICO). This is materially faster than E239's pathological
50–106 second p95 trajectories while preserving strict syntax validity.

Evidence-generation reruns needed to correct matrix summary persistence and add
the repository trace boundary reproduced every quality metric exactly; only
wall-clock latency varied. The final trace-backed run is canonical, evaluated at
`2026-07-16T21:38:01.128738+00:00`, trace
`28459197a7de6749b18ce2d6bda8f5a1`. Its AgentEvals payload uses the pinned
AgentV SDK contract.

E240 therefore does not trigger the control falsifier: corrected greedy-tree
decode reproduces valid behavior without fallback. It is still not promotable or
ship-ready. The subsequent matched V9 campaign confirmed the prediction from
this control: E241 bounded rollback was output-identical because no trigger
activated. Wider always-on search later regressed semantic quality, as recorded
in [lattice-recursive-search.md](lattice-recursive-search.md).

Machine-readable evidence:
[quality-matrix-v9-e240-results.json](quality-matrix-v9-e240-results.json).
