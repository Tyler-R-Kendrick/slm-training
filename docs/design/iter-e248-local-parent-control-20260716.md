# E248 — exact-state preference parent control

Date: 2026-07-16
Status: completed; matched control established; four gates failed; no checkpoint created

E248 evaluates the unchanged E228 parent before any V10 local-preference update.
It is evaluation-only and did not train, copy, modify, create, sync, or promote a
checkpoint. Parent SHA-256:
`7a9be4a665e216d7f7e73883ad74ad972bbf30846896d0c29188d6482f5b093a`.
The committed E230 train manifest fingerprint
`9f72d85b6cc7118e0f69e010d0debdd2b40ede514e03178dded8e164daaae9bb`
is lineage metadata only; no training data was consumed.

The first attempt was invalidated because V10 duplicated and omitted parts of
the canonical compiler-tree policy. Despite tree decode, it disabled schema and
slot context, honest slot contracts, final validation, and fallback prohibition;
syntax consequently measured 0 on all suites. This was a harness-policy failure,
not model or training evidence. V9 and V10 now share one strict compiler-tree
policy function, with a regression test requiring field-for-field parity.

The corrected run used CPU, local frozen HF context, seed 0, all five committed
remediated suites, schema and slot-contract context, honest slot contracts,
compiler-tree greedy decode, eight generation steps, final validation, and no
unconstrained fallback.

| Suite | n | Syntax | Meaningful | Fidelity | Structure | Reward | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.5278 | 0.4642 | 0.8073 | 2,844 | 6,358 |
| held_out | 5 | 1.0000 | 0 | 0.2800 | 0.3369 | 0.7330 | 2,634 | 3,069 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.5417 | 0.4744 | 0.8115 | 2,414 | 2,899 |
| ood | 4 | 1.0000 | 0 | 0.2583 | 0.3750 | 0.7265 | 3,424 | 4,231 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.1250 | 0.1628 | 0.6865 | 3,206 | 3,336 |

Smoke, held-out, and OOD meaningful-program gates failed; RICO structural
similarity was 0.1628 against 0.20. AgentV passed 1/5 with zero execution errors
and mean score 0.60. All 19 examples were syntax-valid with zero compiler or
unconstrained fallback and zero decode timeout. The scores exactly reproduce the
strict E240 control, establishing the V10 baseline but no readiness claim.

Canonical evaluation time: `2026-07-16T21:53:23.514971+00:00`; trace:
`543d294b44d54415a1fc8b73cc5725ba`.

Machine-readable evidence:
[quality-matrix-v10-e248-results.json](quality-matrix-v10-e248-results.json).
