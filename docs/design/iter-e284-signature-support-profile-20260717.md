# E284 — admitted-corpus signature profile

Date: 2026-07-17
Status: **completed; training blocked by objective-signature conflicts**

E284 reruns E276's frozen-parent safety profile on the admitted E283 corpus.
The recipe is unchanged: CPU, E228 checkpoint, `ftpo_set`,
legal-token-conditioned probability metrics, unit-normalized train gradients,
decision-kind train strata, decision-signature held-out strata, and at most
5,000 deterministic minimum-norm iterations. No optimizer or checkpoint
mutation occurred.

The input is the committed
`e283_signature_support_repair_v1` corpus: 372 independently evidenced events
(311 train / 61 held-out). E283's admission gate covers stable support
signatures based on decision kind, legal tokens, and judged positive tokens.
E284 additionally profiles objective signatures, which include the sampled
bad-token set because it changes the actual FTPO gradient.

## Result

The normalized kind-level train direction remains internally feasible:

| Measure | Result |
| --- | ---: |
| Train guard objectives | 64 (63 active) |
| Train regressions | 0 |
| Minimum-norm `norm_sq` | 0.000479761 |
| Minimum active-task dot | 0.000398168 |
| Solver iterations | 5,000 |
| Duality gap | 0.000087755 |
| Solver converged | No |
| Common train descent | Yes |

The held-out objective profile is unsafe:

| Measure | Result |
| --- | ---: |
| Train objective signatures | 93 |
| Held-out objective signatures | 26 |
| Exact held-out signatures present in train | 20 |
| Exact held-out signatures absent from train | 6 |
| Signatures with train-count deficit | 7 |
| Held-out guard objectives | 104 |
| Held-out regressions | **35** |
| Regressing signatures | 13 |

Regressions by decision kind are: bound-child references 14, literals 9,
bound components 7, root-child references 3, and populated bound brackets 2.
By metric they are: bad probability mass 11, good probability mass 9, mean
margin 8, and loss 7.

The six objective signatures absent from train are four bound-component
signatures, one empty-bound-list signature, and one literal signature. One
additional bound-child signature has some train support but fewer train than
held-out events, so the report's multiset coverage lists seven deficits.

## Decision

Do not train the E283 corpus with the coarse kind-level direction. Stable
support admission was necessary and now works, but it is not sufficient for
objective geometry because independently judged legal alternatives can produce
different bad-token sets at the same grammar state.

The next read-only diagnostic should combine train objectives at the same
decision-signature granularity used for held-out evaluation. If that still
opposes covered held-out signatures, the data pipeline must enforce
objective-signature support or change the training objective to marginalize
over independently judged legal alternatives. Do not add token/component
special cases or increase training duration.

Machine-readable evidence:
[quality-matrix-v10-e284-signature-profile-results.json](quality-matrix-v10-e284-signature-profile-results.json).
