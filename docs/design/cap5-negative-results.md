# CAP5 negative-result registry

This table collects falsified, inconclusive, and abandoned CAP hypotheses. It is a
first-class deliverable, not a footnote. Each row links to the durable evidence that
produced the negative conclusion.

| ID | Hypothesis / lever | Expected effect | Observed outcome | Evidence | Verdict |
| --- | --- | --- | --- | --- | --- |
| CAP3-03-N1 | Equal-storage ternary latent codes improve semantic quality | Ternary matches or beats continuous/binary at equal bytes | All semantic metrics remain zero; no reliable advantage | [iter-cap3-03-ternary-falsification-20260718.md](iter-cap3-03-ternary-falsification-20260718.md) | falsified at tested budget |
| CAP4-02-N1 | Adaptive-plane routing reduces measured p95 latency before kernel support | Latency gain from compiler-floor + runtime-signal routing | Fixture demonstrates schedule feasibility; no optimized kernel ⇒ no measured speed claim | [iter-cap4-02-adaptive-plane-routing-20260718.md](iter-cap4-02-adaptive-plane-routing-20260718.md) | inconclusive for speed claim |
| CAP4-04-N1 | Compiler-routed block sparsity improves wall-clock cost without kernel support | Fewer operations translate to faster inference | Missing optimized sparse kernel prevents comparative speed claim | [iter-cap4-04-block-sparsity-20260718.md](iter-cap4-04-block-sparsity-20260718.md) | not promotable |
| CAP4-05-N1 | Quotient-state diffusion graph alone carries enough structure for valid generation | Topology mixing supports valid AST sampling | Graph connectivity passes but semantic gates fail; diffusion not sufficient alone | [iter-cap4-05-quotient-diffusion-graph-20260718.md](iter-cap4-05-quotient-diffusion-graph-20260718.md) | not promotable |
| CAP1-05-N1 | Template abstraction discards information needed for structural decisions | Replacing literals with template slots changes structural choice stream | 16-record audit: zero violations for the tested value classes | [cap1-05-template-sufficiency-20260718.md](cap1-05-template-sufficiency-20260718.md) | falsified as a failure mode (template sufficiency holds in tested scope) |
| EFS0-05-E175 | Retrieval k=4 improves meaningful quality | Meaningful parse increases with retrieval augmentation | Bounded syntax/parse regress to 0.0; rejected control | [iter-e175-retrieval-20260716.md](iter-e175-retrieval-20260716.md) | rejected |
| EFS0-05-E176 | Broad 1,417-record corpus improves quality | More data raises semantic metrics | Bounded syntax/parse regress to 0.0; rejected control | [iter-e176-broad-corpus-20260716.md](iter-e176-broad-corpus-20260716.md) | rejected |
| EFS0-05-E191 | Random all-branch compiler alignment improves root selection | Full alignment recovers root | Regresses root selection; meaningful parse 0.0 | [iter-e181-e194-compiler-alignment-20260716.md](iter-e181-e194-compiler-alignment-20260716.md) | rejected |
| EFS0-05-E236 | Binder-topology objective improves semantic decisions | Topology loss changes legal choices | Changes 0/38 applied choices; semantic metrics collapse | [iter-e236-binder-topology-20260716.md](iter-e236-binder-topology-20260716.md) | rejected |
| EFS0-05-E244 | Always-on PTRM decoder | Quality gain from persistent trigger telemetry | Strong negative control / decoder bug; retained as closed sentinel | [iter-efs0-05-rejected-lever-readjudication-20260719.md](iter-efs0-05-rejected-lever-readjudication-20260719.md) | closed |

## Infeasible mathematical arms

* `[q=6, d=4]` for the toy robust-coding requirement: no exact code construction is
  claimed; the MDS and Hamming constructions are verified only for their declared
  parameters.
* Ternary `d=6` for the toy robust requirement: not claimed; the exact constructions
  are `[4,2,3]_7` and `[7,4,3]_3`.

## Abandoned hypotheses

* Unfrozen full SmolLM2 context (E174): regressed bounded syntax to 0.0.
* Task-balanced exposure without capacity awareness (E221): effective exposure
  29.68/128, nine gates fail.
* Quota-capacity sampler without semantic retention (E223): task quotas correct but
  all semantic metrics zero.
* Safe / stratified / block-coordinate / projected / MGDA set-valued FTPO
  (E265–E269, E272): every scale regresses held-out per-kind metrics; parent
  restored bit-identically.

See also the full rejected-lever registry:
[iter-efs0-05-rejected-lever-readjudication-20260719.md](iter-efs0-05-rejected-lever-readjudication-20260719.md).
