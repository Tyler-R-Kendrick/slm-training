# Autotrain c1764: tail-weighted literal-close arm is quality-null

**Verdict:** the preregistered `literal-close` arm completed, but it does not
improve OpenUI quality over its size-matched control. Both CPU scratch arms
trained 20 steps on `wf_smoke_v2` with seed 101764, batch size 2, and exactly
1,608,962 trainable parameters. The only treatment difference was
`ltr_tail_loss_weight=2.0` versus 0.0. All eight selected documents completed,
both AgentV bundles finalized without execution errors, and both arms failed
honest ship gates.

| Arm | Params | Last weighted loss | Smoke | Held-out | AgentV | Decision |
| --- | ---: | ---: | --- | --- | --- | --- |
| literal-close, tail weight 2 | 1,608,962 | 26.5681 | n=3; parse 1; meaning .3333; structure .17417; binder F1 .6333; p50 1,167.26 ms | n=5; parse 1; meaning 0; structure .09758; binder F1 .43714; p50 1,150.26 ms | 0/2; 0 execution errors | rejected quality null |
| matched control, tail weight 0 | 1,608,962 | 13.7285 | n=3; parse 1; meaning .3333; structure .17417; binder F1 .6333; p50 1,189.39 ms | n=5; parse 1; meaning 0; structure .09758; binder F1 .43714; p50 1,153.93 ms | 0/2; 0 execution errors | complete gate rejection |

The lever was active: the train summaries record different tail weights, the
last weighted losses differ, and checkpoint SHA-256 digests differ. However,
the eight generated programs are prediction-identical across arms and every
quality metric ties exactly. The 1.86% smoke and 0.32% held-out p50 differences
are below the five-percent efficiency floor and cannot rescue a quality-null
arm. The larger candidate loss is not directly comparable as raw optimization
quality because its objective deliberately carries additional weight.

This result also narrows the c1763 diagnosis. c1764 has no decode timeouts in
either arm, so the numeric-literal stall is not reproduced at seed 101764,
20 steps, and batch size 2. Because those recipe dimensions differ from the
c1763 frozen batch-size-one checkpoint, the recovery cannot be attributed to
tail weighting. The observed c1763 stall remains real, but its causal family is
training-realization sensitive rather than a universally active literal-close
defect.

The next useful hypothesis should target legal transition ranking directly and
measure it before end-to-end decoding: add a grammar-derived closure-vs-byte
margin diagnostic/training objective over numeric literal states, then require
an improved close-token rank on a held-out transition suite before another
full decode comparison. This is more informative than a generic component-plan
arm for this failure family. It must remain size-matched and grammar-constrained;
no timeout increase or I6 weakening is authorized.

Both checkpoints are local, explicit no-sync diagnostics. Neither is reusable,
promoted, or ship evidence. Lean is `not_applicable:no_champion`; promotion
formal proofs remain mandatory when a champion exists.

Machine evidence:
[`autotrain-cycle-1764-literal-close-null.json`](autotrain-cycle-1764-literal-close-null.json).
