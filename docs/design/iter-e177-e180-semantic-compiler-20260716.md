# E177–E180 semantic admission and compiler ownership (2026-07-16)

E177 changed one data lever: the independent judge now derives ordinary component
mentions from the official component schema, uses structured edit instructions,
and compares repair outputs with their generated clean AST. An invalid preflight
snapshot over-read embedded programs and was discarded before training. The
corrected immutable corpus retains 496 of 498 records and rejects two actual
prompt/output mismatches (`train_form_01` and `train_gallery_01`). It is committed
under `src/slm_training/resources/train_data/e177_semantic_judge_v2`, including
records, manifest, stats, mixture, synthesis telemetry, and governance.

The matched E177 train used 32 CPU steps, seed 0, frozen SmolLM2-135M context,
lexer output, schema/slot context, no DESIGN.md context, and no checkpoint sync.
Loss was 12.2220 versus E173's 11.0876. The bounded probe still failed syntax
because the final output omitted a required `SelectItem.label`.

E178–E180 reused the exact E177 checkpoint and changed only deterministic decode:

| Iteration | Isolated change | Syntax | Meaningful parse | Struct | Reward | p50 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E177 | judged corpus | 0.0000 | 0.0000 | 0.0583 | 0.000 | 7538.12 |
| E178 | generated-schema required/max arity | 0.0000 | 0.0000 | 0.0583 | 0.000 | 7168.36 |
| E179 | compiler output no longer overwritten by legacy repair | 0.0000 | 0.0000 | 0.0000 | 0.000 | 3421.32 |
| E180 | symbolic root binding restricted to generated component kinds | 1.0000 | 0.0000 | 0.1542 | 0.607 | 3712.78 |

E180 produces the valid document `root = TextContent(":hero.title")` in seven
tokens with five denoiser forwards and no full-vocabulary projections. The
reported `parse_rate=0` is therefore not a syntax failure: this evaluator requires
a meaningful program, and component recall is only 0.25. The remaining failure is
semantic component selection. The next matched lever is balanced, judge-gated
prompt-to-component role supervision; syntax gates and ship thresholds remain
unchanged.

Evidence: [result JSON](iter-e177-e180-semantic-compiler-20260716.json), E177
train telemetry and summary paths recorded there, four persisted eval bundles,
and their pinned AgentV JSONL paths. All evaluations are one-record diagnostics,
not full smoke or ship evaluations.
