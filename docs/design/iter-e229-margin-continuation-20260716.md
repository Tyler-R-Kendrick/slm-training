# E229 — margin continuation and lexical-frame correction

Status: **bit-exact continuation completed; corrected honest evaluation saturates
at four failed gates; checkpoint rejected; deterministic lexical bug fixed**.

E228 improved semantic topology while its training margin-violation rate remained
0.5636. E229 tested whether the same objective needed more semantic branch
optimization, not whether training could repair parsing. It resumed the complete
E228 optimizer/RNG/sampler state from step 32 to total step 64.

Immediately before launch, `origin/main` was fetched; branch and remote were
identical with a clean worktree. The recipe retained canonical judged E218 data,
quota-capacity sampling, batch 4, learning rate 0.0003, seed 0, frozen local
SmolLM2 context, lexer output, unit candidate CE plus margin 1.0, schema and slot
context, no DESIGN context, no fallback, and no checkpoint sync. Cumulative
exposure reached 46,730 prompt and 13,076 target tokens. Step-64 loss was 9.4505;
margin loss was 1.2135 with violation rate 0.6140. The continuation consumed
132.44 s; trace ID `6db2bd144773feefc1764dae72fbc59d`.

The first evaluation exposed syntax below 1.0 despite compiler-tree decoding.
This was not treated as a model/training failure. Lexer-native `LIT_STR` renders
as a quote to Lark before its required `BYTE* + LIT_END` frame is complete. The
terminal mapper therefore admitted quote-equivalent opener, symbol, and closer
tokens inside one literal, producing adjacent strings and excess component args.
The generalized fix tracks literal frame state from token IDs/kinds and restricts
the open frame to byte tokens plus `LIT_END`, shared by compiler-tree and ordinary
constrained selection. No output, component, or prompt literals are matched.

Corrected evaluation of the unchanged checkpoint:

| Suite | n | syntax | meaningful | structure | component recall | fidelity | contract precision | reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.4475 | 0.1667 | 0.5556 | 0.6667 | 0.6073 |
| held_out | 5 | 1.0000 | 0.0000 | 0.3564 | 0.1567 | 0.5600 | 1.0000 | 0.8290 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.4387 | 0.4583 | 0.8333 | 1.0000 | 0.9110 |
| ood | 4 | 1.0000 | 0.0000 | 0.3481 | 0.1458 | 0.5583 | 1.0000 | 0.8285 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.1720 | 0.4444 | 0.2500 | 1.0000 | 0.7360 |

The deterministic syntax invariant is restored across all suites with zero
fallback. The same four quality gates still fail and AgentV remains 1/5. Compared
with E228, smoke structure/recall/reward and OOD structure/recall regress while
RICO remains below its structure gate. More steps on this recipe are therefore
falsified; E228 remains the better diagnostic checkpoint. The next train should
change semantic component-ranking supervision or data coverage, not parsing,
grammar literals, or duration.

Machine-readable evidence:
[iter-e229-margin-continuation-20260716.json](iter-e229-margin-continuation-20260716.json).
