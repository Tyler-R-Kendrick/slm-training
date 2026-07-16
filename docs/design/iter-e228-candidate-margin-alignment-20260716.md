# E228 — legal-candidate margin alignment

Status: **matched training improved honest quality; four ship gates still fail;
checkpoint not promoted**.

E227 showed that cross-entropy over grammar-legal candidates optimized but still
allowed the empty-list branch to narrowly outrank populated-child choices. E228
adds a configurable positive margin between each gold candidate and the strongest
legal alternative. Candidate inventories come only from the Lark/compiler forest;
the margin defaults off, singleton decisions bypass inference, and no literal or
component-specific cases were added.

Immediately before training, `origin/main` was fetched and found one commit ahead.
E228 rebased cleanly with no conflicts, then passed 69 focused tests and lint. The
matched recipe retained E227's canonical judged E218 data, quota-capacity sampler,
32 CPU steps, batch 4, learning rate 0.0003, seed 0, frozen local SmolLM2 context,
lexer output, schema and slot context, no DESIGN context, no fallback, and no
checkpoint sync. The sole experiment knob was
`compiler_alignment_margin=1.0`.

The margin mechanism executed: violation rate fell from 0.9130 at step 1 to
0.5636 at step 32; margin loss fell from 16.1754 to 2.2001 and candidate CE from
15.3994 to 2.0318. Total loss ended at 14.6153 after 22,924 prompt and 6,401
target tokens in 122.80 s; trace ID `4e45de187ef22cbbe9178656308d6761`.

| Suite | n | syntax | meaningful | structure | component recall | fidelity | contract precision | reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.3333 | 0.4642 | 0.2500 | 0.5278 | 1.0000 | 0.8073 |
| held_out | 5 | 1.0000 | 0.0000 | 0.3369 | 0.1567 | 0.2800 | 1.0000 | 0.7330 |
| adversarial | 4 | 1.0000 | 0.5000 | 0.4744 | 0.4583 | 0.5417 | 1.0000 | 0.8115 |
| ood | 4 | 1.0000 | 0.0000 | 0.3750 | 0.2083 | 0.2583 | 1.0000 | 0.7265 |
| rico_held | 3 | 1.0000 | 0.6667 | 0.1628 | 0.4444 | 0.1250 | 1.0000 | 0.6865 |

Syntax and contract precision are 1.0 on every suite with zero fallback. Honest
failures dropped from E227's 12 and E226's five to four: smoke, held-out, and OOD
meaningful-program rate plus RICO structure. AgentV remains 1/5 with no execution
errors. E228 is therefore the best current diagnostic checkpoint but is not ship
or promotion evidence. The remaining 56.4% training margin-violation rate and
low component recall justify a matched continuation/scale check before changing
the grammar or data again.

Machine-readable evidence:
[iter-e228-candidate-margin-alignment-20260716.json](iter-e228-candidate-margin-alignment-20260716.json).
