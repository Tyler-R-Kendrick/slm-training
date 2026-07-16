# Minimal OpenUI output contract rebuild (2026-07-15)

The committed `remediated_roots_judged` training snapshot now teaches the
smallest valid OpenUI surface for each language-contract unit. This is a
deterministic corpus rebuild, not a train/eval run, checkpoint, promotion, or
ship claim.

The canonical target remains `ExampleRecord.openui`. New `target_kind`,
`target_category`, and `accepted_outputs` fields distinguish lexical,
expression, statement, and full-document answers. Only the shortest answer is
used as supervised gold. Accepted alternatives are available to evaluation and
preference scoring without duplicating SFT labels.

For the boolean unit, the gold is the one-symbol output `true`; `false` and the
valid rooted `Separator` documents are also correct. Correctness is scored
independently from lexical efficiency, with the diagnostic composite fixed at
`0.8 * correctness + 0.2 * efficiency`. A correct full document therefore
remains correct but scores below its one-symbol equivalent.

## Rebuild result

| Measure | Before | After |
| --- | ---: | ---: |
| Total committed records | 498 | 255 |
| Legacy verbose language-contract derivatives | 305 | 0 |
| Canonical language-contract records | 0 | 62 |
| Verifier / quality rejects | n/a | 0 / 0 |

The 62 canonical records contain 4 lexical, 56 expression, 1 statement, and 1
document target. Their primary golds total 555 compiler-derived output symbols
(mean 8.9516). The sole document target is the forward-reference construct,
which cannot be represented faithfully as an isolated fragment. Comments stay
validator-only because the output lexer intentionally discards them.

The rebuild used CPU-only deterministic synthesis with no augmentation,
repairs, edit derivatives, frontier artifacts, or contrastive DESIGN.md rows.
Its content fingerprint is
`2f5a2f0c2def4b7e5032a9c0b7fc871c6f0b95bf5f102ce94ead78a5115752d9`.
Machine-readable evidence is in
[iter-minimal-output-contract-20260715.json](iter-minimal-output-contract-20260715.json).

Document parse, fidelity, and ship metrics continue to use only document
records. Fragment correctness and efficiency are diagnostic additions and do
not weaken existing ship gates.
