# E224 — semantic-exhaustive alignment diagnostic

Status: **training and strict evaluation completed; alignment path verified;
hypothesis falsified; 12 ship gates failed; no checkpoint promoted**.

E223 achieved deterministic syntax and balanced high-exposure sampling but zero
meaningful parse, component recall, fidelity, and reward. Its stratified compiler
alignment retained only one random state per decision kind per record. E224
tested whether every grammar-derived AST-role decision should be retained while
structural serialization decisions remain one-per-kind. AST roles come from
tokenizer categories (`component`, `bind`, `state`, `builtin`), not component
names, prompts, or literal output matches.

Latest `main` was fetched immediately before launch; the clean branch was zero
commits behind. The five-candidate matrix selected the exact E223 recipe with
only `compiler_alignment_semantic_exhaustive=true`: canonical E218 data,
quota-capacity sampling, 32 CPU steps, batch 4, learning rate 0.0003, seed 0,
frozen local HF context, lexer output, unit alignment weight, schema and slot
context, no DESIGN context, no checkpoint sync, strict tree decode, and
unconstrained fallback disabled.

The knob materially executed:

| Run | total alignment rows | AST-role rows | last loss |
| --- | ---: | ---: | ---: |
| E223 one-per-kind | 1,421 | 643 | 11.9060 |
| E224 semantic exhaustive | 1,901 | 1,123 | 15.9786 |

AST-role coverage rose 74.7% and checkpoint SHA changed to
`c9f38df1…22bb8ef`, so the result is not a silent no-op. Training consumed the
same 22,924 prompt and 6,401 target tokens in 158.53 s; trace ID is
`391a0d48746b259ab3f9b1a6208b9907`. Exposure exactly matched E223 at 103 unique
and 81.11 effective records from 128 draws, with maximum repeat four.

The strict five-suite scoreboard was exactly unchanged from E223:

| Suite | n | syntax | meaningful parse | structure | component recall | fidelity | reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 3 | 1.0000 | 0.0000 | 0.3094 | 0.0000 | 0.0000 | 0.0000 |
| held_out | 5 | 1.0000 | 0.0000 | 0.2514 | 0.0000 | 0.0000 | 0.0000 |
| adversarial | 4 | 1.0000 | 0.0000 | 0.2905 | 0.0000 | 0.0000 | 0.0000 |
| ood | 4 | 1.0000 | 0.0000 | 0.2369 | 0.0000 | 0.0000 | 0.0000 |
| rico_held | 3 | 1.0000 | 0.0000 | 0.0901 | 0.0000 | 0.0000 | 0.0000 |

Every suite had zero fallback and constrained-fallback rate. Twelve gates failed
and AgentV passed 0/5 with no execution errors. Increasing the number of
independent AST-role alignment states is therefore falsified at unit weight: it
raises optimization load without changing decoded semantics. The next diagnosis
must test whether alignment losses are coupled across related AST decisions or
whether the decode objective can express multi-decision semantic structure;
another per-state quantity increase is not justified.
