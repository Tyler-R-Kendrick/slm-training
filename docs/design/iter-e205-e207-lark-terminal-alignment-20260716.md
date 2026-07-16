# E205–E207 — Lark-terminal alignment and schema enum paths

Status: **diagnostic only; no checkpoint promoted; ship gates not run**.

## Hypothesis and audit

E204 repeatedly preferred comma over the grammar-valid `]`. The existing
stratified objective collapsed every punctuation branch into one `struct` kind,
so one random structural row represented list closure, extension, call closure,
and newline decisions.

A deterministic audit of the first 64 committed E177 records found 1,211 gold
compiler decisions, including 134 Lark `RSQB` closures and 69 `COMMA`
extensions. It also found 202 `RPAR` and 173 newline decisions. The exhaustive
496-row replay was stopped after exceeding five minutes without partial output;
64 rows match the maximum row exposure of the 32-step batch-4 train and keep the
audit bounded and reproducible.

E205 classifies structural gold decisions by the active terminal names returned
by Lark. It does not inspect prompts, component arrangements, or known outputs.

## Matched train

E205 used the exact E201 data hashes and recipe: CPU, 32 steps, batch 4, seed 0,
frozen SmolLM2-135M context, lexer output, schema/slot context, no DESIGN.md
context, alignment weight 1.0, and no checkpoint sync. Only the parser-derived
alignment strata changed.

| Last loss | Wall s | Aligned rows | Root component | Bound component | Comma | Newline | RPAR | RSQB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8.1997 | 104.24 | 1,124 | 128 | 126 | 128 | 128 | 128 | 128 |

Alignment loss fell from 69.9854 to 2.8458 (mean 12.6827). Complete hashes and
telemetry are in
[the result JSON](iter-e205-e207-lark-terminal-alignment-20260716.json).

## Evaluations

Both rows are strict one-example smoke diagnostics with no unconstrained
fallback. Each emitted AgentEvals JSONL and an AgentV SDK bundle.

| Experiment | Change | Syntax | Meaningful parse | Structure | Component recall | Compiler fallback | Tokens | p50 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E206 | Lark-terminal-aligned checkpoint | 0.0 | 0.0 | 0.1917 | 0.0 | 1 | 14 | 4019.58 |
| E207 | Generated enum token sequences + lexical framing | 1.0 | 0.0 | 0.3125 | 0.0 | 0 | 70 | 6118.08 |

E206 closes the root children list after two binders: `]` beats comma by 3.78
log-score, reversing the E204 runaway. It then extends optional `Stack`
arguments into an enum absent from the fixed-token vocabulary, producing an
empty compiler forest.

E207 encodes every generated-schema enum through the tokenizer as an exact
completion path. Fixed atoms remain one token; arbitrary enum strings use the
typed `LIT_STR` + byte + `LIT_END` channel. Those framing symbols now advance
Lark as quote lexemes, so valid partial lexical symbols remain reachable. The
fallback disappears and syntax reaches 1.0.

The resulting program assigns both referenced binders to empty `Stack`
components. It is correctly rejected as `empty_root_stack`: meaningful parse,
component recall, placeholders, and reward remain zero. This is not a ship
result and E205 is not promotable.

## Next hypothesis

Use generated reference roles and gold AST child occupancy to supervise
empty-versus-populated collection decisions. The implementation must derive
roles from parser/schema/AST state, never observed component or prompt strings.
