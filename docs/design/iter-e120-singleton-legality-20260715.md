# E120 constrained singleton legality (2026-07-15)

E120 tested the hypothesis that candidate probing can create false grammar dead
ends when the decoder already has one legal token. The decoder now accepts a
singleton grammar token without probing it, but only after special-token,
semantic, slot-contract, and completion guards pass.

The investigation also fixed three interactions exposed by that path:

- BOS/EOS framing tokens no longer contaminate the incremental DFA surface.
- Native slot contracts restrict only `<SYM_i>` tokens; they cannot replace
  forced punctuation or root components.
- `root` must begin with a component, and `)` is rejected for any component
  that still has missing or excess required arguments.

An eight-step CPU scratch run over 1,417 `v2_prompt_contract` records completed
with final loss **37.5242**. Before the full legality ordering fix, a strict
three-record `rico_held` diagnostic produced `$s44 = TextArea` for every row,
with parse **0.0** despite zero constrained dead ends. After the fix, a bounded
one-record diagnostic forced `root = Form(`, used no fallback, and recorded no
probe dead end. It still failed parse at the 64-token cap because the eight-step
model did not finish a semantically valid `Form`; parse **0.0** is therefore the
correct metric, not a parser bug.

This is a harness/decoder correction and a negative scratch-model result. It is
not a ship claim. Durable AgentEvals JSONL and AgentV artifacts are under
`outputs/runs/iter-e120-unsandboxed-20260715/e120_required_arity/agentv/`.

Recipe: CPU, scratch context, lexer output tokenizer, 8 train steps, batch 4,
LTR primary + repair, honest visible slot contract, constrained slot decode,
64-token cap, no DESIGN.md context, no unconstrained fallback, `rico_held n=1`
diagnostic subset, and explicit `--no-sync-checkpoints` scratch policy.
