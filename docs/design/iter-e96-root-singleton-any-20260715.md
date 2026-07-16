# E96 root singleton fast path (2026-07-15)

E96 generalized the root singleton bypass to the lexer tokenizer. At an empty
prefix, the decoder's root invariant makes `<BIND_0>` the only semantically
legal first token even though the surface DFA exposes broad terminals.

The focused grammar suite passed (`27` tests). The strict E91 checkpoint smoke
diagnostic was unchanged: parse `0.0`, raw syntax `0.0`, structural similarity
`0.5333`, contract precision/recall `0.8/1.0`, one constrained dead end, and
zero fallback/template emissions. This means the observed dead end is in a
different constrained decode path than this picker, or the root token is being
rejected before this picker is reached.

Decision: retain the semantics and regression coverage because it prevents a
real false-dead-end class in this picker, but do not claim an evaluation gain.
Next step is to trace the LTR caller/path that records the position-1 dead end
and identify where its candidate set is constructed.
