# E546 — strict-subset root-reference sampling

E546 adds deterministic oversampling for records whose terminal root references
a nonempty strict subset of generated sections. Classification comes from the
choice tokenizer and grammar state, not record IDs. The canonical 244-record
corpus and every target, loss, and gate remain unchanged.

Matched clean 24-step CPU HF-context continuations start from E544. The
multiplier-1 control sees 7 negative-target rows, processes 1,270 target tokens
in 29.10 seconds, and writes SHA `46aba9048624f766e6052d202a94b689440baca9f1ab94d8d6c8d48adc40fc55`.
Multiplier 5 expands the sampling view from 244 to 412 entries, sees 22
negative-target rows, processes 1,318 target tokens in 30.50 seconds, and writes
SHA `a1a6bfc94108a8bba9aac18e5570d70e317cdec5bb706f126bf47e67e2b4efe2`.
Both runs use `max_wall_minutes=3` and explicit no-sync scratch persistence.

The treatment improves early exact-set accuracy from 0.1250 to 0.2083 and
early negative accuracy from 0.6667 to 0.7750. In the second half, negative
accuracy improves 0.3333→0.4702, but exact-set accuracy falls 0.2917→0.2500
and positive recall falls 0.7014→0.6111.

On matched OOD `n=4`, multiplier 5 raises fidelity 0.4250→0.6083, validity
0.6550→0.7650, structure 0.1494→0.2038, reward 0.5078→0.8120, AST node F1
0.2574→0.2976, and AST edge F1 0.0000→0.0417. Component recall regresses
0.2083→0.0625. Meaningful-v1 and strict-v2 stay 0.0; AgentV remains 0/1
without execution errors. Root-reference identity decoding applies zero times
in both arms, attributing the changed programs to training distribution.

**Verdict:** retain the generalized sampler, but reject multiplier 5 and both
scratch checkpoints for promotion. The tradeoff is directionally useful but
too aggressive; test a moderate multiplier next. Machine-readable evidence:
[JSON](iter-e546-root-reference-coverage-sampling-20260719.json).
