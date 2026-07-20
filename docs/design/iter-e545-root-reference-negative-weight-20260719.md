# E545 — root-reference negative weighting

E545 tests whether stronger supervision on generated sections excluded from the
terminal root improves the bounded identity head. The implementation uses
per-class BCE weights: positive classes remain weight 1, while negative classes
use a configurable weight. The default remains 1.

Two clean, matched 24-step CPU HF-context continuations start from the E544
checkpoint with the same seed and batches. The weight-1 control processes 1,270
target tokens in 30.64 seconds and writes SHA
`9e54d4700938c2e1feececfa3b952d4188c76873281e54d38f19bcea4cc76fa1`.
The weight-4 treatment processes the same tokens in 28.64 seconds and writes SHA
`14dd44043887cfb6b5a14b1a99fee3750dc8f72c2d27f205fe3bdc0506de61ae`.
Both use `max_wall_minutes=3` and explicit no-sync scratch persistence.

The treatment changes only sparse negative calibration. Over steps 13–24,
mean negative accuracy rises from 0.3333 to 0.3958, while exact-set accuracy
stays 0.2917 and positive recall stays 0.7014. Only four batch rows in that
window contain a nontrivial negative target; the first half contains three.
Increasing loss weight therefore cannot substitute for exposing the model to
the 42 strict-subset records more often.

Matched four-record OOD evaluation is exactly neutral: the two arms produce
byte-identical programs. Both score syntax 1.0, meaningful-v1 0.0, fidelity
0.4250, validity 0.6550, structure 0.1494, component recall 0.2083, reward
0.5078, AST node F1 0.2574, AST edge F1 0.0, and strict binding-aware meaning
0.0. Neither root-arity nor root-identity decoding applies on this replay.
AgentV fails 0/1 without an execution error.

Both extra continuations also regress from the retained E544 treatment:
meaningful-v1 falls 0.25→0.0, structure 0.1688→0.1494, component recall
0.2708→0.2083, reward 0.7370→0.5078, and AST node F1 0.2833→0.2574.

**Verdict:** reject negative weight 4 for this recipe and reject both scratch
checkpoints for promotion. Keep the generalized weighted-loss capability, but
the next experiment should condition sampling on strict-subset identity targets
instead of raising their loss weight. Machine-readable evidence:
[JSON](iter-e545-root-reference-negative-weight-20260719.json).
