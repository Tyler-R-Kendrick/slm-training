# E830-E831: strict request harness

## Outcome

E829 was a harness-policy failure, not evidence that the model should learn to
convert content strings or caller-authored marker names. Its checkpoint-declared
decode allowed schema enum tokens in content positions, timed out four of five
rows, and its result artifact incorrectly hid the opaque slot inventory that
the evaluator had passed to generation.

E830 replayed the same local scratch checkpoint and held-out five-record subset
with the existing strict compiler-tree policy. Parse rose from 0.0 to 1.0,
placeholder fidelity from 0.0 to 0.8857, reward from 0.0 to 0.9195, and timeout
count from four to zero. Strict-v2 and AgentV still fail, so this remains a
diagnostic and the checkpoint remains rejected.

E831 then exercised the canonical CLI without an evaluation-policy flag after
the harness fix. The default resolved to `strict_compiler_tree`; the smoke row
completed in 2.73 seconds with parse and meaningful-v1 at 1.0, fidelity 0.75,
reward 0.874, and no fallback or timeout. Its recorded generation request
contains the four opaque slots actually passed to decoding.

## Harness correction

- The single lever catalog now owns the policy identifiers, strict policy
  bundle, and canonical evaluation default.
- Canonical evaluation defaults to strict compiler-tree decoding.
- The checkpoint-declared comparison may retain checkpoint architecture and
  conditioning, but still inherits the fail-closed grammar, final validation,
  opaque-slot constraint, honesty, and no-fallback boundary.
- Evaluation evidence preserves the actual request slot inventory regardless
  of whether that inventory is also rendered into prompt context.

Only grammar/AST symbols, closed schema literals, and opaque `:slot_N` markers
are valid completion material. Content binding and caller-name conversion stay
outside the model.

## Decision

Retain the harness correction. Reject checkpoint promotion, sync, deployment,
and ship claims: E830 is only held-out n=5, E831 is smoke n=1, strict-v2 is 0,
and AgentV is 0/1. Both runs were local CPU scratch diagnostics and completed
inside the two-minute command cap.
