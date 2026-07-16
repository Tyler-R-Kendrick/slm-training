# E87 root RHS semantic guard — 2026-07-15

E87 rejected native symbol/binder tokens immediately after `root =`, based on
the E86 dead-end trace. Strict learned smoke remained invalid: parse/raw syntax
0.0, structural similarity 0.1909, reward 0.0, and exact placeholder fidelity
0.75. Contract precision/recall improved to 0.75/0.75; latency was 8,431.57 ms
with no timeout.

Decision: retain the narrow guard as a valid invariant, but reject it as a
quality intervention. The remaining malformed sequence occurs later than the
root RHS boundary and requires stronger learned structural supervision.

This is a bounded scratch diagnostic, not a ship claim.
