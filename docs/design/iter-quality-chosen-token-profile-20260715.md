# Iteration: chosen-token-only constrained profile (2026-07-15)

The Lark profile suggested avoiding broad candidate scans, so standalone eval
now exposes `--verify-chosen-only`. This preserves grammar-constrained mode but
checks only the model-selected token at each step.

A one-record, one-step, one-attempt smoke probe with this control still exceeded
the execution window. Thus the remaining cost is not only broad top-k ranking;
the constrained DFA/Lark path itself remains too slow in this environment.
No generated-quality or ship claim is made.
