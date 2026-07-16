# E52 symbol-literal adjacency guard — 2026-07-15

The strict E52 traces showed quoted symbol literals followed immediately by
brackets or another literal-open token. The picker now rejects those specific
adjacent transitions when no separator is present.

Focused tests passed: 25. Strict smoke results changed to raw syntax validity
1/3, parse 0/3, structural similarity 0.3600, fidelity/reward 0, and p50
latency 22.2s. Syntax validity improved, but structural similarity regressed
from the arity-guard result, so this is retained as a correctness guard rather
than promoted as a quality win.

Next target: preserve valid component structure while closing native symbol
arguments.

This is strict scratch smoke evidence, not a ship claim.
