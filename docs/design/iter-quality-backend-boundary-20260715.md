# Iteration: grammar backend boundary check (2026-07-15)

The bounded constrained smoke probe was rerun with the explicit `openui-lark`
backend, one record, one decode step, one attempt, chosen-token-only
verification, and the stream-probe skip enabled. It still exceeded the
execution window before a scoreboard.

This rules out the hybrid backend selecting LangCore as the only explanation.
The remaining bottleneck is in constrained candidate/state handling around the
incremental DFA/Lark path. The unconstrained control remains the only bounded
generated feedback path and is diagnostic-only; no generated-quality or ship
claim is made.
