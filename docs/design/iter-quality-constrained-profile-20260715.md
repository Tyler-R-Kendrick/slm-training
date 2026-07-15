# Iteration: constrained decoder profile after stream-probe fixes (2026-07-15)

After the clean-main DFA and remask probe fixes, the one-record constrained
smoke probe was retried with one decode step, one attempt, and no DESIGN
context. It still exceeded the execution window. A timed stack trace now shows
the active frame in `lark.lexer`, rather than the earlier LangCore subprocess.

This moves the diagnosis forward: the remaining cost is incremental DFA/Lark
candidate probing. The unconstrained control remains the bounded feedback
fallback and completed in 3.39 seconds with persisted AgentV artifacts, but
its parse, structure, and reward scores were all zero. No generated-quality or
ship claim is made from either diagnostic path.
