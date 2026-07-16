# E89 fallback-aware scoreboard — 2026-07-15

E89 corrected evaluation accounting so certified template output contributes to
the top-level `fallback_count` instead of appearing as learned success.

The same E82 smoke probe still had parse/fidelity 1.0/1.0 and 2,583.66 ms p50,
but now reports `fallback_count=1`, `template_fastpath_count=1`, and
`template_fallback_count=0`. AgentEvals persisted normally.

Decision: retain the scoring correction. Any future promotion gate must reject
or separately classify results with nonzero fallback count; this prevents the
certified template upper bound from being confused with model quality.

This is a harness validation, not a ship claim.
