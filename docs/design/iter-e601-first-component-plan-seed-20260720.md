# E601 — first-component semantic-plan seed

Date: 2026-07-20
Status: negative; default-off, not promotable or ship

E601 adds a default-off semantic-plan score restricted to the first legal
root-component family choice. It leaves the retained global plan weight at 4,
so stronger initial pressure cannot repeat E600 weight 8's later duplicate
component regression.

The initial v37 implementation used parser-frame emptiness as its gate.
Completed weights 8, 16, and 32 were all prediction-identical to E600. That
negative ladder exposed that production root-family selection already has a
parser frame. Version v38 instead gates on compiler `component_root` candidate
kinds while requiring zero completed sections.

The corrected v38 weights 8 and 32 are also prediction- and metric-identical
to E600 on capped CPU OOD `n=4`: syntax 1.0, meaningful-v1 0.5, strict
meaning-v2 0, fidelity 0.5917, validity 0.7550, structure 0.5169, component
recall 0.6250, reward 0.8115, AST-node F1 0.5754, and AST-edge F1 0.4143.
Dashboard and gallery remain collapsed to `TextContent`.

All five runs completed within 170 seconds and emitted AgentEvals JSONL plus
AgentV bundles. AgentV remains 0/1. Keep the lever default-off; do not promote,
create, or sync a checkpoint. The next experiment should capture the actual
first-family score decomposition before adding more pressure.

Evidence: [JSON](iter-e601-first-component-plan-seed-20260720.json).
