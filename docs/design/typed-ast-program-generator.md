# Coverage-guided typed-AST ProgramSpec generation

`src/slm_training/data/progspec/generate.py` is the deterministic root-data
source for ProgramSpec families. It constructs typed component, reference, list,
scalar, and placeholder nodes; source text is only the final serialization
boundary. Every candidate is round-tripped through the pinned official OpenUI
bridge and the G0–G12 verifier before it can become a Silver ProgramSpec.

Candidate selection maximizes uncovered singles, component pairs, and a small
set of selected three-way interactions. This avoids a Cartesian product while
still reporting uncovered cells for components, props/value classes, graph
depth/width/topology, length, viewport/state combinations, and grammar forms.

OpenUI 0.2.x content props require placeholders. DSL-like or instruction-like
literal probes therefore live in ProgramSpec facts while the AST contains only
the placeholder; the literal is data for downstream substitution and is never
parsed as generator source.

## Contract boundary

The current repository contract is OpenUI 0.2.x layout syntax. State, query,
mutation, action, and tool cells are reported as **deferred**; the generator does
not invent unsupported syntax or call that coverage complete. When the official
language surface is pinned and hashed in a future contract version, those typed
nodes and target cells can be added without changing ProgramSpec lineage.

This generator and its unit tests establish data plumbing only. They are not a
model-quality, full-corpus, or ship-gate result; those claims still require the
documented train/eval matrices and durable measured evidence.
