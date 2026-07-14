# Coverage-guided typed-AST ProgramSpec generation

`src/slm_training/data/progspec/generate.py` is the deterministic root-data
source for ProgramSpec families. It derives its component and property space
from the pinned OpenUI schema, currently covering all 54 published components,
and constructs typed component, reference, list, scalar, and placeholder nodes.
Source text is only the final serialization boundary. Every candidate is
round-tripped through the pinned official OpenUI bridge and the G0–G12 verifier
before it can become a Silver ProgramSpec.

Candidate selection maximizes uncovered singles, component pairs, and a small
set of selected three-way interactions. It adaptively upweights uncovered and
low-hit cells instead of building a Cartesian product. Reports include covered,
uncovered, and explicitly unsupported cells for components, authoritative
props/value classes, graph depth/width/topology, length, viewport/state
combinations, content forms, and dataflow features.

OpenUI 0.2.x content props require placeholders. DSL-like or instruction-like
literal probes therefore live in ProgramSpec facts while the AST contains only
the placeholder; the literal is data for downstream substitution and is never
parsed as generator source.

Generate verified roots and a machine-readable coverage report with:

```console
generate-progspecs --count 16 --output outputs/progspec/programs.jsonl \
  --coverage outputs/progspec/coverage.json
```

## Contract boundary

The current repository contract is OpenUI 0.2.x layout syntax. State, query,
mutation, action, and tool cells are reported as **deferred**; the generator does
not invent unsupported syntax or call that coverage complete. When the official
language surface is pinned and hashed in a future contract version, those typed
nodes and target cells can be added without changing the stable family and
split-group identities. Each generated root still receives a unique ID and
lineage ID.

This generator and its unit tests establish data plumbing only. They are not a
model-quality, full-corpus, or ship-gate result; those claims still require the
documented train/eval matrices and durable measured evidence.
