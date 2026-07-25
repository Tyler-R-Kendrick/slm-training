# DSH1-01 declared grammar capability adapter (SLM-353)

**Decision:** supported at the contract-fixture evidence level.
`GrammarCapabilityAdapterV1` exposes a pack's declared grammar authority
through the existing `DslPack` registry. It does not create another registry
and does not infer productions from corpus examples.

Machine-readable evidence:
[`dsh1-01-grammar-capability-adapter-20260723.json`](dsh1-01-grammar-capability-adapter-20260723.json).

## Contract

The adapter exposes named start symbols, exact production alternatives,
terminal categories, reachability/productivity/nullable/recursive analysis,
fragment parsing, canonical serialization, static validation, scope policy,
and completion frontiers. Missing declarations return
`UnsupportedCapabilityV1(status="UNSUPPORTED")`; they do not become empty
success values. A partial pack therefore cannot report itself complete.

Five independent authority surfaces are fingerprinted: grammar, backend,
library schema, property order, and placeholder policy. A combined fingerprint
binds those five hashes. Changing the declared production authority changes
both the grammar and combined fingerprints.

OpenUI-specific construction remains in its pack wiring. The generic adapter
module imports no OpenUI implementation and the complete test-only mini-DSL
uses the same conformance suite.

## Evidence

| Control | Result |
| --- | --- |
| OpenUI complete-pack conformance | pass |
| independent mini-DSL complete-pack conformance | pass |
| partial `toy-layout` reports typed unsupported capabilities | pass |
| fake example grammar text cannot alter productions | pass |
| declared-authority mutation changes fingerprint | pass |
| undeclared fragment start is rejected | pass |

The focused pack and adapter run passed 27 tests. Ruff, compileall, and
`git diff --check` passed. OpenUI's declared Lark authority exposes 76
production alternatives, 36 terminal categories, and 28 nonterminals; all 28
are reachable and productive under the declared `start` symbol.

This is deterministic repository-contract evidence. No corpus synthesis,
train, model eval, benchmark, checkpoint, AgentEvals publication, capability
certificate, or ship claim was produced.

## Research lineage

The design follows the language-agnostic authority boundary described by
Macedo, Viera, and Saraiva's property-based testing of attribute grammars:
derive checks from the declared grammar/attribute system and exercise multiple
languages through a shared property surface. Here that principle is adapted
to pack capability conformance; this implementation is not their attribute
grammar testing system.
