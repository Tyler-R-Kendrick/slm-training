# Expert-nomenclature packs — scoping (F4)

Scoping document only. No grammar, ontology import, corpus, model, or ship claim
exists yet. This scopes the fourth DSL family of the program (OpenUI → GraphQL →
patterns → **expert nomenclatures**) and the pack-contract variant it needs.
Linear SLM-45.

## What "expert nomenclature as a DSL" means

Domain vocabularies — clinical terminologies (SNOMED CT, ICD, LOINC), chemistry
(IUPAC), taxonomy, legal citation, etc. — are *controlled vocabularies over an
ontology*: a large set of concepts plus typed relations (is-a, part-of,
has-property) and composition rules. A "program" is a well-formed expression in
that vocabulary (e.g. a post-coordinated SNOMED expression, an IUPAC name, a
citation).

## Why the existing pack contract does not fit as-is

The OpenUI / GraphQL / patterns packs assume a **context-free grammar** with a
small terminal set. Nomenclatures differ on two axes:

1. **Flat, huge symbol space.** The "vocabulary" is 10^4–10^6 concepts, not a
   handful of components. A CFG over that is the wrong tool; the real constraint
   is *graph membership + relation legality*, not phrase structure.
2. **The oracle is an ontology-consistency check, not a parser.** Validity =
   "this expression's concepts and relations are consistent with the ontology"
   (a reasoner / description-logic check), not "this string parses."

So F1's pack contract needs an **ontology variant**:

| Pack slot | CFG DSL (OpenUI/GraphQL/patterns) | Ontology DSL (nomenclatures) |
| --- | --- | --- |
| grammar | Lark CFG | expression skeleton grammar + **graph-walk constraint** over the concept/relation graph |
| validity oracle | parser (+schema) | ontology consistency check (DL reasoner / relation-legality) |
| symbol space | fixed terminals | the ontology's concept set — the schema **is** the symbol table |
| constrained decode | DFA / completion forest | mask to concepts reachable by a legal relation edge from the current node |
| canonicalizer | D2 codec round-trip | ontology normal form (canonical concept ids, sorted role groups) |

## How it reuses the program's own results

This is not a from-scratch build — it is the strongest test of two ideas already
implemented in this program:

- **Schema-as-symbol-table (from the GraphQL pack, F2).** GraphQL proved the
  introspection schema can serve as the runtime symbol inventory. An ontology is
  the same idea at larger scale: the concept graph is the dynamic symbol context
  the decoder constrains against.
- **Dynamic pseudo-embeddings (Track C2, DyVo-style).** A 10^6-concept vocabulary
  is exactly the open-vocabulary regime C2 targets: per-expression concept
  embeddings built on the fly rather than a fixed learned row per concept. F4 is
  where C2 stops being optional.
- **Constrained decoding over a graph** generalizes the existing completion
  forest from a CFG acceptor to an ontology-graph walk — the A-track decode
  machinery with a graph transition relation instead of a DFA.

## Pilot proposal (smallest honest first cut)

Do **not** start with SNOMED. Pick a small, fully-open ontology with a cheap
consistency check (e.g. a few-hundred-concept taxonomy or a bounded IUPAC
substructure) so the whole pack can run the end-to-end fixture loop on CPU. Prove:
concept-graph constrained decode emits only ontology-consistent expressions, and
canonicalization collapses synonymous expressions to one normal form. Only then
scale the vocabulary.

## Open questions (resolve before pack work)

- Which reasoner/consistency check is cheap enough to be the reward oracle in the
  autoresearch loop? (Full DL reasoning may be too slow per sample — a restricted
  relation-legality check may suffice for the first cut, labeled as such.)
- Does constrained decode walk the graph edge-by-edge, or generate an expression
  skeleton then bind concepts? (Latter reuses the sketch-then-fill / choice-codec
  machinery; former is a new decoder mode.)
- Licensing: several expert ontologies are not freely redistributable — the pilot
  must use an openly licensed ontology so committed corpora stay shareable.

## Verification (when built)

- Scoping: this doc committed and reviewed (no code yet).
- Pilot pack: end-to-end fixture run green under the F1 ontology-variant contract,
  reward = ontology-consistency (honestly labeled by which check was used).

## Honesty

No ontology, grammar, oracle, corpus, model, or metric exists yet. This is scope
plus the pack-contract-variant decision, committed before any build so the
ontology oracle and licensing constraints are settled up front.
