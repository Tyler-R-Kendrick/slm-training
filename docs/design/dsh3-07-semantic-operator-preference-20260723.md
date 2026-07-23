# DSH3-07 semantic-first operator preference

SLM-375 defines a deterministic preference contract for operator outcomes that
share one compiler-owned `SemanticFrame` fingerprint and one intended
equivalence-class fingerprint. It deliberately does not expose a weighted or
scalar reward: required semantics and verifier validity are eligibility
conditions that no amount of structural brevity can offset.

## Lexicographic contract

`SemanticPreferenceCostV1` orders candidates by:

1. complete-and-valid eligibility;
2. missing semantic obligations;
3. verifier validity;
4. proven sequence defects;
5. canonical AST nodes, productions, optional nodes, and markers;
6. locality violations;
7. operator-sequence length; and
8. the sum of declaration-owned operator costs in integer micro-cost units.

Every later field is considered only after all earlier fields tie. A complete,
verified candidate therefore outranks an invalid or incomplete candidate even
when it is structurally larger or uses a longer, more expensive sequence.
Candidates from different semantic frames or equivalence classes fail closed
instead of receiving a cross-task ordering.

Canonical AST counts are compiler-supplied measurements carried as an explicit
record. Operator cost is bound from the matching `AstOperatorV1` declaration
and successful `OperatorApplicationV1` proof; it is not accepted as a free
model score.

## Intent direction

`SIMPLIFY` minimizes the four structural AST counts. `EXPAND` reverses only
those structural axes, so a compiler-verified expanded equivalent is preferred
for an explicit expansion task. `PRESERVE` ignores structural size and compares
the later sequence axes. Semantic completeness, verifier validity, sequence
defects, locality, and operator costs never reverse with task wording.

Equal cost vectors form a deterministic tie tier, ordered only for stable
serialization. They do not create a fabricated chosen/rejected pair.

## Sequence safety

An operator preference sequence is state- and AST-contiguous and retains each
successful application ID, semantic action ID, declaration fingerprint,
locality count, and exact operator cost. Diagnostics identify:

- state-preserving no-op steps;
- returns to any previously visited state (cycles);
- repeated application identities; and
- caller-supplied compiler proof that an edit is redundant.

The default `REJECT` policy removes defective sequences from ranking and pair
generation. The diagnostic `PENALIZE` policy keeps them with a lexicographic
defect count ahead of AST brevity, so a smaller no-op/cycle sequence cannot
win. Calling the standalone cost function with a defective sequence under
`REJECT` also fails closed.

## Preference groups

`OperatorPreferenceGroupV1` materializes:

- stable ranked tie tiers;
- explicit rejected candidate IDs;
- groups sharing the same final AST fingerprint;
- groups sharing the same semantic operator-sequence fingerprint; and
- strict chosen/rejected relations with the first differing cost axis.

Pair generation has an explicit bound and fails rather than silently dropping
relations. Input order and request-local application surfaces do not influence
the ranking keys.

## Evidence and scope

Deterministic unit controls cover:

- complete/valid outcomes against shorter incomplete and invalid outcomes;
- frame/equivalence scope mismatches;
- input-order permutation and true ties;
- simplification, expansion, and preserve direction;
- no-op, cycle, and proven-redundancy rejection and penalty;
- locality-before-length/operator-cost ordering;
- final-AST and semantic-sequence grouping;
- pair-bound failure; and
- declaration/application proof binding with exact fixed-point operator cost.

The metamorphic-testing source/follow-up framing in
[Saha and Kanewala (2018)](https://arxiv.org/abs/1802.07361) is used only as
general motivation for systematic equivalent-case controls. This repository
contract is not an implementation or evaluation of that paper. No train, eval,
benchmark, matrix, checkpoint, model-card, ship-gate, or model-quality claim is
produced.
