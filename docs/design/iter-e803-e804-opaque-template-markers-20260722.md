# E803-E804: opaque template-marker enforcement

## Outcome

E803 is invalid evidence. Its apparent settings-record improvement came from
interpreting user-defined marker suffixes and namespaces as semantic labels.
Those metrics cannot support a model, checkpoint, or promotion decision.

E804 changes the harness policy instead of hiding that result. Template markers
are now codec identities only. Training and decode may emit grammar/AST symbols,
closed schema literals, and declared template markers; they may not learn or
route on human-authored marker names.

## Enforcement

- The canonical lever catalog declares `template_markers_are_opaque=true`.
- Output contract v4 rejects every checkpoint trained before harness-owned
  canonical slot persistence.
- Shared train/eval loading rejects explicit template semantic-role label lines
  before model construction or run artifacts.
- Config construction rejects semantic-role marker levers, namespace-based
  repeated-slot routing, and `symbol_anonymization=false` before artifacts exist.
- `RuntimeSymbol(role="external_entity")` rejects namespace, semantic type,
  semantic role, scope, signature, and description metadata.
- Evaluation no longer derives roles from marker suffixes.
- TwoTower and grammar-diffusion context features, pseudo-embeddings,
  slot-component features, and deterministic template fill use ordinal marker
  identities such as `:slot_0`.

## Verification

The focused compiler-decode suite passed 182 tests, and the final cross-model
opaque-marker suite passed 280 tests in 14.04 seconds, both under the 110-second
interrupt cap. No train or evaluation run was launched for E804, so there is no
new checkpoint, scoreboard, AgentEvals file, or AgentV bundle.

## Decision

Reject E803 and invalidate prior checkpoints or recipes that require semantic
marker labels. Future quality work must derive component ownership from prompt
structure and the public grammar/schema, never from marker spelling.
