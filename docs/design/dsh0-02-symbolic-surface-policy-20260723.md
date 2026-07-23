# DSH0-02 symbolic surface policy (SLM-346)

**Disposition:** adopt `SymbolicSurfacePolicyV1` as the fail-closed admission
boundary for future staged targets. This is a deterministic contract fixture,
not a train/eval/benchmark, checkpoint, capability certificate, or ship claim.

Machine-readable evidence:
[`dsh0-02-symbolic-surface-policy-20260723.json`](dsh0-02-symbolic-surface-policy-20260723.json).

## Authority and decisions

The policy consumes the active `DslPack` backend/schema and a
`GenerationRequest` effective runtime-symbol table. It reuses the existing
placeholder policy, `RuntimeSymbol` roles, binder alpha-renaming, and opaque
`ScopeEnv` identities.

| Surface category | Admission rule |
| --- | --- |
| grammar keyword/punctuation | allow only pack surface authority |
| closed enum/primitive | allow only pack schema/grammar authority |
| binder | allow grammar-local definitions/references or declared binder roles |
| external/state reference | allow only the matching declared runtime role |
| open string/number | template only through a compatible pack marker; otherwise reject |
| comment/prose | emit typed `reject` violation |
| undeclared identifier | emit typed `reject` violation |

Every violation includes the exact character span, surface, category, pack ID,
pack surface-authority SHA-256, `symbolic_surface_policy/v1`, decision, and
suggested existing marker role. No marker family was added.

## Fixture evidence

| Fixture | Result |
| --- | --- |
| OpenUI closed `"column"` plus declared `:hero.title` | admitted, 0 violations |
| OpenUI `"Welcome"`, `# note`, and numeric `5` | rejected with string→template, prose→reject, number→reject |
| GraphQL schema fields/types plus declared `$id` | admitted offline, 0 violations |
| undeclared placeholder/state | rejected with required existing role |
| binder alpha rename (`item`→`copy`) | both admitted; pack canonical AST equal |
| content-marker alias permutation | both admitted; opaque `ScopeEnv` fingerprint and stable-ID surface equal |

The two pack-authority fingerprints are:

- OpenUI: `b37da286bdbcc41c3227f7ec6a379f78c9c62348be682cdbd7544c4feb77cb2d`
- GraphQL: `1799d60a6041fe5abcd969c87c48f012eba3a9e8c40cd935ad24574e13c1b947`

The GraphQL fixture reads the committed schema authority directly, so the
surface contract stays testable without pretending the optional Node bridge is
installed. Compiler/parser validity remains a separate mandatory downstream
gate.

## Honesty boundary

This policy is stricter than the existing output-contract-v2 string check, but
it is not retroactively applied to historical corpora. SLM-347 and later
staged builders must call it before materialization. Open numbers are
classified as open-class and must be templated or rejected in staged targets;
existing non-staged numeric-token behavior is unchanged.

The fixture suite passes 15/15 focused tests. Full delivery also checks the
pack, tokenizer, sanitization, runtime-symbol, version, repository-policy, and
static suites. No staged data was built and no model was run.

## Next disposition

Proceed to SLM-347 by making `SynthesisPlanV1` require this policy version and
reject any plan whose requested capability or target surface lacks authority.
