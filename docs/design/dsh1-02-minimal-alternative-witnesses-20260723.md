# DSH1-02 minimal alternative witnesses (SLM-354)

**Decision:** supported at the contract-fixture evidence level. The OpenUI
CAP0 grammar basis now accounts for every reachable/productive declared Lark
alternative: 71 have verified containing-context witnesses and five have typed
unsupported reasons. There are no unexplained gaps.

Machine-readable evidence:
[`dsh1-02-minimal-alternative-witnesses-20260723.json`](dsh1-02-minimal-alternative-witnesses-20260723.json).

## Search and admission

The pack owns a finite candidate authority; generic code never imports OpenUI.
For each named start and exact production alternative, the selector admits only
candidates that pass fragment parse, backend static validation, scope policy,
symbolic-surface policy, canonicalization, canonical idempotence, and parsed
AST round trip. Exact Lark reduction tracing supplies the focus AST path.

Among admitted candidates that exercise an alternative, selection minimizes:

```text
(ast_nodes, productions, optional_nodes, markers, surface_tokens)
```

Canonical source and original source provide stable tie-breaks. Canonicalization
alpha-normalizes binders; canonical state markers are re-declared to the
symbolic-surface policy before admission. The resulting basis identity is
`7519b9eae082dc71f57bd541946497c2cfd6ef2d35c44b823e2fbe4103a623f4`.

## Coverage

| Disposition | Alternatives |
| --- | ---: |
| verified witness | 71 |
| typed unsupported | 5 |
| unexplained | 0 |
| total reachable/productive | 76 |

The 71 rows select from 35 exact containing sources and 32 canonical sources.
The five unsupported rows are not counted as witnesses:

- two start alternatives cannot contain the statically required root;
- Lark's `_NL` regex consumes repeated newlines in one token, making the
  generated recursive newline helper lexer-shadowed;
- numeric literals violate the symbolic-surface ban on open target numbers;
- the builtin-call terminal has no declared symbolic-surface authority.

If any reachable/productive alternative lacks either an admitted exact trace
or one of these pack-owned typed reasons, `UnexplainedAlternativeGap` blocks
the basis.

## Verification and claim limits

The shared OpenUI/mini-DSL focused suite passed 32 tests. The mini-DSL has no
unsupported rows and proves that the same selector chooses the cheaper atomic
source over its containing pair. Determinism, exact focus replay, marker
closure, canonical AST round trip, and the unexplained-gap stop rule are
covered directly.

This is deterministic repository-contract evidence. Minimum cost is relative
to the finite pack-declared candidate authority. No data build, train, model
eval, benchmark, checkpoint, AgentEvals publication, capability certificate,
or ship claim was produced.

## Research lineage

Abstract Syntax Networks and TRANX motivate grammar/AST-governed generation
whose structure follows a declared target language. This implementation is an
adapted deterministic coverage enumerator, not either neural decoder.
