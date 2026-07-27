# DSH5-11 parameterized-template admission disposition

SLM-419 is an explicitly gated integration issue. Its precondition is not a
review preference: model or registry integration must not begin until both a
train-only parameterized-template manifest and a matched
BASE/FLAT/CLOSED/PARAM evidence set exist.

## Audit (2026-07-26)

The tracked repository contains the fixed-span macro miner
(`src/slm_training/data/macro_induction.py`), fixed template aliases
(`src/slm_training/dsl/operators/topology.py`), and the E280 fixture report
(`iter-e280-c3-macro-tokens-20260717.md`). None is a parameterized-template
artifact: E280 mines fixed lexer spans and explicitly records no matched
no-macro quality control.

The following repository-wide tracked-path audit returned no matches:

```text
TSA-013 | TSA-014 | ParameterizedTemplateAction | parameterized template |
template manifest | template_manifest
```

Accordingly, there is no frozen train-only content-addressed manifest with
provenance, support counts, held-out overlap audit, or capacity cap, and no
matched BASE/FLAT/CLOSED/PARAM semantic/work comparison. There is also no
`ParameterizedTemplateActionV1`, parameterized action matrix, or associated
test owner to extend.

## Disposition

**Integration unavailable; no model or registry change made.** The existing
fixed macro and template-alias paths remain unchanged. A fixture or fixed-span
compression result is not evidence for typed open-leaf template actions and
cannot supply the missing gate.

The required successor work is to produce the two explicitly named admission
artifacts under a train-only, leakage-audited provenance boundary. Only then
may a subsequent implementation introduce parameter slots, compiler lowering,
or matched serving/model arms. This disposition makes no checkpoint, remote/HF,
human-rating, semantic-improvement, efficiency, or ship claim.
