# DSH3-10 verified symbolic operator corpus (SLM-378)

Date: 2026-07-23
Status: implemented; strict fixture contract passed
Scope: CAP2 compiler-owned operator data; no model, checkpoint, or ship claim

## Decision

The train-data pipeline can now emit a sibling symbolic corpus from admitted
OpenUI document roots. Every operator target comes from the same
`OperatorLegalSetV1` and pack-owned registry used by inference. The builder
does not ask an LLM to define a transformation and does not coerce operator
records into the ordinary natural-language `ExampleRecord` target schema.

The closed question opcodes are `APPLY_OPERATOR`, `NEXT_OPERATOR`, and `FORK`.
Successful turns emit operator-only, result-AST-only, and dual target views.
Forks emit only the immutable history operation. No open-class natural-language
field appears in a symbolic question or answer.

## Exact evidence boundary

Each successful record persists:

- source record and semantic family;
- before and after canonical source ASTs;
- the selected legal action and legal-set fingerprint;
- complete `OperatorApplicationV1` effect and proof;
- the canonical semantic-first preference sequence;
- the immutable conversation trace required for replay.

The legal-set report persists the full bounded domains, admitted actions,
complete rejection counts, and bounded typed rejected applications. Illegal
applications are evidence for conflict/unsupported strata only; they are never
emitted as target actions. Next-turn legal sets are rebuilt from the exact
intermediate state and persisted separately. Optional sibling forks remap
branch-local refs through the immutable conversation contract.

## Strict fixture run

The final evidence run used CPU, the strict profile, the fixture source, no
synthesizer, two admitted document roots, two actions per state, a 32-combination
per-operator bound, and sibling forks. It completed inside the three-minute
cap.

| Measure | Result |
| --- | ---: |
| Train candidates / admitted / rejected | 20 / 19 / 1 |
| Operator roots / records | 2 / 20 |
| Single-turn / next-turn / fork records | 12 / 6 / 2 |
| Operator-only / result-only / dual / history views | 6 / 6 / 6 / 2 |
| Enumerated legal successes | 27 |
| Rejected combinations retained in coverage | 533 |
| Bounded typed rejection samples | 158 |
| Illegal target actions emitted | 0 |
| Invalid generated families | 0 |
| Mean admitted source quality | 1.0 |

The corpus fingerprint is
`c44ebd4fb5e30a26bb3dce4bcce2b42f7d17cf5505f4e29d07c702a668d054ed`.
The parent strict build fingerprint is
`e086b62faf8cecb326a5697ecb12e5f7e6af5bc2e34e922dc3be1cafb9510928`.
The complete measured record is
[`dsh3-10-symbolic-operator-corpus-20260723.json`](dsh3-10-symbolic-operator-corpus-20260723.json).

This is fixture/wiring evidence only. It demonstrates deterministic generation,
pack validity, application replay, trace replay, and artifact completeness on
the configured roots. It does not establish full-corpus operator coverage or
model quality and is not a ship-gate result.

## Coverage and synthesis feedback

The four persisted legal sets expose 24 operator/state gap entries spanning
partial domains, invalid role/cardinality/property combinations, root deletion,
no-change, unsupported pack semantics, and authority rejection. The dataset is
therefore explicit about both successful and conflicting strata. The small
fixture naturally covers only `replace_node`, `set_property`, and history fork
targets; broader operator, state-size, scope, and semantic-family balance
remains measurable work rather than an implied claim.

The original strict parent build rejected only `train_text_only_01` at
`decontamination/test_fixture_structure`. `quality_report.json` contained no
warnings; `synthesis_feedback.json` emitted `eval_leakage_source` for the
`human_curated` family. SLM-392 confirmed that strict default elision made the
train seed isomorphic to reserved `adv_empty_prompt_01`, removed the accidental
train source, and reran the same strict recipe. The rerun admits 19/19 with zero
decontamination drops, no warnings or recommendations, and the same admitted
content fingerprint. The firewall and thresholds remain unchanged. Full audit:
[`slm392-human-curated-source-overlap-20260723.md`](slm392-human-curated-source-overlap-20260723.md).

## Validation

Focused controls prove:

- deterministic content fingerprints across repeated builds;
- exact legal-set membership and pack-owned application replay;
- bounded deterministic typed rejection samples;
- closed question/answer schemas and all four target views;
- single-turn, next-turn, and sibling-fork trace replay;
- train-pipeline manifest/stats registration;
- report stamps pin train-data, operator-contract, and legal-set versions;
- strict quality, rejection, and synthesis-feedback artifact production.

No checkpoint was created, so the model card and README checkpoint summary do
not change. This was a data build, not an evaluation, so AgentV output is not
applicable.

## Research lineage

[Brockschmidt et al., 2019](https://arxiv.org/abs/1911.01205) motivates explicit
graph-structured edit representations, while
[Gong et al., 2024](https://arxiv.org/abs/2405.20519) motivates syntax-tree
structure in program generation. DSH3-10 is an adapted repository contract:
neither paper specifies this OpenUI registry, legal-set enumeration, immutable
conversation trace, QA schema, rejection ledger, or replay boundary, and no
paper result is reproduced here.
