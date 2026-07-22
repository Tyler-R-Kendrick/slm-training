# Symbol-only output contract

OpenUI model targets and completions contain only grammar/AST symbols and
template placeholders. The model never learns or emits arbitrary string
content.

The canonical contract is `OUTPUT_CONTRACT_VERSION = 4` in
`src/slm_training/dsl/language_contract.py`. Its closed string-literal set is
derived from the pinned component schema. Values outside that set are content,
including lowercase identifiers, empty strings, operational names, and strings
inside arrays; data sanitization rewrites them to deterministic placeholders.

Both supported output tokenizers enforce the same contract. The choice codec has
no `LIT_STR` vocabulary entry, and the lexer-native tokenizer has no free-form
string opener. Generic string schema positions admit placeholders only; explicit
schema enums remain atomic grammar literals. Legacy compositional output
tokenization is not a supported training target.

Checkpoint loading requires output contract v4 exactly. Every earlier checkpoint
is intentionally incompatible and must not be evaluated, resumed, promoted, or
served by current code. Version 4 requires the train/test harnesses to rewrite
caller marker spellings to contiguous `:slot_<ordinal>` identities across every
persisted field, accepted target, and nested contract before admission. There is
no migration because v3 corpora and weights could retain marker-name semantics.

The invariant is enforced at five boundaries:

1. train/test sanitization templatizes every non-grammar string and atomically
   canonicalizes all marker identities before persistence;
2. the shared train/eval record loader rejects a nonconforming corpus before
   model construction, checkpoint creation, or run artifacts;
3. model construction rejects nonconforming records and free-form-capable output
   tokenizers;
4. tokenizer encoding and constrained decoding cannot represent free-form string
   literals; and
5. both meaningfulness metrics reject any free-form output string.

## Measured smoke

The strict fixture build `symbol-only-contract-smoke-20260721` processed 28
candidates and admitted 24. Sanitization templatized six literals across four
records with zero fallbacks. An independent audit found zero output-contract
violations, zero `<unk>` records in either supported tokenizer, and no `LIT_STR`
vocabulary entry. The build emitted no warnings.

The feedback ledger quarantined one existing human-curated seed because its
structure is reserved by the test corpus and dropped three abstraction-ladder
rows at the unchanged per-parent exposure cap. The resulting producer-input
cleanup candidate is retained; neither gate was weakened. No training ran and no
checkpoint was created.

Evidence: [JSON](symbol-only-output-contract-smoke-20260721.json).

A second strict integrated smoke covered all local producer families with one
ProgramSpec and one RICO row. It collected 186 candidates, admitted 141,
templatized 78 literals across 65 records with zero fallbacks, restored an
otherwise-invalid corruption-repair family to the admitted mixture, and again
found zero free-form strings across every primary and accepted target. Its four
feedback candidates concern existing Awwwards provenance quarantine and reserved
test-structure producer inputs; those gates remain unchanged and the candidates
are recorded in the evidence JSON.
