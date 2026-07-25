# DSH1-07 partial-state classification with compiler certificates (SLM-359)

**Decision:** supported at the deterministic contract-fixture level. Every
candidate partial source is classified through the declared grammar capability
adapter (DSH1-01 `GrammarCapabilityAdapterV1`) into exactly one
`PartialStateClassV1` — `COMPLETE`, `FRAGMENT(start_symbol)`,
`PREFIX(frontier)`, or `INVALID` — with lexer/marker cut discipline, bounded
accepting-completion certificates, coverage-gated pruning, and a family stop
rule.

Machine-readable evidence:
[`iter-slm359-partial-states-20260725.json`](iter-slm359-partial-states-20260725.json).

## Classifier

`partial_states/v1`
(`src/slm_training/harnesses/train_data/partial_states.py`):

- **COMPLETE** — parses under the pack's primary document start symbol and
  passes `static_validate` (`min_completion_cost = 0`, coverage `full`).
- **FRAGMENT(start_symbol)** — not complete, but parses under another named,
  declared start symbol via the live fragment parser (LALR over the declared
  grammar). Acceptance: every FRAGMENT re-parses under its declared start
  rule (tested).
- **PREFIX(frontier)** — not complete and not a named fragment, but the live
  completion frontier is non-empty **and** a `CompletionCertificateV1` proves
  at least one bounded accepting completion: a declared pack witness whose
  token stream extends the candidate at a lexer boundary, replayed through
  the frontier at every boundary step to a verified terminal
  (`static_validate` on the full witness). The certificate records the
  witness, completion suffix, token cost (bounded by `max_completion_edits`),
  replay steps, and the verified-terminal flag.
- **INVALID** — anything else, with a stable reason:
  `cut_inside_token:string`, `cut_inside_marker:placeholder`,
  `unreachable_from_compiler`, `no_certified_completion`,
  `capability_unsupported:*`, `empty_source`.

## Cut discipline

Candidates are only ever admitted at lexer/production boundaries. A cut that
ends inside a string token (odd unescaped quote balance) or inside an opaque
placeholder marker (trailing marker run rejected by the pack's declared
`placeholder_re`) is `INVALID` before any parse/frontier probe runs. Arbitrary
internal-hole generation is **excluded** from initial CAP0: the module
classifies compiler-reachable states only and never synthesizes holes.

## Report contents

Each `PartialStateReportV1` records the class and reason plus: the sorted
frontier terminal set, open nonterminals (declared productions whose rhs
mentions a frontier terminal), required roles and alpha-renamable binder
surfaces (via `harness_dsl.runtime_symbols_for_payload`), exact singleton
positions (character offsets in the certified replay where the frontier pins
exactly one content terminal), support coverage
(`full` / `partial` / `unknown`), minimum completion cost, and the compiler
certificate for admitted PREFIX states.

## Guards and stop rule

- `authorize_pruning` raises `PartialStatePruneError` unless the class is
  admitted with `full` coverage — `UNKNOWN` or `PARTIAL` coverage **never**
  authorizes destructive pruning, and a PREFIX without its certificate is
  rejected (tested).
- `family_valid` implements the stop rule: any partial unreachable from the
  live compiler/decoder (`unreachable_from_compiler` / unsupported frontier)
  invalidates its family; a `partial`-coverage member alone does not.

## Verification

- 13 contract tests
  (`tests/test_harnesses/train_data/test_partial_states.py`): COMPLETE /
  FRAGMENT / PREFIX / INVALID fixtures, PREFIX replay to a verified terminal,
  FRAGMENT under its declared start rule (multi-start mini pack), token/marker
  cut rejection, pruning guard, family stop rule, coverage + cost recording,
  `max_completion_edits` bounding, and byte-level determinism.
- Claim limits: fixture-scale contract evidence only; no corpus publication,
  no model evaluation, no ship-gate claim.
