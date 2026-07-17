# VSS0-01 (SLM-57): the bounded verified-scope-solver contract

**Status:** specification only. This document defines the guarantee boundary for
the *Verified Scope Solving & Hybrid Realization* project. It adds no solver
code, no runtime dependency, no experiment, and no checkpoint, and makes no model
or ship claim. Existing checkpoints and decode behavior are unchanged until a
later VSS issue enables new flags behind a feature gate.

**Code:** none (contract/spec only).

**Reader contract:** `docs/design/` is the source of truth for coding agents and
experiment reviewers; the Linear project document is planning context only. The
terms and transitions below are meant to be precise enough that subsequent issues
can implement dataclasses and state transitions without inventing semantics.

## The distinction this contract adds

The repository already keeps [`CompletionForest`](../../src/slm_training/dsl/grammar/fastpath/compiler_draft.py)
authoritative and lets soft scores only *rank* it — see
[lattice-recursive-search.md](lattice-recursive-search.md), whose subordination
principle is that logits "cannot create a legal branch, discard all legal
branches, or bypass final OpenUI validation."

What is missing is a contract distinguishing two properties that are currently
conflated in review discussion:

- **prefix legality** — a candidate is compiler-admissible as the next action at
  the current decode position (Lark CFG reachability plus AST/binder/schema
  filters, as enumerated by `build_completion_forest`). This already exists.
- **support** — a candidate participates in **at least one bounded, fully
  verified completion**. This does **not** exist yet and is what this contract
  governs.

> **Disambiguation (do not drift).** "Support" here is *participation in a
> verified completion*. It is distinct from the existing **support signature**
> telemetry of E283
> ([iter-e283-signature-support-repair-20260717.md](iter-e283-signature-support-repair-20260717.md)),
> which is the coverage tuple `decision_kind + legal_token_ids + selected_token_id`.
> The E283 signature is decode-coverage instrumentation; it is **not** a
> completion-participation verdict and never authorizes candidate removal.

## Terms

Each term is defined so subsequent issues can implement dataclasses and state
transitions without inventing semantics.

- **`HoleId`** — stable identifier for a single unresolved decision site in the
  program IR (a position that must be filled by a choice/subtree). Stable across
  reversible search so certificates and nogoods can reference it. Implemented by
  VSS0-03 in `dsl/solver/state.py`.
- **finite domain** — the declared, bounded set of candidate fillings for a
  `HoleId` under the current pack/compiler coverage. Finiteness within declared
  bounds is a precondition for exact closure; a domain not provably finite within
  declared bounds yields `UNKNOWN` results. (Nearest existing phrasing: the
  lattice doc's "finite partial-information state"; formalized here.)
- **state fingerprint** — a canonical, order-insensitive hash of the solver state
  relevant to a hole (bound holes, active constraints, pack/compiler/verifier
  versions) used to key caches, certificates, and nogoods. Two states with the
  same fingerprint are interchangeable for replay. (Compare the existing
  `RankedForest.signature` and `ChoiceDecodeState.signature`, which hash search
  state but do not certify support.)
- **bottom** (`⊥`) — the empty domain: a hole with no legal candidate. A hole
  reaching bottom is a certified local contradiction; its incoming decision is
  certified UNSAT, distinct from timeout/`UNKNOWN`. (The existing empty
  `CompletionForest` / `RankedForest.is_bottom` is the structural precursor.)
- **structurally solved** — every `HoleId` in scope has a chosen candidate and
  the resulting IR has no remaining holes, but semantic verification has not yet
  been applied.
- **verified terminal** — a structurally solved state whose completion has been
  accepted by the applicable verifier(s) (the G0-G12 gate stack of
  [verifier-stack.md](verifier-stack.md), at the property tier claimed). The only
  state from which `SUPPORTED` may be asserted.
- **`SUPPORTED`** — status of a candidate for which a verifier-accepted witness
  completion (a verified terminal using that candidate) exists.
- **`UNSUPPORTED`** — status of a candidate proven to appear in **no** verified
  terminal, established only by exhaustive search with complete compiler coverage
  over that candidate's finite sub-domain, inside declared bounds.
- **`UNKNOWN`** — status when coverage is partial, a pack capability is
  unavailable, a node/verifier/time budget is exhausted, or a version is stale.
  `UNKNOWN` never permits candidate removal.
- **exact closure** — the deterministic fixpoint that propagates certified
  deductions (domain reductions that hold in *every* completion) until no further
  certified reduction applies. Exact closure only ever removes candidates that
  are certified `UNSUPPORTED`.
- **proof certificate** — a replayable record justifying a destructive
  transition: the state fingerprint, the pack/compiler/verifier versions, the
  transition, and the minimal evidence (witness for `SUPPORTED`,
  complete-coverage attestation for `UNSUPPORTED`, or a nogood derivation).
  Replaying a certificate against its recorded fingerprint must reproduce the same
  verdict; a certificate that does not replay is treated as absent.
- **certified deduction** — a domain reduction accepted by exact closure because
  a proof certificate shows it holds in every completion. Certified deductions
  are **not** reversible.
- **reversible decision** — a search branch that is *not* certified; it is
  recorded on the trail and undone on backtracking. Learned proposals enter here.
  (Existing precursor: `LatticeSearchState.choose`/`rollback`.)
- **local nogood** — a certified partial assignment that leads to bottom,
  recorded (keyed by state fingerprint) so the same dead region is not
  re-explored. A nogood is itself a proof certificate. (Existing precursor: the
  `Nogood` memory in `lattice_search.py`, which today lacks a replay certificate.)
- **verification capsule** — a dependency-closed unit of holes solved and verified
  together. Capsule boundaries are **strongly-connected components (SCCs)** of the
  hole-dependency graph, built from the def/use edges of
  [`ScopeContract`](../../src/slm_training/data/progspec/scopes.py)
  (`definitions`/`uses`/`visible_binders`). New vocabulary: a capsule is **not** a
  lexical/AST `ScopeKind` scope and is **not** the `dependency_closed_failure_cone`
  helper (which computes an AST least-common-ancestor, not an SCC). Scopes are not
  assumed independent.
- **opaque region** — a span the solver does not model semantically (e.g.
  user-owned/stripped content); preserved verbatim and spliced back hygienically,
  never used to justify a destructive transition.
- **semantic field** — a field whose value is constrained by verified semantics;
  it must pass through choice/AST IR and the verifier.
- **surface-only field** — a field with no semantic constraint beyond
  well-formedness; the only field class where unrestricted autoregression is
  permitted, subject to canonicalization plus the applicable pack oracle
  afterward.

## Implemented finite-domain state (VSS0-03 / SLM-59)

[`dsl/solver/`](../../src/slm_training/dsl/solver/) now implements the Torch-free,
model-independent hard-state carrier. It is not invoked by decode by default.

| Type / operation | Implemented contract |
| --- | --- |
| `HoleId` | Immutable namespace, mixed string/integer path, and kind; ordered by tagged canonical JSON rather than unsafe raw mixed-type tuple comparison. |
| `SolverBounds` | Non-negative token, node, depth, backtrack, and verifier-call bounds; every field participates in hard-state identity. |
| `DomainValue` | A tag plus immutable canonical JSON text. `to_dict` exposes ordinary tagged JSON while construction rejects objects, non-string object keys, and non-finite floats. |
| `HoleDomain` | Unique values and scalar metadata, both canonically ordered. An empty domain is bottom. |
| `FiniteDomainState` | Unique canonically ordered holes; monotone `refine`, domain-wise `meet`, reversible `with_decision`, stable JSON round-trip, and compact numeric summary. |
| `completion_forest_state` | Projects exactly one current compiler decision. Values preserve each full `CompletionPath.token_ids` plus `kind`; metadata preserves `coverage` and global support remains `UNKNOWN`. |

The full SHA-256 fingerprint includes `problem_id`, `pack_id`, constraint version,
bounds, hole IDs, domain values, and hard scalar metadata. It excludes model
logits/scores, timestamps, process IDs, mutable caches, certificate references, and
reversible search lineage (`decision_level` / `parent_fingerprint`). Thus two
search trails reaching the same hard domains share an identity. A decision records
the prior fingerprint as its parent but never changes `UNKNOWN` into
`UNSUPPORTED`.

`refine` accepts a forward-compatible `certificate_ref` argument but does not
persist it yet: certificate schema, checking, and replay belong to VSS0-04. Meet
requires identical problem/pack/constraint/bounds identity, the same hole IDs, and
type-preserving matching metadata rather than inventing a merge rule. Because meet
can combine unrelated reversible trails, it explicitly resets decision lineage to
level zero with no parent. All mutation-like operations return new validated
states; none embeds soft scores.

The compiler adapter deliberately ignores any optional explanation evidence added
by VSS0-02. Considered-candidate evidence is provenance, not an exhaustive support
proof. Empty forest paths produce bottom. A singleton forest is structurally solved
**only for that next-decision projection**; it is not a globally solved program,
verified terminal, `SUPPORTED` claim, correctness gain, or ship claim. The
`TopologyDomainAdapter` protocol is only a future model-independent seam and imports
no model or Torch code.

## Implemented support oracle (VSS0-04 / SLM-60)

[`dsl/solver/support.py`](../../src/slm_training/dsl/solver/support.py) implements
the deterministic reference oracle — the **first** component allowed to produce
`UNSUPPORTED` — plus a pure replay checker. It is Torch-free, is not wired into
decode, and makes no speed or ship claim. The problem-specific parts (how a chosen
value expands, how a terminal is verified) are injected through the `ProblemExpander`
and `Verifier` protocols; [`openui_support.py`](../../src/slm_training/dsl/solver/openui_support.py)
supplies the OpenUI wiring while tiny closed fixtures drive the exhaustive tests.

`EnumerativeSupportOracle.check(state, query)` iteratively explores the queried
candidate's completions (no Python recursion), deduplicates by the VSS0-03
`FiniteDomainState.fingerprint`, enforces every `SolverBounds` budget, and returns
one verdict with a replayable `SupportCertificate`:

| Verdict | Reference behavior | Removal? |
| --- | --- | --- |
| `SUPPORTED` | A `Verifier` `ACCEPT` on a decoded terminal was found. Valid even if other branches stay partial; the certificate stores only the witness **digest** plus a source label (never raw text/logits/timestamps). | Candidate stays live. |
| `UNSUPPORTED` | The candidate's whole finite completion space was exhausted with **only** `complete` coverage and no accepted terminal (`exhausted=True`, `stop_reason=None`). | **Yes — only** under a replayed `UNSUPPORTED` certificate. |
| `UNKNOWN` | Any `partial`/`none` coverage at a required expansion, a `Verifier` `UNAVAILABLE` (missing bridge/pack capability/timeout), or any budget stop. `exhausted=False`. | **Never.** |

- **Search order** — `canonical-domain-value-v1`: values are explored in the
  canonical order `HoleDomain` guarantees, so the search is deterministic and
  logit-independent (`build_completion_forest` takes no scores).
- **Certificate** — schema version `1`; records the query, verdict, problem/pack/
  constraint identity, bounds, search order, explored-state fingerprints, coverage
  observations, verifier profile, witness digest/source, per-reason failure counts,
  `exhausted`, and `stop_reason`.
- **Verifier profile (fixture wiring)** — `openui/lang-core-validate/well-formed@0.2.x`:
  a genuine `ParseError` is a hard `REJECT`; a missing lang-core bridge, timeout, or
  other `RuntimeError` is `UNAVAILABLE` → `UNKNOWN`. A timeout/UNAVAILABLE is **never**
  `UNSUPPORTED` (the timeout-vs-UNSAT distinction the contract requires).
- **Replay** — `replay_support_certificate(...)` re-derives identity, reruns the
  deterministic search, and returns structured violations (never a bare bool). It
  rejects an `UNSUPPORTED` certificate whose `exhausted` is false, whose coverage is
  incomplete, or that carries a budget stop; and a `SUPPORTED` certificate whose
  witness digest does not match a re-verified witness. `UNKNOWN` is accepted as an
  honest result but never as pruning authority.

Non-goals honored: no multi-candidate closure loop, no model ranking, no SMT
dependency, no decode/runtime flag, and no production-scale performance claim.

## Reference support semantics

| Verdict | Requirement | Removal permitted? |
| --- | --- | --- |
| `SUPPORTED` | A verifier-accepted witness completion (verified terminal) exists that uses the candidate. | Candidate stays live. |
| `UNSUPPORTED` | Exhaustive search with **complete** compiler coverage over the candidate's finite sub-domain, inside declared bounds, finds no verified terminal using it. | **Yes — only** with a replayable `UNSUPPORTED` certificate. |
| `UNKNOWN` | Partial coverage, unavailable pack capability, node/verifier/time-budget exhaustion, or stale version. | **Never.** |

Coverage completeness is evaluated over **each candidate's own finite
sub-domain**, not only globally: a forest whose overall `coverage` is `partial`
may still certify one candidate `UNSUPPORTED` when that candidate's sub-domain has
complete coverage, while another candidate in the same forest is `UNKNOWN`.

### Required invariants

- Learned modules may rank or propose; they never create legality and never
  produce `SUPPORTED`/`UNSUPPORTED` without a certificate.
- A learned proposal without an accepted certificate is reversible or ignored.
- Singleton domains still require final verification before `SUPPORTED`.
- Certified `UNSUPPORTED`/UNSAT is distinct from timeout/`UNKNOWN`.
- Correctness claims name the verified property tier; `G0-G12` is **not**
  represented as a universal semantic proof (see the verification ceiling in
  [verifier-stack.md](verifier-stack.md)).
- Existing checkpoints and decode behavior remain unchanged until a later issue
  enables new flags.

## Default reference backend

The default reference backend is **deterministic bounded enumeration** over the
existing choice/compiler path (`build_completion_forest` plus
`CompletionForest.coverage` plus the length-aware `ChoiceDecodeState.allowed_ids`
/ `exhaustive_allowed_ids`). **No SMT/Z3 is required** to install or run the
default backend.

An **optional future backend protocol** is defined for later issues only. A
conforming backend must (1) accept a state fingerprint plus a finite domain,
(2) return exactly one of `SUPPORTED`/`UNSUPPORTED`/`UNKNOWN` together with a
replayable certificate, and (3) never assert `UNSUPPORTED` without a
complete-coverage attestation. Backends are pluggable behind this protocol only;
none beyond bounded enumeration is required or assumed installed.

## Certificate replay and the deduction-vs-decision transition

Reference closure pseudocode. **Every destructive transition (the only lines that
remove a candidate or retract a decision) cites a required exact proof/replay
step**; non-destructive transitions do not:

```text
def close(hole):                       # hole.domain is a finite domain
    for cand in hole.domain:
        v = classify(cand, hole.state)          # SUPPORTED | UNSUPPORTED | UNKNOWN
        if v is SUPPORTED:
            keep_live(cand)                      # non-destructive; needs verifier witness
        elif v is UNSUPPORTED:
            cert = exhaustive_certificate(cand, hole.state)   # REQUIRED: complete-coverage attestation
            require(replay(cert) is UNSUPPORTED)              # REQUIRED: replay reproduces verdict
            remove(cand, because=cert)           # DESTRUCTIVE — the only candidate-removal path
        else:                                    # UNKNOWN
            keep_live(cand)                      # removal forbidden by invariant
    if hole.domain is BOTTOM:                    # ⊥ : certified local contradiction
        nogood = derive_nogood(hole.state)       # REQUIRED before retracting the parent
        record(nogood)                           # nogood is itself a replayable certificate
        retract_parent(hole, because=nogood)     # DESTRUCTIVE — cites the nogood
```

Deduction vs decision:

- A **certified deduction** (`remove`, `retract_parent`) is destructive and
  irreversible; it exists only with a replayable certificate.
- A **reversible decision** (`choose` a branch) is non-destructive, recorded on
  the trail, and undone on backtrack; no certificate required.
- Replay requirement: every certificate stores the state fingerprint and the
  pack/compiler/verifier versions, so an independent replay reaches the identical
  verdict. Stale versions or a non-replaying certificate ⇒ transition disallowed,
  candidate stays live.

## Capsule boundaries (dependency-closed SCCs)

Capsules are the strongly-connected components of the hole-dependency graph built
from `ScopeContract` def/use edges. Mutually constraining holes are solved jointly
in one capsule; independent SCCs are solved separately and joined through
interface summaries. This contract fixes that **capsules, not lexical/AST scopes,
are the unit of joint solving and verification**; the SCC decomposition itself is
implemented by a later issue. Lexical/AST `ScopeKind` boundaries
(`COMPONENT_CALL`/`STATEMENT`/`CHILD_LIST`) are inputs to that construction, not
the closure.

## Late realization

- Unrestricted autoregression is permitted **only** for fields classified
  surface-only.
- Realization occurs against the AST/choice IR — the B1 choice-sequence codec
  ([iter-b1-choice-sequence-codec-20260717.md](iter-b1-choice-sequence-codec-20260717.md),
  `models/choice_tokenizer.py`) — not the raw token stream.
- Every realized field is followed by canonicalization plus the applicable pack
  oracle (`DslPack` `ValidityOracle`, see
  [dsl-pack-contract.md](dsl-pack-contract.md)); a field that fails the oracle is
  rejected as a reversible decision, never destructively removed without a
  certificate.
- Semantic fields never take the autoregressive path.

## Relationship to existing implementation

| Symbol (file:line) | Existing role | Disposition under this contract |
| --- | --- | --- |
| `CompletionForest` (`fastpath/compiler_draft.py:29`) | Frozen enumerated next-action set for a prefix, carrying a `coverage: Coverage` guarantee (`complete`/`partial`/`none`) and `candidate_ids`. | **Retained, authoritative.** The support layer sits above it; its `coverage` tag is the substrate mapping to `SUPPORTED`/`UNSUPPORTED`/`UNKNOWN`. Not duplicated. |
| `build_completion_forest` (`fastpath/compiler_draft.py:619`) | Deterministic bounded enumerator over the Lark CFG path (prefix legality via `OpenUIIncrementalEngine`). | **Retained.** Serves as the default reference backend's enumeration primitive. Not duplicated. |
| `LatticeSearchState` + `Nogood` (`fastpath/lattice_search.py:182`, `:16`) | Bounded backtracking search trail with local nogood memory (`choose`/`rollback`). | **Extended (later issue).** Reversible decisions and local nogoods gain replayable certificates; certified deductions become non-reversible. |
| `RankedForest` / `rank_forest` (`fastpath/lattice_search.py:24`/`:44`) | Independent soft ordering over the hard compiler candidate set. | **Retained.** Grounds the invariant that learned modules rank/propose but never create legality. Not duplicated. |
| `ScopeContract` / `ScopeKind` (`data/progspec/scopes.py:33`/`:18`) | AST scopes (`COMPONENT_CALL`/`STATEMENT`/`CHILD_LIST`) plus a def/use binder overlay (`definitions`/`uses`/`visible_binders`). | **Extended.** Scope def/use edges feed capsule SCC construction; scopes are **not** assumed independent. |
| `dependency_closed_failure_cone` (`data/progspec/scopes.py:113`) | Despite the name, computes the least-common-ancestor of failing AST paths — **not** a dependency closure or SCC. | **Not reused for capsules.** Capsule SCCs are a new, distinct construction; this contract does not duplicate or overload this helper. |
| `ChoiceTokenizer` / `ChoiceDecodeState` (`models/choice_tokenizer.py:172`/`:608`) | Grammar-closed choice IR; length-aware legality (`allowed_ids`, `exhaustive_allowed_ids`, `minimal_completion_length`). | **Retained.** The choice IR is the late-realization and verification surface. Not duplicated. |
| `HoleId`, `SolverBounds`, `DomainValue`, `HoleDomain`, `FiniteDomainState` (`dsl/solver/state.py`) | Immutable JSON-safe finite-domain carrier with canonical hard-state identity and monotone operations. | **Implemented by VSS0-03.** Torch-free and not invoked by default; soft scores and proof artifacts remain outside the state. |
| `completion_forest_state` (`dsl/solver/adapters.py`) | One-hole projection of full compiler completion paths plus coverage and `UNKNOWN` provenance. | **Implemented by VSS0-03.** Retains the compiler as owner; a singleton solves only the projection. |
| `verification capsule`, `proof certificate` | Not implemented yet. | **Future VSS issues.** Capsule solving and replayable proof ownership are not duplicated here. |

## End-to-end example (partial coverage → live `UNKNOWN`)

Decode a `component_call` scope with hole `h_type` for the component-type slot.
`build_completion_forest` returns three prefix-legal candidates — `Button`,
`Card`, `IconButton` — and stamps `CompletionForest.coverage = "partial"` because
the pack's schema oracle for `IconButton`'s required `icon` enum is not loaded (an
unavailable pack capability: `DslPack.require("schema")` would raise
`PackSlotUnavailable` for that node).

Closure over `h_type.domain`:

- **`Button`** → a verifier-accepted witness completion exists (fill `label`,
  close the call). Verdict `SUPPORTED`; candidate stays live.
- **`Card`** → its finite child-list sub-domain has **complete** coverage, and
  exhaustive bounded enumeration finds no verified terminal (the surrounding scope
  structurally forbids the child that `Card` requires). Verdict `UNSUPPORTED`;
  removed **with** a replayable `UNSUPPORTED` certificate (destructive transition,
  cited).
- **`IconButton`** → its sub-domain coverage is **partial** (the missing schema
  capability), so no complete-coverage exhaustive search is possible. Verdict
  `UNKNOWN`. It is **not** removed: it remains live and may still be ranked or
  realized, but no `SUPPORTED`/`UNSUPPORTED` verdict is asserted and no
  certificate is issued.

The forest's overall `coverage` is `partial`, yet `Card` is still certified
`UNSUPPORTED` (its own sub-domain is fully covered) while `IconButton` stays a
live `UNKNOWN`. Removing `IconButton` would violate the invariant that `UNKNOWN`
never permits candidate removal.

## Research anchors

Fidelity labels follow the controlled vocabulary owned by
[research-lineage.md](research-lineage.md) (**Faithful / Adapted / Surrogate /
Adjacent**); the authoritative rows for this contract are added there under
*Verified scope solving & hybrid realization (VSS0)*. Because SLM-57 ships no
code, none of these is **Faithful** — the labels are lineage-only, consistent with
the X16-X21 convention ("lineage labels, not reproduced results").

- **DeepCoder** (Balog et al., ICLR 2017, [arXiv:1611.01989](https://arxiv.org/abs/1611.01989))
  — **Adjacent**. Learned search-guidance / candidate ranking; the contract keeps
  learned scores *soft* (ranking only, `rank_forest`) and does not reimplement it.
- **Counterexample-guided (neural) synthesis / CEGIS** — **Adapted boundary**. The
  deduction-vs-decision split and local nogoods adopt the counterexample →
  refinement principle; neural synthesis training is not reimplemented. See the
  existing *LLM-Modulo / CEGIS planning* anchor in research-lineage.md and R19/R23
  in [lattice-recursive-search.md](lattice-recursive-search.md).
- **egg / e-graphs** (Willsey et al., POPL 2021, [arXiv:2004.03082](https://arxiv.org/abs/2004.03082))
  — **Adjacent**. Post-realization canonicalization is motivated by equality
  saturation, but the contract is **not** an e-graph engine
  (cf. `iter-e252-canonicalizer-20260716.md`).
- **EDLM** (energy-based diffusion language model, 2024) — **Adjacent**. Energy /
  score-based candidate ranking is analogous to the soft-scoring layer; no EDLM
  training or energy head is implemented or assumed.
- **TreeDiff** (Zeng et al., 2025, [arXiv:2508.01473](https://arxiv.org/abs/2508.01473))
  — **Adapted boundary** (same as R13 in the V9 sources). Tree-edit diffusion
  informs late realization against AST/choice IR; architecture and training remain
  future empirical work.
- **Lattice Deduction Transformer** (Davis et al., 2026, [arXiv:2605.08605](https://arxiv.org/abs/2605.08605))
  — **Adapted** (same as R7 in the V9 sources). Monotone lattice projection plus
  rollback are the closure/deduction model; LDT architecture, alpha supervision,
  and training remain future work.

## Measured status

2026-07-17 — specification-only change (VSS0-01 / SLM-57). No solver code, no
runtime dependency, no experiment, no checkpoint, and no eval run; therefore no
measured results and **no model or ship claim**. This document defines the
guarantee boundary only. `python -m scripts.repo_policy` passes; documentation
changes select no test suites (`scripts/check_changed` skips `docs/`). VSS0-03 now
implements the carrier without decode integration; later VSS issues add proof and
search behavior behind a feature flag before any decode-behavior change.

2026-07-17 — VSS0-03 / SLM-59 implements the Torch-free finite-domain state,
canonical SHA-256 identity, monotone refinement/meet, reversible decision lineage,
compiler projection adapter, JSON round-trip, and focused regression coverage. It
does not implement recursive support search, a proof checker, capsule solving,
decode integration, or model scoring. No train, eval, benchmark, checkpoint, or
experiment ran; this is infrastructure only and makes **no correctness, readiness,
or ship claim**.

2026-07-17 — VSS0-04 / SLM-60 implements the deterministic enumerative support
oracle and pure certificate replay (`dsl/solver/support.py`) plus the OpenUI
expander/verifier wiring (`dsl/solver/openui_support.py`). Correctness is pinned by
tiny closed fixtures (`tests/test_dsl/test_solver_support.py`): SUPPORTED with a
witness digest, UNSUPPORTED only after exhausted complete-coverage search,
UNKNOWN for every partial-coverage/budget/unavailable condition, duplicate-state
suppression, deterministic ordering, and replay rejection of tampered digests,
non-exhausted `UNSUPPORTED`, incomplete coverage, and stale versions. The oracle is
**not** wired into generation; default decode behavior is unchanged. Verifier
`UNAVAILABLE`/timeouts never become `UNSUPPORTED`. This is a reference correctness
implementation — no train/eval/benchmark/checkpoint ran and **no ship or
speed claim** is made. Verified: `python -m pytest
tests/test_dsl/test_solver_support.py tests/test_dsl/test_solver_state.py
tests/test_models/test_compiler_decode.py -q` (92 passed) and
`python -m scripts.repo_policy`.
