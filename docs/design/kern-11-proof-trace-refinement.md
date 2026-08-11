# KERN-11 — Fixture-level runtime trace refinement (SLM-539)

## Claim

Canonical production replay traces for a **supported bounded fixture subset**
refine the abstract Lean decode/search semantics. The runtime is untrusted —
only its sealed INTEG-01 compact proof trace is checked.

This theorem is **fixture/subset-scoped**. It does **not** claim full PyTorch
semantics or physical latency equivalence (those remain empirical remainder on
the `implementation_refinement` axis).

## Supported fixture subset

| Fixture id | Source | Expected cost | Expected forwards |
| --- | --- | ---: | ---: |
| `bundle_stats_mechanism` | INTEG-01 frozen `proof_trace_fixtures.v1.json` | 28 | 11 |

## Explicit assumptions

| Assumption | Rule |
| --- | --- |
| Domain completeness | `legal_domain_status=complete` requires a non-empty digest; bare `coverageComplete` never authorizes (KERN-02) |
| Judgments | Unknown / invalid / missing judgments never authorize removal (KERN-01) |
| Event ordering | Ordinals must be dense `0 .. n-1` in list order |
| Identities | Event identity digests must match the sealed envelope `request_id` |
| Cost counters | Declared `observed_work.abstract_cost` / forwards must match expected DecodeUnitWork totals (KERN-06) |
| Model forwards | `neural_forward > 0` requires a `ranker_model_invocation` event |

## Checker / replayer

| Surface | Role |
| --- | --- |
| Lean `LeverProofLean.ProofTraceRefinement` | Compact events, `checkTrace` / `replay`, rejection theorems, fixture theorem |
| Python `formal/trace_refinement.py` | Mirror checker over `CanonicalProofTraceV1`, injectors, certificate |
| `formal_authority/v2` | Successful checks bind via `bind_refinement_authority` + Lean claim catalog |

Rejected shapes: illegal commits, stale-state domains, fabricated
completeness/refutation, reordered events, omitted model forwards, mismatched
identity digests (plus integrity failures on unsealed mutations).

## Acceptance

- Injected invalid traces fail (semantic reject after reseal, or integrity
  reject when hash is left stale).
- Valid frozen runtime fixture `bundle_stats_mechanism` replays successfully.
- Scope disclaimer theorems stay `fixture_subset_scoped=true` with PyTorch /
  latency claims false.

## Tests

`tests/test_formal/test_trace_refinement.py` +
`resources/formal/trace_refinement_fixtures.v1.json`.
