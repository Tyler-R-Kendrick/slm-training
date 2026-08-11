# KERN-02 — Proof-bearing finite domains (SLM-517)

## Claim

Authority-critical completeness is a *proved* finite domain, not a Boolean.
`coverageComplete` / `coverage_complete` remains readable telemetry and
historical-artifact payload only; a forged true bit has zero semantic
authority for pruning or forced emission.

## Lean

- Module: `LeverProofLean.CompleteDomain`
- Structure: `CompleteDomain` with `stateId`, `candidates`, `legal`, plus
  proofs `sound : Subset candidates legal` and `complete : Subset legal candidates`
  (and `nodup`).
- Decode boundary: `DecodeInvariants.commitComplete` / `singleton_bypasses_ranker`
  require a proved domain; legacy `Domain.coverageComplete` never forces bypass
  (`forged_coverage_complete_never_bypasses`, `coverage_complete_flag_not_singleton`).
- Exact-closure bridge: `ExactClosure.finiteSearchAuthorized` rejects partial,
  unknown, and stale-state domains.

## Python

- Adapter: `slm_training.formal.complete_domain`
- `build_complete_domain` checks soundness/completeness against an explicit
  legality extension.
- `from_legacy_coverage_flag` preserves the Boolean for migration / telemetry
  and never sets `checked_sound` / `checked_complete`.
- Structural law `singleton_bypass` is demoted (Boolean-only rows expect false);
  `proved_complete_singleton` is the authoritative classifier.

## Empirical boundary

Runtime decode backends still enforce singleton bypass via CI
(`forwards_count == 0`, compiler coverage checks). This change proves the
*authority contract*; end-to-end PyTorch/ONNX wiring remains the existing
empirical gate, not a Lean claim about a specific backend.
