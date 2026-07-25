# DSH0-07 capability progression gates

Date: 2026-07-23  
Issue: SLM-351  
Disposition: contract and enforcement implemented; no certificate issued

This change makes staged progression executable and fail closed. It is a
governance/harness change, not a train, eval, benchmark, checkpoint, capability
claim, or ship claim.

## Contract

`src/slm_training/harnesses/capability_gates.py` owns three immutable,
content-addressed records:

- `CapabilityGateSpecV1` pins the target capability, confidence-bound metric
  thresholds, and exact retention-suite hashes.
- `CapabilityGateResultV1` binds the spec, gate implementation, checkpoint
  reference and bytes, dataset manifest, eval suites, code, and config. Only a
  completed `ship_eval` result can certify.
- `CapabilityCertificateV1` repeats those identities, binds the exact gate
  result and complete ordered prior-certificate chain, records human or CI
  promotion authority, and carries distillation permission independently.

The similarly named `CapabilityCertificateV1` in `capability_artifacts.py`
remains the existing synthesis-artifact disposition record. It is not accepted
as a progression certificate.

Higher stages require all earlier certificates: CAP1 requires CAP0; CAP2
requires CAP0 and CAP1. Every retention suite named by the gate spec must be
present and passing. Fixture/diagnostic checkpoints, fixture evidence,
diagnostic runs, interrupted runs, invalid runs, and timeouts cannot issue a
certificate.

## Training preflight and lever profiles

`train_model` accepts `--requested-capability`, the exact `--capability-plan`,
repeatable `--capability-certificate`, and the separate
`--capability-distillation` request. The canonical training loop validates the
chain and lever profile
immediately after feature-flag resolution, before corpus loading, accelerator
initialization, checkpoint loading, or model construction. The model factory
repeats the check as defense in depth.

Preflight loads the explicit checked-in synthesis plan, verifies its immutable
ID and hash against the staged dataset manifest, and requires its capability to
equal the requested training capability. Missing, mismatched, or differently
staged plan/manifest pairs fail.

- CAP0 forbids schema/natural-language conditioning, teacher initialization,
  and operator/action levers.
- CAP1 permits schema/natural-language conditioning.
- CAP2 additionally permits operator/action levers.
- Teacher initialization requires the explicit distillation request, and that
  request requires a distillation-enabled prior certificate.

Unclassified legacy invocations remain unchanged; staged invocations opt into
the fail-closed contract explicitly.

## Promotion interface

`python -m scripts.manage_capability_certificate dry-run ...` validates all
evidence and prints the would-be immutable certificate ID without writing it.
`promote` requires an output plus either explicit human confirmation
(`--authority human --confirm-human`) or CI attestation
(`--authority ci --ci-attested`). Promotion never infers missing identities or
converts fixture-only evidence.

## Verification

Focused tests cover identity immutability, complete ordered prerequisites,
retention regression, all non-certifying terminal states, fixture rejection,
lever profiles, independent distillation permission, and failure before data
or model initialization. No train/eval/benchmark was run, so no AgentEvals or
AgentV result bundle is applicable and no model-card update is warranted.
