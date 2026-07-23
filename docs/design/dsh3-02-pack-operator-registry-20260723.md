# DSH3-02 pack-owned operator registry

SLM-370 adds the pure execution and replay boundary above the DSH3-01
contracts. It does not add a production operator family. A `DslPack` may expose
one immutable `operator_library`; absent capabilities continue to fail through
the existing `DslPack.require` path.

## Execution contract

`OperatorLibraryV1` stores one declaration and pure executor per operator ID.
Its registry fingerprint covers sorted declaration payloads, never Python
callables, display names, token IDs, or learned vectors. Every call requires
the library to be the exact object owned by the supplied pack.

Both `dry_run` and `apply` route through one internal path:

1. Reconstruct and verify the input `OperatorStateV1` from canonical source.
2. Resolve the declaration and validate required slot IDs and reference kinds.
3. Execute the pure callback against the frozen input state.
4. Reject any input mutation.
5. Run the resulting source through the pack's parse/serialize,
   static/schema oracle, scope extractor, property-order provider,
   canonicalizer, and canonical round trip.
6. Return either a new immutable state plus `OperatorApplicationV1` proof, or
   a typed rejection with no effect/after-state claim.

`dry_run` discards the returned state but preserves the exact application
identity, so its legality verdict and evidence match `apply`. Unsupported
operators receive a deterministic `operator.unsupported` rejection.

## Replay and authority

`replay` requires exact before-state and AST digests, resolves the recorded
operator fingerprint in the same immutable registry, re-executes with the
recorded typed arguments and provenance, and compares the complete application
identity. Missing operators, changed input state, nondeterministic execution,
authority drift, or changed evidence fail closed.

The OpenUI built-in pack intentionally has no registered operators in this
change. DSH3-04 owns the first production local operator family. Tests attach a
fixture registry to an immutable copy of the real OpenUI pack, so successful
fixture rewrites pass the actual ordinary OpenUI authorities. `toy-layout`
continues to fail closed because it has neither an operator library nor the
canonical/oracle authorities required for operator execution.

## Verification

`tests/test_dsl/test_operator_registry.py` covers:

- registry order invariance and lookup;
- dry-run/apply identity and input immutability;
- real OpenUI pack-authority success;
- required/type rejection and invalid-source rejection;
- unsupported operators;
- exact replay and changed-state failure;
- cross-library ownership and forged-state rejection;
- partial-pack failure.

This is contract/unit evidence only. No train, eval, benchmark, matrix,
checkpoint, model card, operator capability, or ship claim was produced.
