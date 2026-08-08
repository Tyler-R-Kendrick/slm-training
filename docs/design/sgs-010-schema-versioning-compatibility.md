# SGS-010 — Schema versioning, migrations, and compatibility for synthesis contracts

Linear: [SLM-445](https://linear.app/quickdeploy-ai/issue/SLM-445). Component:
`governance.ownership_map` **v7**. Blockers closed by SGS-003 (SLM-437),
VCE-001 (SLM-438), SRP-001 (SLM-441), PCT-004 (SLM-440).

## Policy

Every serialized contract this initiative introduces obeys the same four rules,
and both halves of the rule are executable — nothing here relies on prose.

1. **Explicit version.** The contract declares a version symbol in its owner
   module and serializes it in every payload.
2. **Readers fail closed.** The reader consults that symbol and raises on any
   mismatch. An unrecognized version is never coerced, defaulted, or
   reinterpreted — the same fail-closed rule decode uses (I6).
3. **Missing version is a rejection**, not "assume current". Absence carries no
   information, so it can never be read as agreement.
4. **Semantics-changing migrations mint a new identity.** An existing version
   number is never redefined in place, which is exactly why historical
   artifacts stay readable: an old payload still means what it meant when it
   was written, or it is explicitly unsupported.

## Enforcement

| Surface | What it certifies |
| --- | --- |
| `scripts/verify_ownership_map.py` → `SERIALIZED_CONTRACTS` | Static (AST): the contract class exists, defines its version symbol, and its **class-scoped** reader both mentions that symbol and contains a `raise`. Runs in the dependency-light CI job and in `check_changed`. |
| `tests/test_data/test_synthesis_contract_versions.py` | Runtime, table-driven: current version round-trips losslessly, a future version is rejected, a missing version is rejected, and the runtime table matches `SERIALIZED_CONTRACTS` exactly. |
| `tests/test_scripts/test_verify_ownership_map.py` | Negative control: deleting a reader's version comparison makes the certificate fail with `never consults`. |
| `python -m scripts.verify_version_stamps --check` | The registry bump that must accompany any change to the above. |

The registry-parity assertion is the load-bearing part: adding a contract to
one surface and not the other fails immediately, so the two lists cannot drift
into the "map that reads as current while lying" failure SGS-001 catalogued.

## Covered contracts

| Contract | Owner module | Version symbol | Current |
| --- | --- | --- | --- |
| `PromptSemanticRequirementsV1` | `data/progspec/prompt_requirements.py` | `REQUIREMENTS_SCHEMA_VERSION` | `1` |
| `VerifiedSynthesisProblemV1` | `data/progspec/synthesis_problem.py` | `SYNTHESIS_PROBLEM_SCHEMA_VERSION` | `1` |
| `VerifierWitnessV1` | `evals/semantic_failure.py` | `WITNESS_SCHEMA_VERSION` | `verifier_witness/v1` |
| `DecodeStatsRecordV1` | `models/decode_stats.py` | `DECODE_STATS_RECORD_SCHEMA` | `decode_stats_record/v1` |

Symbolic-regression problem/AST/certificate artifacts (SRP-002+) and
static-artifact provider records join this table as they land; the parity
assertion is what forces that, since a contract added to the runtime table
without a `SERIALIZED_CONTRACTS` row fails the suite.

## Validation

```bash
python -m scripts.verify_ownership_map            # includes serialized_contracts
python -m pytest tests/test_data/test_synthesis_contract_versions.py \
                 tests/test_scripts/test_verify_ownership_map.py -q
python -m scripts.verify_version_stamps --check
```

Result: 17 + 13 tests pass; the certificate emits
`"serialized_contracts": ["PromptSemanticRequirementsV1",
"VerifiedSynthesisProblemV1", "VerifierWitnessV1", "DecodeStatsRecordV1"]`.
