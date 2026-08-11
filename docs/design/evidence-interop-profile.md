# EVID-12 — Evidence interoperability profile (SLM-543)

Thin interoperability profile that **projects** repository-native formal and
experiment evidence onto established provenance / attestation / artifact
standards. External formats never replace canonical authority.

| Item | Value |
| --- | --- |
| Profile id | `slm-evidence-interop/v1` |
| Linear | [SLM-543](https://linear.app/quickdeploy-ai/issue/SLM-543) / EVID-12 |
| Package | `src/slm_training/evidence_interop/` |
| Tests | `tests/test_evidence_interop/` |

## Authority law

Repository IDs, content digests, and checker capabilities remain authoritative:

| Native source | Authority-relevant identifiers |
| --- | --- |
| `FormalObjectV1` | `object_id`, `content_digest`, `required_checkers` |
| `EvidenceBundleV1` | `run_id`, `config_sha256`, `corpus_sha256`, checkpoint / suite / envelope digests |
| `ExperimentCampaignV1` | `campaign_id`, `locked_eval_manifest_sha256`, `source_commit` |
| `DataSnapshot` | `snapshot_id`, `records_sha`, `annotations_sha` |

External PROV / RO-Crate / SPDX / in-toto / OCI documents are **projections
only**. `import_*` always returns `ProjectionView(semantic_authority=False)`.
Importing a provenance record alone never constructs a certified
`FormalObjectV1`, never locks a campaign, and never clears ship gates.

## Format mapping (thin)

| Standard | Projection module | Maps |
| --- | --- | --- |
| [W3C PROV-O](https://www.w3.org/TR/prov-o/) | `export_prov.py` | entity / activity / agent / derivation |
| [RO-Crate](https://www.researchobject.org/ro-crate/) | `export_rocrate.py` | RootDataset + File descriptors |
| [SPDX 3](https://spdx.github.io/spdx-spec/v3.0.1/) | `export_spdx.py` | Package / Dataset / Build / AIPackage shaped elements |
| [in-toto](https://github.com/in-toto/attestation) / [SLSA](https://slsa.dev/) | `export_intoto.py` | Statement + test-result or provenance predicate (unsigned) |
| [OCI image-spec](https://github.com/opencontainers/image-spec) referrers | `export_oci.py` | subject descriptor + authority referrer |

Every projection embeds `slm:profile`, `slm:kind`, `slm:authority`, and
`slm:semantic_authority: false`. Exporters are pure stdlib JSON — **no
mandatory external service**, registry, or signing backend.

## Import validation

`import_projection` / `import_*`:

1. Require a recognizable format and `slm:authority.authority_ids`.
2. Collect unknown top-level / `prov:*` / `spdx:*` predicates into
   `unsupported_fields` (or raise `UnsupportedFieldError` when
   `raise_on_unsupported=True`).
3. Force `semantic_authority=False` even if the document claims otherwise.
4. Never call native constructors as “certified” from the projection alone.

`roundtrip` / `roundtrip_all` export then import and raise
`AuthorityRoundtripError` if any authority-relevant identifier drifts.

## Threat model — what each standard does **NOT** prove

| Standard | Does **not** prove |
| --- | --- |
| **PROV-O** | Ship-gate passage, Lean kernel success, checker independence, or that a derivation was honestly measured. PROV records graph structure; it does not certify semantics. |
| **RO-Crate** | Experiment honesty, publishability of an `EvidenceBundleV1`, or that packaged files match live corpus digests beyond the projected checksums. |
| **SPDX 3** | That `required_checkers` ran, that an AI package is safe to promote, or that a dataset cleared decontamination / ship gates. SPDX identifies packages and checksums. |
| **in-toto** | Semantic authority. A `PASSED` test-result predicate in this profile means “projection packaged”, not “FormalObject loop closed”. Unsigned envelopes prove nothing about signer identity. |
| **SLSA provenance** | Build hermeticity or supply-chain level claims for this repo’s exports. The projected `slsa_provenance` body is documentary packaging only. |
| **OCI referrers** | Registry presence, signature validity, or that an attached blob authorizes decode / promotion. Subject/referrer links are local descriptors. |

### Shared non-claims

- No projection upgrades `semantic_authority`.
- No projection bypasses `FormalObjectV1` digest recomputation or campaign
  manifest lock digests.
- No projection is a substitute for `--ship-gates` or constrained decoding (I6).
- Absence of an external service is intentional: offline, fail-closed, local.

## Acceptance (EVID-12)

- Round-trip projections preserve all authority-relevant identifiers or fail
  explicitly (`AuthorityRoundtripError`).
- Importing a provenance record alone never upgrades semantic authority.
- No mandatory external service.

## Related

- Canonical formal objects: `docs/design/formal-objects-multi-prover.md`
- Evidence bundles: `docs/design/evidence-store.md`,
  `harness_core/evidence_bundle.py`
- Campaign governance: `docs/design/experiment-campaign-governance.md`
- Research catalog ownership (parallel): RESEARCH-01 / SLM-518 — this profile
  cites standards; it does **not** own the research catalog.
