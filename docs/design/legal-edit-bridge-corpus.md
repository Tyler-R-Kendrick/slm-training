# Canonical legal-edit bridge corpus

`LegalEditBridgeRowV1` is the objective-neutral supervision boundary between
the canonical edit algebra and any model that scores exact live edits. The
corpus builder owns membership, replay evidence, support labels, provenance,
and split safety. It does not own a loss, rate parameterization, decoder, or
model-selection decision.

## Exact candidate membership

Every row references an `ExactLegalEditCandidateSetV1` artifact by the SHA-256
of its canonical payload. Candidates are enumerated only from the current
program and a request-visible `RequestEditContractV1` containing typed
productions, slots, statement identities, and enums. Target-directed planner
helpers are never used to decide membership.

Candidate IDs hash the semantic `CanonicalEdit` fields and exclude planner
UUIDs, wall time, cost diagnostics, and input ordering. Request-local node and
slot pointers remain per-request feature ordinals; they do not allocate
permanent edit-tuple vocabulary rows. Applying each candidate produces its
successor fingerprint and an independently replayable
`TransitionCertificateV1`. Candidate files and exact-state files reject
filename, payload, certificate, source, or successor digest mismatches.

The model-facing view is an explicit allowlist:

- state summary, step index, normalized progress, and sampled time;
- focus and dependency capsule available at the current state;
- action, production, arity/cardinality, request-local pointers,
  literal/enum/frame fields, and verified successor fingerprint.

It recursively rejects confirmation, future-witness, hidden-gold, target AST,
planner-selection, and positive-label fields. Labels and diagnostics remain
outside that view. Model outputs can score the ragged batch but cannot change
membership.

## Labels and splits

All target-consistent next edits are positives; the planner-selected edit is a
diagnostic only. `supported`, `unsupported`, and `UNKNOWN` are disjoint and
exhaust the exact live set. `UNKNOWN` is never part of the negative mask.
Multi-positive targets are uniform over certified positives and remain keyed
by candidate ID under permutation.

Alternate paths for one target share `target_cluster_id`. Both
`target_cluster_id` and `split_group` must map to exactly one train/dev split.
Confirmation rows are forbidden until the later single-touch experiment.

## Publication gate and storage

Production corpora belong under `outputs/data/train/<dataset_id>` and use the
common `DataStore` manifest with immutable artifact hashes. A tiny wiring
fixture is committed at
`src/slm_training/resources/data/train/slm196_legal_edit_bridge_fixture`.

Production builds require a frozen selected planner manifest with:

- an SLM-184 claim-ledger digest and confirmation disabled;
- measured reachability at or above 95%;
- 100% transition replay;
- a supported selected planner/source policy.

The committed SLM-189 evidence remains fixture-only and inconclusive, so the
development manifest cannot pass the production gate. Missing hash-pinned X22
or solver baselines are reported as unavailable rather than converted into an
assumed coverage gain.

## CLI

```bash
python -m scripts.build_legal_edit_bridges --describe
python -m scripts.build_legal_edit_bridges --fixture
python -m scripts.build_legal_edit_bridges \
  --records records.jsonl \
  --planner-manifest planner.json \
  --output outputs/data/train/<dataset_id>
python -m scripts.build_legal_edit_bridges --output <dataset> --validate
python -m scripts.build_legal_edit_bridges --output <dataset> --stats
```

Every build emits `quality_report.json`, `rejected.jsonl`, and
`synthesis_feedback.json`. Producer failures must be repaired at the edit
algebra or planner; corpus gates are not tuning knobs.
