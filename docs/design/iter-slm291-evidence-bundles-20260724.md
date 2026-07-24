# SLM-291 / LAR0-05: Content-addressed evidence + preregistered claim bars

Matrix set: `slm291-evidence-bundles` · Version: `slm291-v1` · Status: **complete** · Claim class: wiring

## What was built

- **`EvidenceBundleV1`** (`src/slm_training/harness_core/evidence_bundle.py`) — fail-closed provenance envelope: source commit + dirty-tree status, config digest, seeds, environment digest, corpus and suite hashes, checkpoint SHA-256/size/URI, raw envelopes, pinned evaluator and gate versions, claim class. Reuses `DURABLE_CLAIM_CLASSES` from `checkpoint_reference`; non-fixture claims fail closed with explicit `blocking_reasons()`.
- **Content-addressed store** — `LocalEvidenceStore` (create-only, digest-verified `objects/<aa>/<sha256>`) behind the pluggable `EvidenceStore` protocol; store-addressed artifacts use `cas://sha256/<digest>` URIs.
- **`verify_evidence_bundle`** — deterministic, no-write; hash-checks checkpoint bytes, suite bytes, and raw envelopes; reports "missing" / "altered" / "not resolvable" per artifact.
- **Census CLI** — `scripts/verify_evidence_bundles.py` over committed `docs/design/*.json` bundles plus the `docs/MODEL_CARD.md` roster; a roster row positively claiming promoted/champion/frontier/ship with a non-durable location fails closed (exit 1).
- **`PreregisteredClaimBarV1`** — versioned bar: target metrics + thresholds, min sample size, seeds, confidence level / required lower bound, stop rule, entry gate, falsifier; `validate_claim_bar` / `require_valid_claim_bar` are the entry point new LAR harnesses must call.

## Census results (2026-07-24)

| Metric | Count |
| --- | --- |
| Committed `evidence_bundle/v1` payloads | 0 (verified from now on) |
| Roster rows scanned | 166 |
| Durable remote (`hf://`) | 10 (9 real + 1 template) |
| Repo-local (`outputs/`, git fixture) | 148 |
| Ephemeral local (`/tmp`) | 6 |
| Rows positively claiming a durable class | 0 |
| Fail-closed failures | 0 |

Every roster row is honestly classified today: the six `/tmp` references and
148 local references all carry explicit "not promotable or ship" disclaimers,
and the nine durable bucket checkpoints claim only diagnostic status.

## Retroactive adjudication

`docs/design/slm291-evidence-bar-adjudications-20260724.json` applies the bar
append-only (hash-chained events) to the nine durable bucket rows
(E396…E531): each **satisfies the bar as diagnostic evidence** and is **below
the frontier/ship_candidate bar**, because no `evidence_bundle/v1` payload
(source commit/dirty tree, config digest, seeds, environment digest, corpus /
suite hashes, pinned evaluator/gate versions) was recorded for those runs.
Historical rows are unchanged; adjudications are separate records.

## Tests

`tests/test_harness_core/test_evidence_bundle.py` +
`tests/test_scripts/test_verify_evidence_bundles.py` — 26 tests: valid bundle,
missing checkpoint, altered checkpoint bytes, altered suite bytes, dirty
source tree, fixture claim with local path, production claim pointing at
`/tmp`, unpinned evaluator, CAS roundtrip + corruption fail-closed,
store-addressed artifacts, claim-bar schema/validation edge cases, census
classification/determinism, CLI exit codes. All verification is no-write.

## Honest caveats

- The roster claim detector is keyword-based (positive durable words without
  negation words); it is a census heuristic, not a replacement for the
  fail-closed `CheckpointReferenceV1` audit in
  `scripts/verify_checkpoint_references.py`.
- Durable remote artifacts without a local copy are trusted via their
  recorded sync-time verification (checkpoint_bucket), not re-downloaded.
- No experiment was run; nothing here changes gates, metrics, or defaults.
