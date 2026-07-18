# Checkpoint provenance (fail-closed)

Makes every checkpoint cited by a campaign, frontier comparison,
README/model-card row, or follow-on issue **resolvable and verifiable from a
fresh clone**. A local/gitignored path is never accepted as evidence for a
frontier or ship-grade conclusion.

Contract owner: `src/slm_training/harnesses/model_build/checkpoint_reference.py`
(schema) and `checkpoint_bucket.py` (sync). Audit:
`scripts/verify_checkpoint_references.py`. Backfill:
`scripts/backfill_checkpoint_references.py`. Related:
[checkpoint-bucket.md](checkpoint-bucket.md).

## Claim classes

Every reference declares one, ordered by evidentiary strength:

| Claim class | Durable remote required | May stay local-only | Backs a ship/frontier claim |
| --- | --- | --- | --- |
| `fixture` | no | yes | never |
| `diagnostic` | no | yes | never |
| `frontier` | **yes** | no | yes |
| `ship_candidate` | **yes** | no | yes |

`fixture`/`diagnostic` are wiring / exploratory evidence and may live only in a
gitignored `outputs/` path as long as they are honestly labeled.
`frontier`/`ship_candidate` are **fail-closed**: they cannot be published unless
every required provenance field is present and the artifact has been verified.

## `CheckpointReferenceV1`

An immutable, JSON-safe record with a stable content hash. Missing provenance is
never inferred from a filename — it stays the explicit sentinel `UNKNOWN` and
blocks durable publication. Fields: schema/claim class; `run_id`, role, filename,
byte size, SHA-256; durable `remote_uri` + `bucket_id`; `training_source_commit`
and `evaluation_source_commit`; parent URI/hash; model config / tokenizer /
output codec hashes and context-tower identity; corpus manifest + split hashes
and data version; train steps/tokens and seed; sync + verification timestamps and
verifier version; and the immutable inventory of uploaded companion files.

Publication gate (`blocking_reasons()` / `require_publishable()`): a
`frontier`/`ship_candidate` reference must carry `remote_uri`, `bucket_id`,
`size_bytes`, `sha256`, both source commits, the config/tokenizer/codec hashes,
the corpus manifest hash + data version, and a `verification_timestamp` +
`verifier_version`. Any `UNKNOWN` among these blocks publication.

## Sync (fail-closed)

`sync_run_checkpoints(...)` / `maybe_sync_train_checkpoints(...)` now:

1. hash every artifact (streaming SHA-256) **before** upload;
2. write a `<checkpoint>.ref.json` sidecar per checkpoint plus an aggregate
   `checkpoint_references.json` manifest into the uploaded set;
3. for a **real** sync, re-plan the upload as a dry run and confirm every file
   landed remotely — only then are the references stamped `verified` and the
   verified sidecars pushed; a positive mismatch raises (train fails closed);
4. return the durable URI, the size + SHA-256 inventory, the reference blobs,
   and the verification verdict (persisted to `checkpoint_bucket.json` /
   `train_summary.json`).

A **dry run is not persistence evidence**: it hashes and builds references but
never verifies them, so a dry-run reference can never back a durable claim.

```bash
# Real sync of a frontier checkpoint (requires HF auth + huggingface_hub)
python -m scripts.sync_checkpoints \
  --run-dir outputs/runs/<run_id> --run-id <run_id> \
  --claim-class frontier --ensure-bucket
```

`--claim-class` defaults to `diagnostic`; `--training-source-commit` defaults to
the current git HEAD; extra provenance can be supplied via `--provenance-json`.

## Audit

`python -m scripts.verify_checkpoint_references --check` (wired into CI) scans
`docs/design/*.json` for structured references (embedded, manifest, or `.ref.json`
sidecar — no free-form regex) and fails closed when a `frontier`/`ship_candidate`
reference is under-provenanced, unresolvable, unverified, or byte-mismatched. It
also rejects duplicate `(run_id, role)` references mapping to different SHA-256
and conflicting training source commits. `fixture`/`diagnostic` references are
allowed to stay local-only; they fail only on a positive byte mismatch. Model
card / README changes route to this audit via `scripts/check_changed.py`.

## Recovery — making an unresolved checkpoint durable

Historical frontier checkpoints are gitignored under `outputs/` and absent from a
clone (see the backfill report,
[checkpoint-reference-backfill-20260717.md](checkpoint-reference-backfill-20260717.md)).
To make one durable and citable as a frontier:

1. on the host that still holds `outputs/runs/<run_id>/checkpoints/`, set
   `HF_TOKEN` (or `hf auth login`);
2. run the `--claim-class frontier --ensure-bucket` sync above;
3. store the returned `remote_uri` + reference in the result JSON and update the
   model card / README rows;
4. `python -m scripts.verify_checkpoint_references --check` must pass.

## Honesty boundary

The audit gates **durable claims**, never honest local-scratch rows, and it does
not weaken any ship gate. Meaningful-parse remains the primary quality metric;
provenance is orthogonal — a fully-provenanced checkpoint is still not a ship
claim without the full honest scoreboard (see
[`AGENTS.md`](../../AGENTS.md) and the `honest-ship-eval` skill).
