# SLM-306: verify the locked promotion manifest digest against real bytes

SLM-285 (`docs/design/slm285-locked-promotion-manifest.md`) locked an
immutable, all-view-decontaminated promotion holdout at
`src/slm_training/resources/data/eval/manifests/abstract_planning_locked_v1.jsonl`
and required every `promotion_candidate` / `ship_gate`
`ExperimentCampaignV1` to declare its `locked_eval_manifest_sha256`.
`validate_result_claim` already rejected a missing digest or a digest that
disagreed between the campaign manifest and its result
(`locked_eval_manifest_sha256_missing` / `_mismatch`).

What it never checked: whether that declared digest corresponded to any real
manifest bytes at all. It is a free-form 64-character hex string, validated
only by a regex pattern and by self-consistency between the manifest and the
result. `tests/test_autoresearch/test_experiment_campaign.py`'s own fixture
used the literal digest `"e" * 64` -- not a real digest of anything -- and
`validate_result_claim` returned `()` (no failures) for it. A campaign author
could declare an arbitrary correctly-shaped digest, or a stale digest from a
manifest that no longer exists, and the SLM-285 gate would not notice.

## What changed

`src/slm_training/data/locked_eval_manifest.py` gained three functions:

- `canonical_manifest_path()` -- the one committed locked manifest's path.
- `load_locked_manifest_payload(path)` -- loads a manifest file and fails
  closed (`ValueError`) if its schema is unsupported or if its own declared
  `manifest_sha256` does not match `sha256({schema, rows, metadata})`
  recomputed with the exact canonical encoding `LockedManifest.sha256`
  already uses. This also catches a hand-edited manifest file whose author
  forgot (or declined) to update the digest field.
- `verify_locked_manifest_digest(path, expected_sha256)` -- `True` only when
  `path` is an untampered manifest whose digest equals `expected_sha256`.

`validate_result_claim` (`src/slm_training/autoresearch/experiment_campaign.py`)
gained an optional `locked_manifest_path: Path | None = None` parameter. When
a claim's `locked_eval_manifest_sha256` passes the existing missing/mismatch
checks and a path is supplied, the digest is now independently re-derived
from real bytes on disk; a mismatch (or a missing/tampered file) adds
`locked_eval_manifest_digest_unverified_on_disk` to the failures. The same
parameter was threaded through `CampaignStore.validate_campaign_result`
(`storage.py`) and through the promotion entrypoints that wrap it --
`evaluate_promotion`, `register_promoted_checkpoint`, and
`load_campaign_governance` (`src/slm_training/harnesses/experiments/promotion.py`).

The parameter defaults to `None` everywhere it was added, so every existing
caller and test that never had a manifest file to point at keeps its prior
(self-consistency-only) behavior unchanged -- this is additive, not a
breaking re-gate. No call site was defaulted to
`locked_eval_manifest.canonical_manifest_path()` automatically; callers that
want the stronger, content-addressed check opt in explicitly. That is a
deliberate scope limit for this change, noted below for follow-up.

## Evidence (local CPU, fixture/audit scale)

`docs/design/slm306-locked-manifest-digest-verification-20260725.json`
records four `validate_result_claim` scenarios reproduced against the real
committed manifest (286 rows; `manifest_sha256`
`b4ad49...e0a2d48`):

| scenario | `locked_manifest_path` | digest | failures |
| --- | --- | --- | --- |
| prior behavior | not supplied | forged (`"e"*64`) | `()` -- gap, now historical |
| new gate | supplied | forged (`"e"*64`) | `('locked_eval_manifest_digest_unverified_on_disk',)` |
| new gate | supplied | real | `()` |
| new gate, file missing | supplied (absent path) | real | `('locked_eval_manifest_digest_unverified_on_disk',)` |

`load_locked_manifest_payload` also recomputes the real manifest's digest
from its `rows`/`metadata` and confirms it equals the file's own declared
`manifest_sha256` (i.e. the SLM-285 manifest itself is not tampered).

This is a governance/gate wiring audit only. No model quality, promotion, or
training claim is made here, and nothing about the SLM-285 manifest content,
partitions, or decontamination views changed.

## Honesty and scope

Fixture/local-CPU evidence, not a ship or promotion claim. The new check is
optional and off by default; it strengthens the gate for callers that supply
`locked_manifest_path`, but a campaign author who omits it still only gets
the pre-existing self-consistency check. Wiring
`canonical_manifest_path()` in as the default for the real promotion CLI
entrypoints (`scripts/resume_climb.py`, `scripts/run_scaling_ladder.py`, and
`ModelBuildConfig.register_promoted` in `train_loop.py`) is a natural,
still-narrow follow-up once a real end-to-end promotion run exercises that
path; it was intentionally left out here to avoid changing default gate
behavior without a live promotion run to validate against.

## Version stamps

- `data.locked_eval_manifest`: v2 -> v3
- `harness.autoresearch.experiment_campaign`: v5 -> v6
- `harness.experiments`: v87 -> v88

See `src/slm_training/resources/versions.json` for the full history notes.
