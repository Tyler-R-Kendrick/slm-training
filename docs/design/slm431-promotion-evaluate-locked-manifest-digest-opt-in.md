# SLM-431: wire the locked-manifest digest check into `POST /api/promotion/evaluate`

SLM-306 (`docs/design/slm306-locked-manifest-digest-verification.md`) gave
`validate_result_claim` (`src/slm_training/autoresearch/experiment_campaign.py`)
an optional `locked_manifest_path: Path | None = None` parameter. When
supplied, a campaign's self-reported `locked_eval_manifest_sha256` is
independently re-derived from the real bytes of the committed locked manifest
(`src/slm_training/data/locked_eval_manifest.canonical_manifest_path()`)
instead of only being trusted as a free-form hex string. SLM-430
(`docs/design/slm430-promotion-cli-locked-manifest-digest-opt-in.md`) wired an
explicit, default-off `--verify-locked-manifest-digest` opt-in into the three
real training/CLI promotion entrypoints that call this primitive: `train()`
(`ModelBuildConfig.register_promoted`), `scripts/resume_climb.py::main`, and
`scripts/run_scaling_ladder.py::main`.

SLM-430's own writeup scoped itself narrowly to those three entrypoints and
did not touch `src/slm_training/web/routes.py`. Reading that file confirmed a
fourth reachable caller of `validate_result_claim` that SLM-430 left
untouched: the pure-compute `POST /api/promotion/evaluate` endpoint, which
powers the Checkpoints dashboard's live gate editor
(`src/apps/dashboard/src/pages/Checkpoints.tsx` and the OpenUI interpreted
mode). Its handler already called

```python
governance_failures = validate_result_claim(
    manifest, governed_result, artifact_root=payload.campaign_artifact_root
)
```

-- never passing `locked_manifest_path`, so there was no way to request the
stronger, content-addressed check from this endpoint either, matching the same
"opt-in mechanism exists in library code with no reachable caller" gap SLM-430
closed for the CLI/train entrypoints.

This change is distinct from both SLM-306 (built the primitive) and SLM-430
(wired the primitive into the three offline CLI/train entrypoints): it closes
the one remaining reachable caller of `validate_result_claim` -- the live HTTP
gate-editor endpoint -- without changing default behavior, exactly mirroring
SLM-430's own non-breaking, additive posture.

## What changed

- `src/slm_training/web/routes.py`: `PromotionEvalRequest` gains
  `verify_locked_manifest_digest: bool = False`. `promotion_evaluate()`
  computes `locked_manifest_path = canonical_manifest_path() if
  payload.verify_locked_manifest_digest else None` (deferred import, matching
  the existing deferred-import comment inside `validate_result_claim` itself,
  to avoid a `harnesses.experiments` <-> `web.routes` import cycle) and passes
  it through to `validate_result_claim`.

In every case, omitting the flag reproduces the exact prior behavior
(`locked_manifest_path=None`, self-consistency-only check between the
manifest's and result's declared `locked_eval_manifest_sha256` strings) --
this is additive. Setting the flag makes the endpoint report
`locked_eval_manifest_digest_unverified_on_disk` inside
`checks.campaign_governance.failures` (and therefore `promotable: false`) if a
campaign's declared `locked_eval_manifest_sha256` does not match the real,
on-disk `canonical_manifest_path()` bytes -- even when the manifest and result
already agree with each other on the (possibly fabricated) string.

## Evidence (local CPU, fixture/audit scale)

`docs/design/slm431-promotion-evaluate-locked-manifest-digest-opt-in-20260727.json`
records the canonical manifest's real digest/row-count (recomputed via
`load_locked_manifest_payload`, matching the value SLM-306 and SLM-430 both
recorded, confirming the committed manifest is still untampered) and the new
test module's pass/fail table.

`tests/test_web/test_promotion_evaluate_locked_manifest_digest.py` (3 tests,
all passing) drives the real `POST /api/promotion/evaluate` route through a
`fastapi.testclient.TestClient` (the same harness `tests/test_web/
test_control_plane.py` uses) with a minimal preregistered
`ExperimentCampaignV1` manifest and a self-consistent `CampaignResultV1`:

| scenario | `verify_locked_manifest_digest` | declared digest | expected |
| --- | --- | --- | --- |
| flag omitted, fabricated digest | absent (default `False`) | fabricated (self-consistent) | no `..._unverified_on_disk` failure -- unchanged prior behavior |
| flag set, fabricated digest | `True` | fabricated (self-consistent) | `locked_eval_manifest_digest_unverified_on_disk` present, `promotable: false` |
| flag set, real digest | `True` | real committed manifest digest | no digest-related failure |

```
python -m pytest tests/test_web/test_promotion_evaluate_locked_manifest_digest.py -q
3 passed in 4.90s
```

Regression check -- existing control-plane web suite unaffected (includes the
preexisting `test_promotion_evaluate_preserves_governance_gate` /
`test_gates_evaluate_matches_pure_function` tests for the same endpoints):

```
python -m pytest tests/test_web/test_control_plane.py -q
```

See the accompanying JSON evidence file for the exact pass count recorded at
run time.

## Honesty and scope

Fixture/local-CPU wiring evidence, not a ship or promotion claim. No model
quality, promotion, or training result is claimed here. The default gate
behavior of every existing `/api/promotion/evaluate` caller (including the
live Checkpoints dashboard gate editor) is unchanged (`verify_locked_manifest_digest`
defaults to `False`); this only adds a way to opt in to the SLM-306 check from
this fourth, last-remaining reachable caller of `validate_result_claim`.
Flipping the *default* to on remains open, exactly as SLM-306 and SLM-430 both
scoped it, pending a live end-to-end promotion run to validate against --
still out of reach in this GPU-less sandbox.

## Version stamps

- `features.openfeature`: v9 -> v10 (this component's `paths` already track
  `src/slm_training/web/routes.py`; adds the new test module to `paths`).

See `src/slm_training/resources/versions.json` for the full history notes.
