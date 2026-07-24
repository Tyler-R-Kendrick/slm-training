# SLM-262 (VSD0-03): durable GPU train→checkpoint→eval reference run

**Status:** `environment_incompatible` — harness implemented and verified; real GPU
execution deferred to an environment with GPU quota/credits.

**Manifest:** [`iter-slm262-gpu-reference-run-20260721.json`](iter-slm262-gpu-reference-run-20260721.json)

This document records the provider-neutral reference-run harness introduced for
SLM-262. The harness persists an immutable `AcceleratorRunManifestV1` through
plan, dry-run, CPU compatibility smoke, submit, reconcile, and evaluate phases.
It does **not** by itself spend GPU budget; it composes the existing canonical
entry points (`scripts.hf_jobs_train`, `scripts.remote_train`,
`scripts.train_model`, `scripts.evaluate_model`).

## Manifest schema

`AcceleratorRunManifestV1` (`slm262_gpu_reference.py`) is a strict,
JSON-serializable dataclass:

- `schema_version`: `accelerator_run_manifest/v1`
- `run_id`, `track`, `source_commit`, `repo_url`
- `provider`: one of `hf_jobs`, `remote_pod`, `dry_run`
- `instance_type`, `gpu_model`, `gpu_count`, `gpu_memory_gb`
- `data_snapshot_id` / `data_snapshot_sha`, `eval_snapshot_id` / `eval_snapshot_sha`
- `target_decisions`, `max_wall_minutes`, `checkpoint_cadence_decisions`
- `expected_artifacts`, `remote_uri_prefix`, `provider_options`
- Runtime fields: `provider_request_id`, `provider_job_id`, `timestamps`,
  `utilization`, `checkpoint_inventory`, `full_state_inventory`,
  `evaluation_report_refs`, `disposition`, `notes`, `version_stamp`

`check_ready()` fails closed: any missing provenance (`source_commit`, snapshot
SHAs, artifacts, remote prefix) returns a blocker. It rejects dirty trees unless
`dirty_tree_ok=true` and verifies that `source_commit` matches `HEAD`.

## CLI workflow

```bash
python -m scripts.run_gpu_reference init \
  --run-id slm262_cpu_smoke \
  --provider dry_run \
  --data-snapshot-id e530_visible_semantic_roles_r1_20260719 \
  --data-snapshot-sha 3ce2fcd915456e6ec0f456dafd858eeb8e133427704eff0f3615535e84dec852 \
  --eval-snapshot-id remediated \
  --eval-snapshot-sha da16626fcbd271f1622f21271235c9315ff8f82ce85e1502380dd6810025c41b \
  --target-decisions 50000 \
  --output docs/design/iter-slm262-gpu-reference-run-20260721.json

python -m scripts.run_gpu_reference validate --manifest docs/design/iter-slm262-gpu-reference-run-20260721.json
python -m scripts.run_gpu_reference describe --manifest docs/design/iter-slm262-gpu-reference-run-20260721.json
python -m scripts.run_gpu_reference submit --manifest docs/design/iter-slm262-gpu-reference-run-20260721.json \
  --provider hf_jobs --dry-run --output docs/design/iter-slm262-gpu-reference-run-20260721.json
python -m scripts.run_gpu_reference local-smoke --manifest docs/design/iter-slm262-gpu-reference-run-20260721.json \
  --steps 2 --resume-steps 1 --output docs/design/iter-slm262-gpu-reference-run-20260721.json
```

## Dry-run plan

`submit --provider hf_jobs --dry-run` produced a complete HF Jobs command plan
(flavor `a10g-large`, timeout `3m`, bucket mount `hf://buckets/TKendrick/OpenUI`,
image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`). The plan clones the
repo, installs deps, builds train/test data, and runs `scripts.train_model`
with `--steps 200 --sync-checkpoints`. It validates that the provider adapter
and CLI correctly compose `scripts.hf_jobs_train` without spending compute.

## Local CPU compatibility smoke

`local-smoke` ran a short CPU train→resume loop:

- Initial train: 2 steps on `cpu` with `--context-backend scratch --fast-train --no-sync-checkpoints`
- Resume train: from `last_full_state.pt` for 3 total steps
- Produced artifacts under `outputs/runs/slm262_cpu_smoke_cpu_smoke/checkpoints/`

Artifact hashes recorded in the manifest:

| artifact | size | sha256 |
| --- | --- | --- |
| `last.pt` | 6,592,924 | `ec1fff0540e4cc07227013b7111d6f602b610ffc977ffaa8a4656f45606948e5` |
| `last_full_state.pt` | 19,815,800 | `9cccf95a220eabf78aa1042d24adfbe78dd029e5a5490555a439ca880cbbb72f` |
| `last.tokenizer.json` | 13,794 | `c17ab2a5ab44dfd54d60a04fb995bda6720df65ff42f362d483bd2308181c927` |
| `last.meta.json` | 9,256 | `7531365b72b352676a331e2e904b96ca8dd6e436850cb1bcace359e5f0c2956c` |

The smoke proves that the exact `run_id`/config can execute, produce a
full-state checkpoint, and resume from it. It is Phase-A no-spend evidence only:
it does not touch a GPU and does not claim durable persistence.

## Disposition and blockers

Final disposition: `environment_incompatible`.

- Local CPU smoke: **passed**.
- HF Jobs dry-run plan: **generated successfully**.
- Real GPU execution: **blocked** because this agent session has no local GPU
  and no confirmed paid/quotated HF Jobs budget. The code path is ready; running
  it requires an environment with GPU quota/credits (e.g. `hf jobs run` against
  a billed organization or a `scripts.remote_train` pod).

## Version stamp

```json
{
  "stamp_schema": "version_stamp/v1",
  "code_commit": "861b7c93c3ef56c44bbe04705a95247341b26fc2",
  "code_dirty": true,
  "components": {
    "harness.experiments": "v60",
    "harness.experiments.slm262_gpu_reference_run": "v1"
  }
}
```

## Files changed

- `src/slm_training/harnesses/experiments/slm262_gpu_reference.py`
- `src/slm_training/harnesses/experiments/__init__.py`
- `scripts/run_gpu_reference.py`
- `src/slm_training/resources/versions.json`
- `tests/test_harnesses/experiments/test_slm262_gpu_reference.py`
- `tests/test_scripts/test_run_gpu_reference.py`
- `docs/design/gpu-reference-run.md`
- `docs/design/iter-slm262-gpu-reference-run-20260721.json`
