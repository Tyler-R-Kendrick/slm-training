#!/usr/bin/env python3
"""Run a locked SLM-266 teacher manifest through an explicit provider endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.teacher_programs import (
    LocalTransformersTeacherTransport,
    OpenAICompatibleTeacherTransport,
    TeacherProgramExecutionRequestV1,
    TeacherProgramExecutor,
    TeacherProgramGenerationManifestV1,
    verify_teacher_generation_campaign_lock,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--archive-root", type=Path, default=Path("outputs/autoresearch"))
    parser.add_argument("--endpoint")
    parser.add_argument("--api-key-env")
    parser.add_argument("--system-prompt", required=True)
    parser.add_argument(
        "--local-transformers",
        action="store_true",
        help="Generate locally from the manifest's pinned model/revision; never downloads or calls a provider.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--campaign-manifest",
        type=Path,
        help="Preregistered ExperimentCampaignV1 JSON; required with --execute.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually call the configured provider")
    args = parser.parse_args(argv)
    manifest = TeacherProgramGenerationManifestV1.from_dict(
        json.loads(args.manifest.read_text(encoding="utf-8"))
    )
    requests = tuple(
        TeacherProgramExecutionRequestV1.from_dict(row)
        for row in json.loads(args.requests.read_text(encoding="utf-8"))
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "describe_only",
                    "manifest_hash": manifest.manifest_hash,
                    "request_ids": [request.request_id for request in requests],
                    "max_dollars": manifest.max_dollars,
                    "max_input_tokens": manifest.max_input_tokens,
                    "max_output_tokens": manifest.max_output_tokens,
                    "provider_called": False,
                }
            )
        )
        return 0
    if args.campaign_manifest is None:
        parser.error("--campaign-manifest is required with --execute")
    campaign = ExperimentCampaignV1.model_validate_json(
        args.campaign_manifest.read_text(encoding="utf-8")
    )
    if campaign.campaign_id != args.campaign_id:
        parser.error("campaign manifest campaign_id must match --campaign-id")
    if campaign.experiment_id != manifest.manifest_id:
        parser.error("campaign manifest experiment_id must match generation manifest_id")
    archive = CampaignStore(args.campaign_id, args.archive_root)
    archive.lock_experiment_campaign(campaign)
    campaign_manifest_sha256 = verify_teacher_generation_campaign_lock(manifest, archive)
    archive.append_event(
        "experiment_started",
        experiment_id=manifest.manifest_id,
        status="running",
        detail={"campaign_manifest_sha256": campaign_manifest_sha256},
    )
    executor = TeacherProgramExecutor(manifest, archive)
    if args.local_transformers:
        transport = LocalTransformersTeacherTransport(
            model=manifest.model,
            revision=manifest.revision,
            system_prompt=args.system_prompt,
            device=args.device,
            dtype=args.dtype,
        )
    else:
        if not args.endpoint or not args.api_key_env:
            parser.error("--endpoint and --api-key-env are required without --local-transformers")
        transport = OpenAICompatibleTeacherTransport(
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
            model=manifest.model,
            system_prompt=args.system_prompt,
        )
    try:
        result = executor.execute(requests, transport)
    except BaseException:
        archive.append_event(
            "experiment_finished",
            experiment_id=manifest.manifest_id,
            status="interrupted",
            detail={"campaign_manifest_sha256": campaign_manifest_sha256},
        )
        raise
    archive.append_event(
        "experiment_finished",
        experiment_id=manifest.manifest_id,
        status="completed",
        detail={"campaign_manifest_sha256": campaign_manifest_sha256},
    )
    print(
        json.dumps(
            {
                "status": "executed",
                "completed_request_ids": result.completed_request_ids,
                "skipped_request_ids": result.skipped_request_ids,
                "attempt_artifact_sha256s": result.attempt_artifact_sha256s,
                "spent_input_tokens": result.spent_input_tokens,
                "spent_output_tokens": result.spent_output_tokens,
                "spent_dollars": result.spent_dollars,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
