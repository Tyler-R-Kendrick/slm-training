"""CLI: lock ExperimentCampaignV1 and run reasoning/revmath tasks (HARN-10 / SLM-548).

Owners/docs: docs/design/reverse-mathematics-computability.md +
.agents/skills/revmath. Fixture/wiring claim_class is not ship authority.
External Lean/provers remain optional experiments (default --hermetic).

Materialize example:
    PYTHONPATH=src uv run python -m scripts.run_revmath_profile \\
      --task src/slm_training/resources/revmath/fixtures/ablation_necessary.task.json \\
      --materialize-fixture \\
      --campaign-id camp.rm.hermetic \\
      --experiment-id exp.rm.hermetic.profile \\
      --store-root outputs/revmath/profile_runs \\
      --hermetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.reasoning.revmath.profile_binding import (
    RevmathCampaignBindingError,
    RevmathProfilePlanV1,
    load_experiment_campaign,
    replay_profile_run_from_store,
    run_revmath_profile,
)
from slm_training.harnesses.reasoning.revmath.schemas import parse_revmath_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        type=Path,
        action="append",
        required=True,
        help="Frozen RevmathTaskV1 JSON (repeatable)",
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        required=True,
        help="CampaignStore root (canonical event/artifact lineage)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Existing ExperimentCampaignV1 JSON to lock (strict binding)",
    )
    parser.add_argument(
        "--materialize-fixture",
        action="store_true",
        help="Materialize a fixture/wiring ExperimentCampaignV1 from tasks",
    )
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--source-commit", default="0" * 40)
    parser.add_argument(
        "--claim-class",
        default="fixture",
        choices=("fixture", "wiring", "diagnostic"),
    )
    parser.add_argument(
        "--hermetic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lean-root", type=Path, default=None)
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Replay the latest completed profile run from --store-root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the profile-run summary JSON",
    )
    args = parser.parse_args(argv)

    tasks = [
        parse_revmath_task(json.loads(path.read_text(encoding="utf-8")))
        for path in args.task
    ]

    try:
        if args.replay_only:
            if not args.campaign_id or not args.experiment_id:
                raise SystemExit("--replay-only requires --campaign-id and --experiment-id")
            payload = replay_profile_run_from_store(
                args.store_root,
                args.campaign_id,
                experiment_id=args.experiment_id,
            )
            text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text, encoding="utf-8")
            print(text, end="")
            return 0

        if args.materialize_fixture == bool(args.manifest):
            raise SystemExit("provide exactly one of --manifest or --materialize-fixture")

        manifest = None
        plan = None
        if args.manifest is not None:
            manifest = load_experiment_campaign(args.manifest)
        else:
            campaign_id = args.campaign_id or tasks[0].campaign.campaign_id
            experiment_id = args.experiment_id or tasks[0].campaign.experiment_id
            plan = RevmathProfilePlanV1(
                campaign_id=campaign_id,
                experiment_id=experiment_id,
                source_commit=args.source_commit,
                claim_class=args.claim_class,
            )

        profile = run_revmath_profile(
            tasks,
            store_root=args.store_root,
            manifest=manifest,
            plan=plan,
            hermetic=args.hermetic,
            lean_root=args.lean_root,
        )
    except RevmathCampaignBindingError as exc:
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        return 2

    summary = {
        "profile_id": profile.profile_id,
        "campaign_id": profile.campaign_id,
        "experiment_id": profile.experiment_id,
        "manifest_sha256": profile.manifest_sha256,
        "claim_class": profile.claim_class,
        "task_count": len(profile.runs),
        "judgments": [run.result.judgment.get("outcome") for run in profile.runs],
        "disposition_hash": profile.disposition.get("record_hash"),
        "store_root": profile.store_root,
        "formal_preflight_sha256s": list(profile.formal_preflight_sha256s),
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
