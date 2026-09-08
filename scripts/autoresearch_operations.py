"""Canonical autoresearch operations subcommands; no alternative supervisor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.autoresearch.runtime.operations_analysis import compare_sign_request
from slm_training.autoresearch.runtime.operations_control import (
    load_start_config,
    service_template,
    start_supervisor,
    stop_supervisor,
)
from slm_training.autoresearch.runtime.operations_doctor import doctor, storage_health
from slm_training.harness_core.execution_release import prepare_release
from slm_training.autoresearch.runtime.operations_status import loop_status


def cmd_doctor(args) -> int:
    result = doctor(
        args.source,
        args.root,
        recovery_config=args.recovery_config,
        require_lean=args.require_lean,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["local_ready"] else 2


def cmd_start(args) -> int:
    config = load_start_config(args.config)
    result = start_supervisor(args.root, config)
    print(json.dumps(result, indent=2))
    return 0 if result.get("started") or result.get("already_owned") else 2


def cmd_stop(args) -> int:
    result = stop_supervisor(args.root, args.loop_id)
    print(json.dumps(result, indent=2))
    return (
        0
        if result.get("stopped") or result.get("reason") == "no_owned_controller"
        else 2
    )


def cmd_verify(args) -> int:
    from scripts.verify_autonomy import main

    return main(["--root", str(args.root), "--batch-size", str(args.batch_size)])


def cmd_prepare(args) -> int:
    result = prepare_release(args.source, args.release, args.execution, args.outputs)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "files"}, indent=2
        )
    )
    return 0


def cmd_service(args) -> int:
    print(service_template(args.config, args.root))
    return 0


def cmd_storage(args) -> int:
    result = storage_health(args.root, minimum_free_bytes=args.minimum_free_bytes)
    print(json.dumps(result, indent=2))
    return 0 if result["ready"] else 2


def cmd_compare(args) -> int:
    print(
        json.dumps(compare_sign_request(args.request, args.root, args.docs), indent=2)
    )
    return 0


def render_status(args) -> int:
    from slm_training.autoresearch.storage import (
        CampaignStore,
        render_loop_result_matrix,
    )

    if args.loop_id:
        if args.matrix:
            print(
                render_loop_result_matrix(
                    args.root, args.loop_id, last=None if args.all else args.last
                )
            )
        else:
            print(json.dumps(loop_status(args.root, args.loop_id), indent=2))
    else:
        if args.matrix or args.all or args.last != 5:
            raise ValueError("matrix/history options require --loop-id")
        print(json.dumps(CampaignStore(args.campaign_id, args.root).status(), indent=2))
    return 0


def add_campaign_operations_parser(sub, status_handler, ack_handler, sync_handler):
    """Keep campaign operational flags together without importing the entrypoint."""
    status = sub.add_parser("status")
    target = status.add_mutually_exclusive_group(required=True)
    target.add_argument("--campaign-id")
    target.add_argument("--loop-id")
    status.add_argument("--matrix", action="store_true")
    history = status.add_mutually_exclusive_group()
    history.add_argument("--last", type=int, default=5)
    history.add_argument("--all", action="store_true")
    status.set_defaults(func=status_handler)
    acknowledge = sub.add_parser("ack-action")
    acknowledge.add_argument("--loop-id", required=True)
    acknowledge.add_argument("--campaign-id", required=True)
    acknowledge.add_argument("--action-index", type=int, required=True)
    acknowledge.add_argument(
        "--status", choices=("completed", "blocked"), default="completed"
    )
    acknowledge.add_argument("--evidence", action="append", required=True)
    acknowledge.set_defaults(func=ack_handler)
    sync = sub.add_parser("sync")
    sync.add_argument("--campaign-id", required=True)
    sync.add_argument("--push", action="store_true")
    sync.set_defaults(func=sync_handler)


def add_operations_parser(
    parser: argparse.ArgumentParser, sub
) -> argparse.ArgumentParser:
    from scripts.merge_verification_controller import add_release_parser

    add_release_parser(sub)
    readiness = sub.add_parser(
        "doctor", help="Probe real local capabilities without provider spending"
    )
    readiness.add_argument("--source", type=Path, default=Path.cwd())
    readiness.add_argument("--recovery-config", type=Path)
    readiness.add_argument("--require-lean", action="store_true")
    readiness.set_defaults(func=cmd_doctor)
    for name in ("start", "resume"):
        start = sub.add_parser(
            name, help="Explicitly start the canonical user-owned supervisor"
        )
        start.add_argument("--config", type=Path, required=True)
        start.set_defaults(func=cmd_start)
    stop = sub.add_parser(
        "stop", help="Stop only the identity-verified owned controller/children"
    )
    stop.add_argument("--loop-id", required=True)
    stop.set_defaults(func=cmd_stop)
    verify = sub.add_parser(
        "verify-autonomy",
        help="Finite resumable operational fault fixtures, not model proof",
    )
    verify.add_argument("--batch-size", type=int, choices=range(1, 21), default=20)
    verify.set_defaults(func=cmd_verify)
    prepare = sub.add_parser(
        "prepare-release",
        help="Materialize pinned source and separate local execution/output",
    )
    for name in ("source", "release", "execution", "outputs"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.set_defaults(func=cmd_prepare)
    service = sub.add_parser(
        "service-template",
        help="Print an optional lifecycle recipe; never install/activate",
    )
    service.add_argument("--config", type=Path, required=True)
    service.set_defaults(func=cmd_service)
    storage = sub.add_parser(
        "storage-health", help="Pressure/backpressure assessment; never delete evidence"
    )
    storage.add_argument("--minimum-free-bytes", type=int, default=256 * 1024 * 1024)
    storage.set_defaults(func=cmd_storage)
    comparison = sub.add_parser(
        "compare-signs",
        help="Default-off fixed versus sequential probability-sign analysis",
    )
    comparison.add_argument("--request", type=Path, required=True)
    comparison.add_argument("--docs", type=Path, default=Path("docs/design"))
    comparison.set_defaults(func=cmd_compare)
    return parser
