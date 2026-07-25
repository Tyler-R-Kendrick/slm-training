"""Dry-run or promote a staged capability gate result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harness_core.checkpoint_reference import CheckpointReferenceV1
from slm_training.harnesses.capability_gates import (
    CapabilityCertificateV1,
    CapabilityGateResultV1,
    CapabilityGateSpecV1,
    PromotionAuthority,
    issue_certificate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dry-run", "promote"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--checkpoint-reference", type=Path, required=True)
    parser.add_argument("--prior", type=Path, action="append", default=[])
    parser.add_argument("--authority", choices=("human", "ci"))
    parser.add_argument("--allow-distillation", action="store_true")
    parser.add_argument("--confirm-human", action="store_true")
    parser.add_argument("--ci-attested", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.command == "promote":
        if args.authority is None or args.output is None:
            parser.error("promote requires --authority and --output")
        if args.authority == "human" and not args.confirm_human:
            parser.error("human promotion requires --confirm-human")
        if args.authority == "ci" and not args.ci_attested:
            parser.error("CI promotion requires --ci-attested")
    authority = PromotionAuthority(args.authority or "human")
    certificate = issue_certificate(
        CapabilityGateSpecV1.load(args.spec),
        CapabilityGateResultV1.load(args.result),
        CheckpointReferenceV1.load_json(args.checkpoint_reference),
        priors=tuple(CapabilityCertificateV1.load(path) for path in args.prior),
        authority=authority,
        distillation_allowed=bool(args.allow_distillation),
    )
    if args.command == "promote":
        certificate.write(args.output)
    print(
        json.dumps(
            {
                "valid": True,
                "dry_run": args.command == "dry-run",
                "certificate_id": certificate.certificate_id,
                "output": None if args.output is None else args.output.as_posix(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
