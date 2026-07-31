#!/usr/bin/env python3
"""Export and multi-prover-check formal objects (close the formal loop).

Does not rely on a single Lean kernel. Default backends are pure Python
(structural + reference). Optional ``--lean`` adds the Lean package build as a
third independent backend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from slm_training.formal.loop import close_formal_loop, write_loop_report
from slm_training.formal.objects import (
    export_closure_law,
    export_lean_claim,
    export_support_certificate,
    lean_claim_catalog,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_objects(args: argparse.Namespace) -> list[Any]:
    objects: list[Any] = []

    if args.catalog_lean:
        for theorem_id in lean_claim_catalog():
            objects.append(export_lean_claim(theorem_id, proof_status="proved"))

    if args.closure_laws:
        for law_id in (
            "removable_only_replay_valid_unsupported",
            "close_pass_only_shrinks",
            "fixed_point_when_no_removals",
        ):
            objects.append(export_closure_law(law_id))

    for path in args.certificate or []:
        data = _load_json(path)
        # Accept raw SupportCertificate JSON or wrapped formal object.
        if data.get("schema") == "formal_object/v1":
            from slm_training.formal.objects import FormalObjectV1

            objects.append(FormalObjectV1.from_dict(data))
        else:
            from slm_training.dsl.solver.support import SupportCertificate

            cert = SupportCertificate.from_dict(data)
            objects.append(export_support_certificate(cert))

    for path in args.formal_object or []:
        from slm_training.formal.objects import FormalObjectV1

        objects.append(FormalObjectV1.from_dict(_load_json(path)))

    if args.export_dir:
        out = Path(args.export_dir)
        out.mkdir(parents=True, exist_ok=True)
        for obj in objects:
            target = out / f"{obj.object_id.replace(':', '_').replace('/', '_')}.json"
            target.write_text(
                json.dumps(obj.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    return objects


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-lean",
        action="store_true",
        help="Export and check the full Lean claim catalog",
    )
    parser.add_argument(
        "--closure-laws",
        action="store_true",
        help="Export and check VSS exact-closure honesty laws",
    )
    parser.add_argument(
        "--certificate",
        type=Path,
        action="append",
        default=[],
        help="Path to a SupportCertificate JSON (repeatable)",
    )
    parser.add_argument(
        "--formal-object",
        type=Path,
        action="append",
        default=[],
        help="Path to an existing formal_object/v1 JSON (repeatable)",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Write exported formal objects as JSON files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/design/formal-loop-report.json"),
        help="Multi-prover loop report JSON",
    )
    parser.add_argument(
        "--min-backends",
        type=int,
        default=2,
        help="Minimum independent backends required (default 2)",
    )
    parser.add_argument(
        "--lean",
        action="store_true",
        help="Also run the optional Lean kernel backend",
    )
    parser.add_argument(
        "--allow-open",
        action="store_true",
        help="Exit 0 even if the formal loop is open (report only)",
    )
    args = parser.parse_args(argv)

    if not (
        args.catalog_lean
        or args.closure_laws
        or args.certificate
        or args.formal_object
    ):
        # Default production-core formal loop surface.
        args.catalog_lean = True
        args.closure_laws = True

    objects = _load_objects(args)
    if not objects:
        print("no formal objects to check", file=sys.stderr)
        return 2

    report = close_formal_loop(
        objects,
        min_backends=args.min_backends,
        enable_lean=args.lean,
    )
    write_loop_report(args.out, report)
    print(report.summary)
    print(f"wrote {args.out}")
    if report.single_kernel_reliance:
        print("warning: single-kernel reliance detected", file=sys.stderr)
    if not report.closed and not args.allow_open:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
