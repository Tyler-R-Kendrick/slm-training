#!/usr/bin/env python3
"""Audit and exercise the SLM-184 experiment confirmation firewall.

Modes:
  describe       Print the manifest schema and broker rules.
  check          Check access for a manifest + ledger + suite.
  audit-history  Classify historical iter JSONs under docs/design.
  fixture        Deterministic fixture showing the second confirmation touch failing closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.claim_manifest import (
    ExperimentClaimManifestV1,
    SuiteAccessBroker,
    TouchLedger,
    build_default_manifest,
    classify_iter_artifact,
    freeze_manifest,
    is_frozen,
    validate_manifest,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm184-claim-manifest-20260720.json"
_DESIGN_MD = "docs/design/iter-slm184-claim-manifest-20260720.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _repo_relative(path: Path) -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _load_manifest(path: Path) -> ExperimentClaimManifestV1:
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest_data = data.get("manifest", data)
    return ExperimentClaimManifestV1.from_dict(manifest_data)


def _load_ledger(path: Path) -> TouchLedger:
    if not path.is_file():
        return TouchLedger()
    return TouchLedger.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _describe_schema() -> str:
    return """\
SLM-184 confirmation firewall schema

ExperimentClaimManifestV1 fields:
  manifest_version, experiment_family_id, source_commit, source_dirty,
  primary_hypothesis, primary_contrast, primary_endpoint, secondary_endpoints,
  mde, alpha, power, multiplicity_family,
  allowed_dev_suite_ids, confirmation_suite_id, confirmation_suite_digest,
  confirmation_touch_id, confirmation_touch_limit,
  frozen_fields, tunable_fields, selection_rule, stop_rule,
  seeds, hardware_class,
  checkpoint_pin, config_pin, codec_pin, metric_pin,
  created_at, author.

Broker rules:
  1. request_dev_access(...) always allows and appends a dev touch.
  2. request_confirmation_access(...) allows only when ALL of the following hold:
     - a frozen manifest file exists;
     - suite_id equals manifest.confirmation_suite_id;
     - suite_digest equals manifest.confirmation_suite_digest;
     - no prior confirmation touch with prediction_materialized=True exists.
  3. A second prediction-materialized confirmation touch is denied (fail-closed).

Iter classification buckets:
  clean_confirmation, development_only, reused_evaluation_data,
  provenance_incomplete, not_applicable_fixture.
"""


def _run_check(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest)
    errors = validate_manifest(manifest)
    if errors:
        print("manifest validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    ledger = _load_ledger(args.ledger)
    broker = SuiteAccessBroker(ledger)

    if args.confirmation:
        frozen_path = args.manifest.parent / "claim_manifest.frozen.json"
        decision = broker.request_confirmation_access(
            manifest,
            args.suite_id,
            args.suite_digest,
            frozen_manifest_path=frozen_path,
            prediction_materialized=True,
            reason="audit check",
        )
    else:
        decision = broker.request_dev_access(
            manifest,
            args.suite_id,
            args.suite_digest,
            prediction_materialized=False,
            reason="audit check",
        )

    print(json.dumps({"allowed": decision.allowed, "reason": decision.reason}, indent=2))
    if decision.touch_record is not None:
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        args.ledger.write_text(
            json.dumps(broker.ledger.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return 0 if decision.allowed else 1


def _run_audit_history(args: argparse.Namespace) -> int:
    iter_dir: Path = args.iter_dir
    results: list[dict[str, Any]] = []
    for path in sorted(iter_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        classification = classify_iter_artifact(data)
        results.append(
            {
                "path": str(path.relative_to(iter_dir.parent if iter_dir.name == "design" else iter_dir)),
                "classification": classification,
                "status": data.get("status") if isinstance(data, dict) else None,
                "claim_class": data.get("claim_class") if isinstance(data, dict) else None,
            }
        )

    summary: dict[str, int] = {}
    for r in results:
        summary[r["classification"]] = summary.get(r["classification"], 0) + 1

    payload = {
        "schema": "Slm184AuditHistoryV1",
        "iter_dir": str(iter_dir),
        "audited_at": _now(),
        "summary": summary,
        "results": results,
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.claim_manifest",
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# SLM-184 historical iter classification",
        "",
        f"Directory: `{iter_dir}`",
        "",
        "## Summary",
        "",
        "| classification | count |",
        "| --- | --- |",
    ]
    for classification, count in sorted(summary.items()):
        lines.append(f"| {classification} | {count} |")
    lines.extend(
        [
            "",
            "## Detail",
            "",
            "| path | classification | status | claim_class |",
            "| --- | --- | --- | --- |",
        ]
    )
    for r in results:
        lines.append(
            f"| {r['path']} | {r['classification']} | {r['status']} | {r['claim_class']} |"
        )
    lines.append("")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")

    print(str(args.output))
    return 0


def _run_fixture(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_default_manifest(
        experiment_family_id="slm184-fixture-family",
        confirmation_suite_id="rico_held",
        confirmation_suite_digest="sha256:" + "a" * 64,
    )
    errors = validate_manifest(manifest)
    if errors:
        raise RuntimeError("default manifest is invalid: " + "; ".join(errors))

    # Freeze the manifest.
    frozen_path = freeze_manifest(manifest, output_dir)
    assert is_frozen(frozen_path)

    ledger_path = output_dir / "claim_manifest_ledger.json"
    broker = SuiteAccessBroker()
    suite_id = manifest.confirmation_suite_id
    suite_digest = manifest.confirmation_suite_digest

    # First confirmation touch with prediction_materialized=True should succeed.
    first = broker.request_confirmation_access(
        manifest,
        suite_id,
        suite_digest,
        frozen_manifest_path=frozen_path,
        prediction_materialized=True,
        reason="fixture first confirmation",
    )

    # Second confirmation touch must fail closed.
    second = broker.request_confirmation_access(
        manifest,
        suite_id,
        suite_digest,
        frozen_manifest_path=frozen_path,
        prediction_materialized=True,
        reason="fixture second confirmation",
    )

    # Dev touch remains allowed.
    dev = broker.request_dev_access(
        manifest,
        "smoke",
        "sha256:" + "b" * 64,
        prediction_materialized=False,
        reason="fixture dev touch",
    )

    payload = {
        "schema": "Slm184ClaimManifestReportV1",
        "matrix_set": "slm184_claim_manifest",
        "matrix_version": "claim-manifest-v1",
        "experiment_id": "slm184-claim-manifest",
        "run_id": f"slm184-claim-manifest-{_today_yyyymmdd()}",
        "status": "fixture",
        "claim_class": "wiring",
        "hypothesis": manifest.primary_hypothesis,
        "falsifier": "The firewall allows more than one prediction-materialized confirmation touch.",
        "manifest": manifest.to_dict(),
        "frozen_manifest": _repo_relative(frozen_path),
        "first_confirmation": {"allowed": first.allowed, "reason": first.reason},
        "second_confirmation": {"allowed": second.allowed, "reason": second.reason},
        "dev_touch": {"allowed": dev.allowed, "reason": dev.reason},
        "ledger": broker.ledger.to_dict(),
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.claim_manifest",
        ),
        "timestamp": _now(),
    }

    run_json = output_dir / "slm184_claim_manifest_report.json"
    run_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    ledger_path.write_text(
        json.dumps(broker.ledger.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    if args.write_design_docs:
        root = Path(__file__).resolve().parents[1]
        json_path = args.design_json or (root / _DESIGN_JSON)
        md_path = args.design_md or (root / _DESIGN_MD)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(_render_markdown(payload), encoding="utf-8")

    print(str(run_json))

    if first.allowed and not second.allowed and dev.allowed:
        return 0
    print("fixture invariant violated", file=sys.stderr)
    return 1


def _render_markdown(payload: dict[str, Any]) -> str:
    manifest = ExperimentClaimManifestV1.from_dict(payload["manifest"])
    first = payload["first_confirmation"]
    second = payload["second_confirmation"]
    dev = payload["dev_touch"]
    lines = [
        f"# SLM-184: single-touch confirmation firewall fixture ({payload['run_id']})",
        "",
        f"Matrix set: `{payload['matrix_set']}`",
        "",
        f"Version: `{payload['matrix_version']}`",
        "",
        f"Status: **{payload['status']}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no trainable weights "
        "were updated, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        payload["hypothesis"],
        "",
        "## Falsifier",
        "",
        payload["falsifier"],
        "",
        "## Manifest",
        "",
        f"- experiment_family_id: `{manifest.experiment_family_id}`",
        f"- confirmation_suite_id: `{manifest.confirmation_suite_id}`",
        f"- confirmation_suite_digest: `{manifest.confirmation_suite_digest}`",
        f"- confirmation_touch_limit: `{manifest.confirmation_touch_limit}`",
        f"- mde: `{manifest.mde}`",
        f"- alpha: `{manifest.alpha}`",
        f"- power: `{manifest.power}`",
        "",
        "## Firewall exercise",
        "",
        "| touch | allowed | reason |",
        "| --- | --- | --- |",
        f"| first confirmation (prediction materialized) | {first['allowed']} | {first['reason']} |",
        f"| second confirmation (prediction materialized) | {second['allowed']} | {second['reason']} |",
        f"| dev touch on smoke suite | {dev['allowed']} | {dev['reason']} |",
        "",
        "## Go / no-go decision",
        "",
        "**No-go for promotion.** This is a wiring fixture. The single-touch confirmation "
        "firewall, preregistered manifest, and ledger semantics are exercised, but no real "
        "model or eval suite was used. The mechanism remains ``retain_diagnostic`` / "
        "``blocked_pending_real_eval`` until it is wired into a real matrix runner.",
        "",
        "## Honest caveats",
        "",
        "- Digest verification is string equality on a placeholder hash; real suites need a "
        "  content-addressed digest from the suite builder.",
        "- The ledger is a local JSON file; production provenance needs an append-only store.",
        "- No ship-gate claim is made; this is wiring evidence only.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m scripts.audit_experiment_firewall --mode describe",
        "python -m scripts.audit_experiment_firewall --mode fixture",
        "python -m scripts.audit_experiment_firewall --mode check \\",
        "  --manifest <path> --ledger <path> --suite-id <id> --suite-digest <digest>",
        "python -m scripts.audit_experiment_firewall --mode audit-history \\",
        "  --iter-dir docs/design --output <json> --output-md <md>",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, exit_on_error=False)
    parser.add_argument(
        "--mode",
        choices={"describe", "check", "audit-history", "fixture"},
        default="describe",
        help="Audit mode.",
    )
    parser.add_argument("--manifest", type=Path, help="Path to manifest JSON for check mode.")
    parser.add_argument("--ledger", type=Path, help="Path to ledger JSON for check mode.")
    parser.add_argument("--suite-id", type=str, help="Suite id for check mode.")
    parser.add_argument("--suite-digest", type=str, help="Suite digest for check mode.")
    parser.add_argument(
        "--confirmation",
        action="store_true",
        help="Request confirmation access in check mode (default: dev).",
    )
    parser.add_argument(
        "--iter-dir",
        type=Path,
        default=Path("docs/design"),
        help="Directory containing historical iter JSONs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/runs/slm184_audit_history/audit_history.json"),
        help="Output JSON for audit-history mode.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/runs/slm184_audit_history/audit_history.md"),
        help="Output markdown for audit-history mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/runs/slm184-claim-manifest-{_today_yyyymmdd()}"),
        help="Output directory for fixture mode.",
    )
    parser.add_argument(
        "--write-design-docs",
        action="store_true",
        help="Write design doc pair in fixture mode.",
    )
    parser.add_argument(
        "--design-json",
        type=Path,
        default=None,
        help="Override path for the design JSON (fixture mode).",
    )
    parser.add_argument(
        "--design-md",
        type=Path,
        default=None,
        help="Override path for the design markdown (fixture mode).",
    )

    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    if args.mode == "describe":
        print(_describe_schema())
        return 0

    if args.mode == "check":
        for required in ("manifest", "ledger", "suite_id", "suite_digest"):
            if getattr(args, required) is None:
                print(f"--{required.replace('_', '-')} is required for check mode", file=sys.stderr)
                return 2
        return _run_check(args)

    if args.mode == "audit-history":
        return _run_audit_history(args)

    if args.mode == "fixture":
        return _run_fixture(args)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
