"""VCE-010 (SLM-469): run the evaluator calibration / blinded adjudication
/ risk-coverage protocol.

    python -m scripts.run_vce010_evaluator_calibration_protocol --mode plan-only
    python -m scripts.run_vce010_evaluator_calibration_protocol --mode fixture \
        --out-dir outputs/runs/vce010_evaluator_calibration_protocol \
        --docs-out docs/design/vce010-evaluator-calibration-results.json

``--mode plan-only`` builds and validates the manifest (preregistering its
promotion/rollback thresholds) without running any arm or writing evidence.
``--mode fixture`` runs the real protocol
(``evaluator_calibration_protocol.run_protocol``): the deterministic-only,
external-only, human-only, and full-triple-judge arms against a frozen
semantic-contrast sample, and writes a durable, version-stamped results
JSON under ``docs/design/``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from slm_training.evals.evaluator_calibration_protocol import (
    EvaluatorCalibrationProtocolV1,
    run_protocol,
)
from slm_training.harness_core.versioning import build_version_stamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan-only", "fixture"), default="plan-only")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/runs/vce010_evaluator_calibration_protocol"),
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path("docs/design/vce010-evaluator-calibration-results.json"),
    )
    parser.add_argument("--source-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args(argv)

    protocol = EvaluatorCalibrationProtocolV1(source_count=args.source_count, seed=args.seed)

    if args.mode == "plan-only":
        manifest = protocol.manifest()
        payload = {
            "status": "plan_only",
            "claim_class": manifest.claim_class,
            "campaign_id": manifest.campaign_id,
            "arms": [arm.arm_id for arm in manifest.arms],
            "rollback_gates": [
                {"gate_id": gate.gate_id, "operator": gate.operator, "threshold": gate.threshold}
                for gate in manifest.rollback_gates
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    result = run_protocol(protocol, root=args.out_dir)
    payload = {
        "kind": "vce010_evaluator_calibration_protocol/v1",
        "claim_class": result["claim_class"],
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "recipe": {
            "command": [
                "python",
                "-m",
                "scripts.run_vce010_evaluator_calibration_protocol",
                "--mode",
                "fixture",
                "--source-count",
                str(args.source_count),
                "--seed",
                str(args.seed),
            ],
            "docs_path": str(args.docs_out),
        },
        "disposition": result["disposition"],
        "pair_n": result["pair_n"],
        "arms": result["arms"],
        "external_human_calibration_error": result.get("external_human_calibration_error"),
        "external_human_cohen_kappa": result.get("external_human_cohen_kappa"),
        "admission_divergence_rate": result.get("admission_divergence_rate"),
        "preregistered_calibration_error_max": result.get("preregistered_calibration_error_max"),
        "rollback_gate_fired": result.get("rollback_gate_fired"),
        "independence_ok": result.get("independence_ok"),
        "ambiguous_pair_ids": result.get("ambiguous_pair_ids"),
        "scope_disclaimer": result.get("scope_disclaimer"),
        "manifest_sha256": result["manifest_sha256"],
        "version_stamp": build_version_stamp("evals.evaluator_calibration_protocol"),
    }
    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.docs_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
