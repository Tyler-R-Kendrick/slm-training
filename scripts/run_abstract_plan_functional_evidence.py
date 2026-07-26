#!/usr/bin/env python3
"""Write SLM-313's fail-closed locked-evidence preflight."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.autoresearch.schemas import CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.abstract_plan_functional_evidence import (
    build_campaign,
    load_locked_protocol,
)
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = "harness.experiments.slm313_abstract_plan_functional_evidence"
DEFAULT_ROOT = Path("outputs/autoresearch/slm313-abstract-plan-functional-evidence")
DEFAULT_JSON = ROOT / "docs/design/iter-slm313-abstract-plan-functional-evidence-20260726.json"
DEFAULT_MARKDOWN = ROOT / "docs/design/abstract-plan-functional-evidence.md"
DEFAULT_AGENTV = ROOT / "docs/design/iter-slm313-abstract-plan-functional-evidence-agentv-20260726"


def _portable(value: Any, root: Path) -> Any:
    prefix = str(root.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "agentv-dir://" + value[len(prefix):].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, root) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(root: Path) -> None:
    replacements = {
        str(root.resolve()): "agentv-dir://",
        quote(str(root.resolve()), safe=""): quote("agentv-dir://", safe=""),
        str(ROOT.resolve()): "repo://",
        quote(str(ROOT.resolve()), safe=""): quote("repo://", safe=""),
    }
    for path in (root / "agentv").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}:
            text = path.read_text(encoding="utf-8")
            for source, replacement in replacements.items():
                text = text.replace(source, replacement)
            path.write_text(text, encoding="utf-8")


def run_preflight(*, root: Path, checkpoint: Path | None, agentv_dir: Path) -> dict[str, Any]:
    protocol = load_locked_protocol(
        ROOT / "src/slm_training/resources/data/eval/manifests/abstract_planning_locked_v1.jsonl"
    )
    campaign = build_campaign(protocol)
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="Measure locked AbstractPlan function without proxy controls.",
            primary_metric="binding_aware_meaningful_v2",
            track="twotower",
            budget=campaign.budget,
            created_at=campaign.created_at,
        )
    )
    lock = store.lock_experiment_campaign(campaign)
    available = checkpoint is not None and checkpoint.is_file()
    stamp = build_version_stamp("harness.experiments", COMPONENT)
    result = {
        "verdict": "execution_not_implemented" if available else "unavailable",
        "reason": (
            "A checkpoint was supplied, but the shared-model execution adapter is not implemented."
            if available
            else "No trained learned-plan checkpoint was supplied; AP-023's side-channel head and AP-024's zero-gate connector are not evidence of learned plan function."
        ),
        "promotion_eligible": False,
        "meaningful_parse": "not_measured",
    }
    report: dict[str, Any] = {
        "schema": "slm313_abstract_plan_functional_evidence/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_class": "diagnostic",
        "protocol": protocol.to_dict(),
        "campaign_manifest_sha256": lock.manifest_sha256,
        "recipe": {
            "device": "not_run",
            "backend": "not_run",
            "locked_test_used_for_selection": False,
            "human_rating_gate": "not_required",
            "checkpoint": str(checkpoint) if checkpoint else None,
        },
        "result": result,
        "version_stamp": stamp,
    }
    report["agentv"] = _portable(
        publish_agentv_evaluation(
            agentv_dir,
            name="slm313-abstract-plan-functional-evidence",
            claim="abstract_plan_preflight_not_ship",
            version_stamp=stamp,
            cases=[{
                "id": "unavailable-does-not-promote",
                "criteria": "No proxy plan result may be promoted.",
                "assertions": [{
                    "id": "no_proxy_promotion",
                    "actual": result["promotion_eligible"] is False,
                    "operator": "eq",
                    "expected": True,
                }],
                "result": result,
            }],
        ),
        agentv_dir,
    )
    _rewrite_agentv_paths(agentv_dir)
    return report


def _markdown(report: dict[str, Any]) -> str:
    result = report["result"]
    return (
        "# SLM-313 AbstractPlan functional evidence\n\n"
        "This locked functional-evidence preflight makes no model or promotion claim.\n\n"
        f"- Verdict: {result['verdict']}; promotion eligible: false.\n"
        f"- Reason: {result['reason']}\n"
        f"- Locked manifest: {report['protocol']['locked_eval_manifest_sha256']}.\n"
        "- Meaningful-parse and meaning-v2 were not measured.\n"
        "- AgentEvals/AgentV records the fail-closed non-promotion assertion.\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--agentv-dir", type=Path, default=DEFAULT_AGENTV)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"claim": "no execution", "root": str(args.root)}, sort_keys=True))
        return 0
    report = run_preflight(root=args.root, checkpoint=args.checkpoint, agentv_dir=args.agentv_dir)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.json_out), "verdict": report["result"]["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
