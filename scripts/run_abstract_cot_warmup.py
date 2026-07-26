#!/usr/bin/env python3
"""Run SLM-311's bounded, local Abstract-CoT warm-up fixture.

The fixture executes the existing AP-020 collector and builds the AP-019
segment-masked payload.  It never instantiates a model, evaluates a locked
suite, or claims a checkpoint/promotion/ship result.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.evals.verifier_cascade import (
    Verdict,
    VerifierCascade,
    VerifierResultV1,
    VerifierStage,
    VerifierStageSpec,
)
from slm_training.harnesses.distill.self_distill_collect import (
    SelfDistillCollectConfig,
    collect_on_policy_traces,
    training_example_from_capture,
)
from slm_training.harnesses.distill.trace_store import TraceStore
from slm_training.harnesses.reasoning.abstract_cot_warmup import (
    AbstractWarmupCampaignV1,
    IterationRunner,
    WarmupResult,
    run_abstract_cot_warmup,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.models.abstract_decode import (
    AbstractPhaseRecord,
    AbstractTraceCapture,
    AbstractTracedGeneration,
    DecodePhase,
)
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = "harness.experiments.slm311_abstract_cot_warmup"
DEFAULT_JSON = ROOT / "docs/design/iter-slm311-abstract-cot-warmup-20260726.json"
DEFAULT_MARKDOWN = ROOT / "docs/design/iter-slm311-abstract-cot-warmup-20260726.md"
DEFAULT_AGENTV = ROOT / "docs/design/iter-slm311-abstract-cot-warmup-agentv-20260726"


def _source_commit() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else "0" * 40


def _fixture_capture(iteration: int) -> AbstractTraceCapture:
    return AbstractTraceCapture(
        plan_fingerprint=f"fixture-plan-{iteration}",
        prompt=AbstractPhaseRecord(DecodePhase.PROMPT, (1, 2), (None, None)),
        abstract=AbstractPhaseRecord(DecodePhase.ABSTRACT, (100, 101), (None, -0.1)),
        answer=AbstractPhaseRecord(DecodePhase.ANSWER, (9, 10), (-0.2, -0.2)),
        abstract_termination="sampled_end",
        forced_end=False,
        stop_reason="eos",
    )


def _fixture_verifier() -> VerifierCascade:
    def evaluate(source: str, _context: object) -> VerifierResultV1:
        passed = source.startswith("root = ")
        return VerifierResultV1("fixture", Verdict.PASS if passed else Verdict.FAIL, sound=not passed)

    spec = VerifierStageSpec("fixture", "1", "hermetic fixture verifier", sound_fail=True, cache_policy="none")
    return VerifierCascade([VerifierStage(spec, evaluate)])


def fixture_iteration_runner(
    campaign: AbstractWarmupCampaignV1, *, command: str
) -> IterationRunner:
    """Bind AP-020/AP-019 existing primitives for a hermetic T-iteration proof."""
    def run(iteration: int, parent: str | None, out: Path) -> dict[str, Any]:
        out.mkdir(parents=True, exist_ok=False)
        capture = _fixture_capture(iteration)
        prompt = f"fixture warmup prompt {iteration}"
        answer = "root = Separator()"
        traces = TraceStore(out / "traces")
        collection = collect_on_policy_traces(
            generate=lambda _prompt: AbstractTracedGeneration(answer, capture, valid=True),
            policy_checkpoint_sha=parent or "fixture-base-policy",
            prompts=(prompt,),
            store=traces,
            decode_config_hash=campaign.fingerprint,
            config=SelfDistillCollectConfig(max_records=1),
            verifier=_fixture_verifier(),
        )
        payload = training_example_from_capture(capture)
        policy_path = out / "policy.json"
        policy = f"fixture://{campaign.campaign_id}/iteration/{iteration}"
        policy_payload = {"policy": policy, "parent_policy": parent, "kind": "fixture_policy_artifact"}
        policy_path.write_text(json.dumps(policy_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        collection_path = out / "collection.json"
        collection_payload = collection.to_dict()
        collection_path.write_text(json.dumps(collection_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sft_path = out / "ap019_payload.json"
        sft_payload = {
            "input_shape": list(payload["input_ids"].shape),
            "segment_shape": list(payload["segment_ids"].shape),
            "token_count": int(payload["input_ids"].numel()),
            "primitive": "training_example_from_capture",
        }
        sft_path.write_text(json.dumps(sft_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        phases = {
            "ap020_collection": {
                "command": command,
                "config_sha256": campaign.fingerprint,
                "input_sha256": content_sha({"prompt": prompt, "parent_policy": parent}),
                "output_sha256": content_sha(collection_payload),
                "token_count": collection.accepted * capture.abstract.token_count,
                "result_ref": str(collection_path),
            },
            "ap019_segment_masked_sft": {
                "command": command,
                "config_sha256": campaign.fingerprint,
                "input_sha256": content_sha(capture.to_dict()),
                "output_sha256": content_sha(sft_payload),
                "token_count": sft_payload["token_count"],
                "result_ref": str(sft_path),
            },
        }
        return {
            "policy": policy,
            "command": command,
            "token_count": sft_payload["token_count"],
            "update_count": 0,
            "result_ref": str(policy_path),
            "phases": phases,
        }
    return run


def _portable(value: Any, output_dir: Path) -> Any:
    prefix = str(output_dir.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "agentv-dir://" + value[len(prefix):].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    replacements = {str(output_dir.resolve()): "agentv-dir://", quote(str(output_dir.resolve()), safe=""): quote("agentv-dir://", safe=""), str(ROOT.resolve()): "repo://", quote(str(ROOT.resolve()), safe=""): quote("repo://", safe="")}
    for path in (output_dir / "agentv").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}:
            text = path.read_text(encoding="utf-8")
            for source, replacement in replacements.items():
                text = text.replace(source, replacement)
            path.write_text(text, encoding="utf-8")


def _evidence(result: WarmupResult, *, stamp: dict[str, Any], agentv_dir: Path) -> dict[str, Any]:
    policies = [row.policy for row in result.iterations]
    complete = len(policies) == 3 and len(set(policies)) == 3 and all(set(row.phases) == {"ap020_collection", "ap019_segment_masked_sft"} for row in result.iterations)
    report: dict[str, Any] = {
        "schema": "slm311_abstract_cot_warmup/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recipe": {"device": "none", "backend": "local hermetic fixture", "iterations": len(result.iterations), "locked_test_used_for_selection": False, "model_checkpoint_written": False},
        "result": result.to_dict(),
        "meaningful_parse": {"status": "not_measured", "reason": "fixture wiring only; no model output evaluation"},
        "ship": {"eligible": False, "reason": "fixture collector/payload wiring is not a trained or evaluated model"},
        "version_stamp": stamp,
    }
    report["agentv"] = _portable(publish_agentv_evaluation(agentv_dir, name="slm311-abstract-cot-warmup", claim="fixture_wiring_not_ship", cases=[{"id": "t3_fixture_wiring", "criteria": "T=3 must retain three addressable policies with AP-020/AP-019 phase evidence and no locked-suite selection.", "assertions": [{"id": "t3_policy_lineage_complete", "actual": complete, "operator": "eq", "expected": True}], "result": report["result"]}], version_stamp=stamp), agentv_dir)
    _rewrite_agentv_paths(agentv_dir)
    return report


def _markdown(report: dict[str, Any]) -> str:
    result = report["result"]
    return f"""# SLM-311 Abstract-CoT warm-up fixture

This local, hermetic T=3 run uses the existing AP-020 collector and AP-019
segment-payload builder. It records immutable per-phase command, configuration,
input/output hashes, token counts, and result references. It is **wiring only**:
no model checkpoint was written, no locked test selected an iteration, and it
does not support a promotion or ship claim.

- Iterations: {len(result['iterations'])}; policies: {', '.join(row['policy'] for row in result['iterations'])}.
- Resume: {result['reused']} reused, {result['ran']} newly run in this invocation.
- Meaningful-parse: not measured (there was no model evaluation).
- AgentEvals/AgentV checks the T=3 lineage assertion in the adjacent artifact.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default="slm311-abstract-cot-warmup")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-commit", default=_source_commit())
    parser.add_argument("--root", type=Path, default=Path("outputs/autoresearch"))
    parser.add_argument("--mode", choices=("plan-only", "fixture"), default="plan-only")
    parser.add_argument("--resume", action="store_true", help="Matching completed stages are always reused.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--agentv-dir", type=Path, default=DEFAULT_AGENTV)
    parser.add_argument("--no-evidence", action="store_true", help="Test-only escape hatch; normal fixture runs write durable evidence.")
    args = parser.parse_args(argv)
    campaign = AbstractWarmupCampaignV1(args.campaign_id, iterations=args.iterations, seed=args.seed, source_commit=args.source_commit)
    if args.dry_run or args.mode == "plan-only":
        print(json.dumps({"campaign": campaign.__dict__, "mode": args.mode, "claim": "no execution"}, indent=2, sort_keys=True))
        return 0
    supplied_args = list(argv) if argv is not None else sys.argv[1:]
    command = shlex.join(["scripts/run_abstract_cot_warmup.py", *supplied_args])
    result = run_abstract_cot_warmup(
        campaign, root=args.root, runner=fixture_iteration_runner(campaign, command=command)
    )
    if args.no_evidence:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    stamp = build_version_stamp("harness.experiments", COMPONENT)
    report = _evidence(result, stamp=stamp, agentv_dir=args.agentv_dir)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.json_out), "ship_eligible": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
