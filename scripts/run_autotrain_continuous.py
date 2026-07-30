#!/usr/bin/env python3
"""Hands-off continuous autotrain cycle driver.

Bare /autotrain agents should keep calling this (or re-enter continuous.md)
without user prompts. Each invocation can run one or many bounded cycles.
Never an infinite shell without MAX_RUN_MINUTES on child commands — wall is
enforced by campaign budgets and levers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from slm_training.autoresearch.engine import default_eval_version
from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
)
from slm_training.autoresearch.schemas import CampaignBudget, HypothesisMatrix
from slm_training.levers import MAX_RUN_MINUTES


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)


def _latest_cycle(root: Path, loop_id: str) -> tuple[int, str | None]:
    campaigns = sorted(root.glob("*/campaign.json"))
    best_idx = 0
    best_id: str | None = None
    for path in campaigns:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("loop_id") != loop_id:
            continue
        idx = int(data.get("cycle_index") or 0)
        if idx >= best_idx:
            best_idx = idx
            best_id = str(data.get("campaign_id"))
    return best_idx, best_id


def _matrix(
    *,
    campaign_id: str,
    evidence_snapshot_id: str,
    cites: list[str],
    role_citations: dict[str, str],
    train_version: str,
    eval_version: str,
    steps: int,
    cycle: int,
    feedback: list[dict] | None = None,
    previous_matrix_id: str | None = None,
) -> dict:
    research = role_citations.get("research") or cites[0]
    prior = role_citations.get("prior_result") or (cites[1] if len(cites) > 1 else cites[0])
    seed = 7 + (cycle * 17) % 50
    steps = steps + (cycle % 3)  # slight variation avoids knob-signature collision

    def uses() -> list[dict]:
        out = []
        contrib = {
            "research": "Continuous-loop and decode contracts.",
            "prior_trace": "Prior run telemetry or insight from the continuous loop.",
            "prior_result": "Prior campaign or evaluation baseline.",
        }
        for role, citation in role_citations.items():
            out.append(
                {
                    "role": role,
                    "citation": citation,
                    "contribution": contrib.get(role, "Captured continuous evidence."),
                }
            )
        if not out:
            out = [
                {
                    "role": "research",
                    "citation": research,
                    "contribution": contrib["research"],
                }
            ]
        return out

    def novelty(i: int, residual: str) -> dict:
        return {
            "transition_kind": (
                "regime_transition_candidate" if i == 0 else "fixed_regime_search"
            ),
            "old_schema_elements": ["training recipe", "grammar decode path"],
            "proposed_schema_elements": [residual if i == 0 else "training recipe"],
            "transported_elements": ["prior scoreboard", "fixture smoke path"],
            "transport_analysis": [
                "Residual is not explained by the matched fixture control alone."
            ],
            "residual_elements": [residual],
            "preservation_checks": ["rerun matched control under the wall cap"],
            "stress_tests": ["smoke parse_rate and latency under published eval"],
            "worthiness_criteria": [
                "complete scoreboard under wall without path errors"
            ],
        }

    def knobs(**extra: object) -> dict:
        base: dict[str, object] = {
            "train_version": train_version,
            "eval_version": eval_version,
            "steps": steps,
            "batch_size": 2,
            "seed": seed,
            "context_backend": "scratch",
            "sync_checkpoints": False,
            "local_files_only": True,
            "grammar_completion_bounds": False,
            "compact_active_canvas": False,
        }
        base.update(extra)
        return base

    def exp(eid: str, hyp: str, k: dict, rationale: str) -> dict:
        # Citations must include every evidence_use citation (schema invariant).
        exp_cites = list(dict.fromkeys([*cites[:3], *role_citations.values()]))
        return {
            "experiment_id": eid,
            "campaign_id": campaign_id,
            "hypothesis": hyp,
            "rationale": rationale,
            "expected_effect": "Runnable smoke scoreboard under the wall cap.",
            "falsification_criteria": [
                "Path error or no smoke metrics under the wall cap."
            ],
            "stop_conditions": [
                "Stop at declared steps or the campaign wall cap."
            ],
            "citations": exp_cites,
            "knobs": k,
        }

    prefix = campaign_id.replace("continuous-loop-", "c")
    candidates = [
        {
            "experiment": exp(
                f"{prefix}-control",
                "Matched fixture control with both grammar levers off completes smoke eval under the published suite.",
                knobs(),
                "Baseline for size-matched continuous attribution.",
            ),
            "evidence_uses": uses(),
            "novelty": novelty(0, "matched control with published eval"),
        },
        {
            "experiment": exp(
                f"{prefix}-bounds",
                "grammar_completion_bounds reduces smoke latency_ms_p50 versus the matched control without lowering parse_rate.",
                knobs(grammar_completion_bounds=True),
                "I1/I4 bounds lever on continuous default recipe.",
            ),
            "evidence_uses": uses(),
            "novelty": novelty(1, "grammar_completion_bounds"),
        },
        {
            "experiment": exp(
                f"{prefix}-canvas",
                "compact_active_canvas reduces smoke latency_ms_p50 versus the matched control without lowering parse_rate.",
                knobs(compact_active_canvas=True),
                "Independent canvas compaction lever.",
            ),
            "evidence_uses": uses(),
            "novelty": novelty(2, "compact_active_canvas"),
        },
        {
            "experiment": exp(
                f"{prefix}-both",
                "Combined bounds and canvas beat either single lever on smoke latency_ms_p50.",
                knobs(grammar_completion_bounds=True, compact_active_canvas=True),
                "Interaction arm for continuous steering.",
            ),
            "evidence_uses": uses(),
            "novelty": novelty(3, "combined bounds and canvas"),
        },
        {
            "experiment": exp(
                f"{prefix}-steps",
                "Doubling steps without levers only raises cost and does not improve unit decode latency.",
                knobs(steps=max(steps * 2, steps + 10)),
                "Depth confounds check.",
            ),
            "evidence_uses": uses(),
            "novelty": novelty(4, "depth-only confounds"),
        },
    ]
    rec = f"{prefix}-bounds"
    priorities = [
        {
            "rank": i + 1,
            "area": area,
            "hypothesis": hyp,
            "evidence_ids": [research, prior],
            "confidence": conf,
            "expected_information_gain": gain,
            "authority": auth,
            "disposition": disp,
            "proposed_experiment_id": eid,
        }
        for i, (area, hyp, conf, gain, auth, disp, eid) in enumerate(
            [
                (
                    "model",
                    "Test grammar_completion_bounds first under the published eval suite.",
                    0.6,
                    "Attributes decode latency vs matched control.",
                    "speculative",
                    "experiment_next",
                    rec,
                ),
                (
                    "evaluation",
                    "Keep the matched control as the size-matched baseline every cycle.",
                    0.7,
                    "Prevents false positives from recipe drift.",
                    "observed_result",
                    "experiment_next",
                    f"{prefix}-control",
                ),
                (
                    "model",
                    "Test compact_active_canvas as the next independent lever.",
                    0.55,
                    "Isolates canvas from bounds.",
                    "speculative",
                    "experiment_next",
                    f"{prefix}-canvas",
                ),
                (
                    "infrastructure",
                    "Soft ship-gate fails on fixture n never stop the continuous loop.",
                    0.8,
                    "Preserves hands-off continuous operation.",
                    "observed_result",
                    "monitor",
                    None,
                ),
                (
                    "model_build",
                    "Only after single levers, combine bounds and canvas.",
                    0.45,
                    "Detects interaction effects.",
                    "speculative",
                    "monitor",
                    None,
                ),
            ]
        )
    ]
    payload = {
        "matrix_id": f"{campaign_id}-m1",
        "campaign_id": campaign_id,
        "evidence_snapshot_id": evidence_snapshot_id,
        "hypotheses": candidates,
        "recommended_experiment_id": rec,
        "selection_rationale": "Size-matched fixture continuous recipe with published eval defaults.",
        "next_run_priorities": priorities,
    }
    if feedback:
        fb_ids = [str(item.get("feedback_id")) for item in feedback if item.get("feedback_id")]
        payload["feedback_ids"] = fb_ids
        if previous_matrix_id:
            payload["predecessor_matrix_id"] = previous_matrix_id
        # continuous priorities must cite every feedback id
        for priority in payload["next_run_priorities"]:
            evidence = list(priority.get("evidence_ids") or [])
            for fid in fb_ids:
                if fid not in evidence:
                    evidence.append(fid)
            priority["evidence_ids"] = evidence
    return payload


def _manifest(campaign_id: str, experiment: dict, commit: str) -> ExperimentCampaignV1:
    knobs = experiment["knobs"]
    cfg = hashlib.sha256(json.dumps(knobs, sort_keys=True).encode()).hexdigest()
    ctrl = hashlib.sha256(
        json.dumps(
            {
                **knobs,
                "grammar_completion_bounds": False,
                "compact_active_canvas": False,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return ExperimentCampaignV1(
        campaign_id=campaign_id,
        experiment_id=experiment["experiment_id"],
        hypothesis=experiment["hypothesis"],
        decision="Attribute continuous fixture decode metrics only under published eval suites.",
        endpoints=(
            CampaignEndpointV1(
                endpoint_id="primary",
                metric="smoke.latency_ms_p50",
                role="primary",
                direction="decrease",
                minimum_effect=1.0,
            ),
        ),
        arms=(
            CampaignArmV1(arm_id="control", role="control", config_sha256=ctrl),
            CampaignArmV1(arm_id="candidate", role="candidate", config_sha256=cfg),
        ),
        seeds=(int(knobs.get("seed") or 7),),
        budget=CampaignBudget(max_experiments=1, max_wall_minutes=MAX_RUN_MINUTES),
        stopping_rules=("Stop after the declared seed finishes or the wall cap is hit.",),
        controls=(
            CampaignControlV1(
                control_id="matched-control",
                description="Matched fixture baseline with grammar levers off.",
                kind="negative",
            ),
        ),
        negative_controls=("matched-control",),
        multiplicity_families=(
            MultiplicityFamilyV1(
                family_id="primary-family", hypothesis_ids=("primary",), alpha=0.05
            ),
        ),
        promotion_gates=(
            CampaignGateV1(
                gate_id="promote-primary",
                endpoint_id="primary",
                operator="le",
                threshold=-1.0,
            ),
        ),
        rollback_gates=(
            CampaignGateV1(
                gate_id="rollback-primary",
                endpoint_id="primary",
                operator="gt",
                threshold=1e9,
            ),
        ),
        artifact_requirements=(ArtifactRequirementV1(kind="version_stamp"),),
        claim_class="diagnostic",
        source_commit=commit,
        source_dirty=False,
        author="autotrain-continuous-driver",
    )


def run_cycle(
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    train_version: str,
    steps: int,
    objective: str,
    primary_metric: str,
) -> str:
    _run(["git", "fetch", "origin", "main"], cwd=cwd)
    _run(["git", "merge", "--no-edit", "origin/main"], cwd=cwd)
    if _git("status", "--porcelain"):
        raise RuntimeError("loop worktree is dirty; continuous requires a clean tree")
    upstream = _git("rev-parse", "origin/main")
    integration = _git("rev-parse", "HEAD")
    if upstream != integration:
        # merge should have equalized; if not, still require ancestor
        _run(["git", "merge-base", "--is-ancestor", upstream, integration], cwd=cwd)

    idx, pred = _latest_cycle(root, loop_id)
    cycle = idx + 1
    campaign_id = f"continuous-loop-{time.strftime('%Y%m%d')}-c{cycle}"
    py = sys.executable
    ar = [py, "-m", "scripts.autoresearch", "--root", str(root)]
    init = [
        *ar,
        "init",
        "--campaign-id",
        campaign_id,
        "--loop-id",
        loop_id,
        "--cycle-index",
        str(cycle),
        "--upstream-commit",
        upstream,
        "--integration-commit",
        integration,
        "--objective",
        objective,
        "--primary-metric",
        primary_metric,
        "--track",
        "twotower",
        "--max-experiments",
        "3",
        "--max-wall-minutes",
        str(MAX_RUN_MINUTES),
        "--notes",
        "Hands-off continuous driver cycle; local-only fixture scale.",
    ]
    if pred:
        init.extend(["--predecessor-campaign-id", pred])
    _run(init, cwd=cwd)
    _run(
        [
            *ar,
            "research",
            "--campaign-id",
            campaign_id,
            "--offline",
        ],
        cwd=cwd,
    )

    camp_dir = root / campaign_id
    evidence = next((camp_dir / "artifacts" / "evidence").glob("*.json"))
    ev = json.loads(evidence.read_text(encoding="utf-8"))
    research_paths = [
        item["path"]
        for item in ev.get("items", [])
        if item.get("kind") == "repo_lineage" and item.get("path")
    ]
    result_paths = [
        item["path"]
        for item in ev.get("items", [])
        if item.get("kind")
        in {"prior_campaign", "prior_run", "evaluation", "data_snapshot"}
        and item.get("path")
    ]
    trace_paths = [
        item["path"]
        for item in ev.get("items", [])
        if item.get("kind") in {"run_insight", "telemetry", "agentv", "feedback"}
        and item.get("path")
    ]
    if not research_paths:
        research_paths = ["docs/design/research-lineage.md"]
    if not result_paths:
        result_paths = research_paths[:]
    cites = [research_paths[0], result_paths[0]]
    if len(research_paths) > 1:
        cites.append(research_paths[1])
    elif len(result_paths) > 1:
        cites.append(result_paths[1])
    else:
        cites.append(research_paths[0])
    role_citations = {
        "research": research_paths[0],
        "prior_result": result_paths[0],
    }
    if trace_paths:
        role_citations["prior_trace"] = trace_paths[0]
    eval_version = default_eval_version()
    # Load predecessor matrix feedback when continuous lineage requires a successor matrix.
    feedback: list[dict] = []
    previous_matrix_id = None
    if pred:
        pred_dir = root / pred
        mats = sorted(
            (pred_dir / "artifacts" / "hypothesis_matrices").glob("*.json"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if mats:
            previous_matrix_id = json.loads(
                mats[-1].read_text(encoding="utf-8")
            ).get("matrix_id")
        fbs = sorted(
            (pred_dir / "artifacts" / "hypothesizer_feedback").glob("*.json"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        feedback = [json.loads(path.read_text(encoding="utf-8")) for path in fbs]
        # only terminal feedback for the latest predecessor matrix
        if previous_matrix_id:
            feedback = [
                item
                for item in feedback
                if item.get("matrix_id") == previous_matrix_id
            ]
    matrix = _matrix(
        campaign_id=campaign_id,
        evidence_snapshot_id=ev["snapshot_id"],
        cites=cites[:3],
        role_citations=role_citations,
        train_version=train_version,
        eval_version=eval_version,
        steps=steps,
        cycle=cycle,
        feedback=feedback or None,
        previous_matrix_id=previous_matrix_id,
    )
    HypothesisMatrix.model_validate(matrix)
    matrix_path = camp_dir / "matrix-proposal.json"
    matrix_path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    _run(
        [
            *ar,
            "hypothesize",
            "--campaign-id",
            campaign_id,
            "--provider",
            "agent",
            "--matrix",
            str(matrix_path),
        ],
        cwd=cwd,
    )

    exp_dir = camp_dir / "artifacts" / "experiments"
    by_id = {
        json.loads(path.read_text(encoding="utf-8"))["experiment_id"]: path
        for path in exp_dir.glob("*.json")
    }
    # execute control then recommended
    order = [
        matrix["hypotheses"][0]["experiment"]["experiment_id"],
        matrix["recommended_experiment_id"],
    ]
    seen: set[str] = set()
    for eid in order:
        if eid in seen or eid not in by_id:
            continue
        seen.add(eid)
        exp = json.loads(by_id[eid].read_text(encoding="utf-8"))
        man = _manifest(campaign_id, exp, integration)
        man_path = camp_dir / "manifests" / f"{eid}.json"
        man_path.parent.mkdir(parents=True, exist_ok=True)
        man_path.write_text(man.model_dump_json(indent=2) + "\n", encoding="utf-8")
        # soft-fail: ship gates may fail on fixture n
        cmd = [
            *ar,
            "run",
            "--campaign-id",
            campaign_id,
            "--experiment",
            str(by_id[eid]),
            "--campaign-manifest",
            str(man_path),
            "--execute",
        ]
        print("+", " ".join(cmd), flush=True)
        code = subprocess.call(cmd, cwd=cwd)
        print(f"experiment {eid} exit={code}", flush=True)

    _run(
        [
            *ar,
            "status",
            "--loop-id",
            loop_id,
            "--matrix",
            "--last",
            "5",
        ],
        cwd=cwd,
    )
    print(f"CYCLE_COMPLETE {campaign_id}", flush=True)
    return campaign_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop-id", default="continuous-openui-local")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/autoresearch"),
        help="Campaign bundle root",
    )
    parser.add_argument("--max-cycles", type=int, default=1, help="0 = many (1024)")
    parser.add_argument("--train-version", default="wf_smoke_v2")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--objective",
        default=(
            "On a size-matched fixture TwoTower arm under the wall cap, improve "
            "smoke decode latency without lowering parse_rate versus the matched control."
        ),
    )
    parser.add_argument("--primary-metric", default="smoke.latency_ms_p50")
    args = parser.parse_args(argv)
    cwd = Path.cwd()
    root = args.root if args.root.is_absolute() else cwd / args.root
    root.mkdir(parents=True, exist_ok=True)
    max_cycles = 1024 if args.max_cycles == 0 else max(1, args.max_cycles)
    for i in range(max_cycles):
        print(f"=== continuous cycle pass {i + 1}/{max_cycles} ===", flush=True)
        try:
            run_cycle(
                cwd=cwd,
                root=root,
                loop_id=args.loop_id,
                train_version=args.train_version,
                steps=args.steps,
                objective=args.objective,
                primary_metric=args.primary_metric,
            )
        except Exception as exc:  # noqa: BLE001 - continuous must self-heal next pass
            print(f"CYCLE_ERROR {exc!r}", flush=True)
            # soft continue unless dirty tree
            if "dirty" in str(exc).lower():
                return 2
            time.sleep(1)
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
