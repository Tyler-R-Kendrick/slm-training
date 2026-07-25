#!/usr/bin/env python3
"""Run the SLM-183 powered cluster-aware confirmation protocol fixture.

Example:
  python -m scripts.run_flow_power_protocol --mode plan-only
  python -m scripts.run_flow_power_protocol --mode fixture
  python -m scripts.run_flow_power_protocol --mode analyze-existing \
      --iter-json outputs/runs/some-run/iter.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.autoresearch.experiment_campaign import campaign_manifest_sha256
from slm_training.autoresearch.schemas import CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.slm183_power_protocol import (
    MATRIX_SET,
    EXPERIMENT_ID,
    MATRIX_VERSION,
    ConfirmationSuiteManifest,
    PowerProtocolReport,
    analyze_existing_iter,
    build_default_manifest,
    build_experiment_campaign,
    render_markdown,
    run_variance_fixture,
)
from slm_training.harnesses.experiments.slm287_power_protocol import (
    LOCKED_MANIFEST,
    build_locked_campaign,
    execute_local_grid,
    execute_local_shard,
    load_locked_protocol,
    merge_locked_shards,
    summarize_cells,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm183-power-protocol-20260720.json"
_DESIGN_MD = "docs/design/iter-slm183-power-protocol-20260720.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _locked_shard_path(
    output_dir: Path, protocol_sha256: str, seed: int, backend: str, shard_index: int
) -> Path:
    return (
        output_dir
        / "shards"
        / protocol_sha256
        / f"seed{seed}"
        / backend
        / f"shard_{shard_index}.json"
    )


def _lock_slm287_campaign(output_dir: Path, protocol: Any) -> tuple[CampaignStore, Any]:
    campaign = build_locked_campaign(protocol)
    store = CampaignStore(campaign.campaign_id, output_dir / "campaigns")
    spec = CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="Measure the five-seed two-backend locked local baseline.",
            primary_metric="binding_aware_meaningful_v2",
            researcher_mode="diagnostic",
            budget=campaign.budget,
            created_at=campaign.created_at,
    )
    if (store.root / "campaign.json").exists():
        existing = store.load_campaign()
        if (
            existing.campaign_id != spec.campaign_id
            or existing.primary_metric != spec.primary_metric
            or existing.budget != spec.budget
            or existing.created_at != spec.created_at
        ):
            raise ValueError("existing SLM-287 campaign spec is incompatible")
    else:
        store.initialize(spec)
    return store, store.lock_experiment_campaign(campaign)


def _build_payload(
    mode: str,
    output_dir: Path,
    seeds: tuple[int, ...],
    n_targets: int,
    paths_per_target: int,
    n_seeds: int,
    iter_json: Path | None,
    locked_manifest: Path,
    locked_limit: int | None,
    locked_seed: int | None,
    locked_backend: str | None,
    shard_index: int | None,
    shard_count: int | None,
) -> tuple[dict[str, Any], str]:
    if mode == "locked-plan":
        protocol = load_locked_protocol(locked_manifest, limit=locked_limit)
        return (
            {
                "schema": "Slm287LockedPowerProtocolPlanV1",
                "experiment_id": "slm287-locked-power-protocol",
                "status": "planned",
                "claim_class": "diagnostic",
                "protocol": protocol.to_dict(),
                "version_stamp": build_version_stamp(
                    "harness.experiments",
                    "evals.power_protocol",
                    "data.locked_eval_manifest",
                ),
            },
            "python -m scripts.run_flow_power_protocol --mode locked-plan",
        )
    if mode == "locked-analyze":
        if iter_json is None:
            raise ValueError("--mode locked-analyze requires --iter-json cells JSON")
        protocol = load_locked_protocol(locked_manifest, limit=locked_limit)
        raw = json.loads(iter_json.read_text(encoding="utf-8"))
        cells = raw.get("cells", raw) if isinstance(raw, dict) else raw
        if not isinstance(cells, list):
            raise ValueError("locked cells JSON must be a list or contain cells")
        return summarize_cells(protocol, cells), (
            "python -m scripts.run_flow_power_protocol --mode locked-analyze "
            f"--iter-json {iter_json}"
        )
    if mode == "locked-run":
        protocol = load_locked_protocol(locked_manifest, limit=locked_limit)
        cells = execute_local_grid(
            protocol,
            manifest_path=locked_manifest,
            output_dir=output_dir,
        )
        return summarize_cells(protocol, cells), (
            "python -m scripts.run_flow_power_protocol --mode locked-run "
            f"--locked-manifest {locked_manifest}"
        )
    if mode == "locked-shard":
        if locked_limit is not None:
            raise ValueError("locked shards require the complete locked manifest")
        if locked_seed is None or locked_backend is None:
            raise ValueError("locked-shard requires --locked-seed and --locked-backend")
        if shard_index is None or shard_count is None:
            raise ValueError("locked-shard requires --shard-index and --shard-count")
        protocol = load_locked_protocol(locked_manifest)
        store, lock = _lock_slm287_campaign(output_dir, protocol)
        store.append_event(
            "shard_started",
            experiment_id=lock.manifest.experiment_id,
            status="running",
            detail={"campaign_manifest_sha256": lock.manifest_sha256, "shard_index": shard_index},
        )
        shard = execute_local_shard(
            protocol,
            manifest_path=locked_manifest,
            output_dir=output_dir,
            seed=locked_seed,
            backend=locked_backend,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        from slm_training.evals.agentv import publish_agentv_evaluation

        stamp = build_version_stamp(
            "harness.experiments", "evals.power_protocol", "data.locked_eval_manifest"
        )
        shard["agentv"] = publish_agentv_evaluation(
            output_dir,
            name=f"slm287-{locked_backend}-seed{locked_seed}-shard{shard_index}",
            claim="The bounded locked shard completed without a decode timeout.",
            cases=[
                {
                    "id": f"seed{locked_seed}-{locked_backend}-shard{shard_index}",
                    "criteria": "The canonical evaluator returned all declared variants for every assigned locked record.",
                    "pass": set(shard["records"]) == set(shard["record_ids"]),
                    "result": {"record_ids": shard["record_ids"], "variants": list(shard["cell_metrics"])},
                }
            ],
            version_stamp=stamp,
        )
        shard["campaign_manifest_sha256"] = lock.manifest_sha256
        path = _locked_shard_path(
            output_dir,
            str(shard["protocol_sha256"]),
            locked_seed,
            locked_backend,
            shard_index,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(shard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifact = store.write_artifact("locked_shards", shard)
        store.append_event(
            "shard_finished",
            experiment_id=lock.manifest.experiment_id,
            status="completed",
            artifact_sha256=artifact.stem,
            detail={"campaign_manifest_sha256": lock.manifest_sha256, "shard_index": shard_index},
        )
        return (
            {
                **shard,
                "status": "completed_shard_not_evidence",
                "claim_class": "diagnostic",
                "shard_path": str(path),
                "version_stamp": build_version_stamp(
                    "harness.experiments",
                    "evals.power_protocol",
                    "data.locked_eval_manifest",
                ),
            },
            "python -m scripts.run_flow_power_protocol --mode locked-shard "
            f"--locked-seed {locked_seed} --locked-backend {locked_backend} "
            f"--shard-index {shard_index} --shard-count {shard_count}",
        )
    if mode == "locked-merge":
        if locked_limit is not None:
            raise ValueError("locked merge requires the complete locked manifest")
        if shard_count is None:
            raise ValueError("locked-merge requires --shard-count")
        protocol = load_locked_protocol(locked_manifest)
        store, lock = _lock_slm287_campaign(output_dir, protocol)
        protocol_sha = str(protocol.to_dict()["protocol_sha256"])
        shards = []
        for seed in protocol.to_dict()["seeds"]:
            for backend in protocol.to_dict()["backends"]:
                for index in range(shard_count):
                    path = _locked_shard_path(output_dir, protocol_sha, seed, backend, index)
                    if not path.is_file():
                        raise ValueError(f"missing locked shard artifact: {path}")
                    shards.append(json.loads(path.read_text(encoding="utf-8")))
        result = summarize_cells(protocol, merge_locked_shards(protocol, shards, shard_count=shard_count))
        result["campaign_manifest_sha256"] = lock.manifest_sha256
        return result, (
            "python -m scripts.run_flow_power_protocol --mode locked-merge "
            f"--shard-count {shard_count}"
        )
    manifest = build_default_manifest(seeds=seeds)
    campaign = build_experiment_campaign(seeds=seeds)
    campaign_payload = campaign.model_dump(mode="json")
    campaign_sha = campaign_manifest_sha256(campaign)
    store: CampaignStore | None = None
    lock = None
    if mode in {"plan-only", "fixture"}:
        store = CampaignStore(campaign.campaign_id, output_dir / "campaigns")
        store.initialize(
            CampaignSpec(
                campaign_id=campaign.campaign_id,
                objective="Exercise paired, powered fixture campaign governance.",
                primary_metric=campaign.endpoints[0].metric,
                researcher_mode="fixture",
                budget=campaign.budget,
                created_at="2026-07-23T00:00:00Z",
            )
        )
        lock = store.lock_experiment_campaign(campaign)

    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "Slm183PowerProtocolManifestV1",
            "matrix_set": MATRIX_SET,
            "matrix_version": MATRIX_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": "plan_only",
            "claim_class": "wiring",
            "manifest": manifest.to_dict(),
            "experiment_campaign": campaign_payload,
            "campaign_manifest_sha256": campaign_sha,
            "version_stamp": build_version_stamp(
                "harness.autoresearch.experiment_campaign",
                "harness.experiments",
                "harness.experiments.slm183_power_protocol",
                "evals.power_protocol",
            ),
            "timestamp": _now(),
        }
        command = "python -m scripts.run_flow_power_protocol --mode plan-only"
        return payload, command

    if mode == "analyze-existing":
        if iter_json is None:
            raise ValueError("--mode analyze-existing requires --iter-json")
        analysis = analyze_existing_iter(iter_json)
        payload = {
            "schema": "Slm183PowerProtocolAnalysisV1",
            "matrix_set": MATRIX_SET,
            "matrix_version": MATRIX_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": "analysis",
            "claim_class": "wiring",
            "analysis": analysis,
            "experiment_campaign": campaign_payload,
            "campaign_manifest_sha256": campaign_sha,
            "version_stamp": build_version_stamp(
                "harness.autoresearch.experiment_campaign",
                "harness.experiments",
                "harness.experiments.slm183_power_protocol",
                "evals.power_protocol",
            ),
            "timestamp": _now(),
        }
        command = (
            f"python -m scripts.run_flow_power_protocol --mode analyze-existing "
            f"--iter-json {iter_json}"
        )
        return payload, command

    assert store is not None and lock is not None
    store.append_event(
        "experiment_started",
        experiment_id=campaign.experiment_id,
        status="running",
        detail={"campaign_manifest_sha256": lock.manifest_sha256},
    )
    report = run_variance_fixture(
        n_targets=n_targets,
        paths_per_target=paths_per_target,
        n_seeds=n_seeds,
        run_id=f"slm183-power-protocol-{_today_yyyymmdd()}",
        output_dir=output_dir,
        seed=seeds[0] if seeds else 0,
        seeds=seeds,
    )
    payload = report.to_dict()
    payload["experiment_campaign"] = campaign_payload
    payload["campaign_manifest_sha256"] = lock.manifest_sha256
    artifact = store.write_artifact("fixture_outcomes", payload)
    store.append_event(
        "experiment_finished",
        experiment_id=campaign.experiment_id,
        status="completed",
        artifact_sha256=artifact.stem,
        detail={"campaign_manifest_sha256": lock.manifest_sha256},
    )
    command = "python -m scripts.run_flow_power_protocol --mode fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "planned" and "protocol" in payload:
        protocol = payload["protocol"]
        return "\n".join(
            [
                "# SLM-287: locked five-seed × two-config baseline protocol",
                "",
                "**Claim class:** diagnostic protocol; no model result is claimed.",
                "",
                f"- locked manifest: `{protocol['locked_eval_manifest_sha256']}`",
                f"- locked test records: `{len(protocol['record_ids'])}`",
                f"- seeds: `{protocol['seeds']}`",
                f"- local configurations: `{protocol['backends']}`",
                "- primary metric: `binding_aware_meaningful_v2`",
                "- human rating: optional audit only, never a gate.",
                "",
                "The protocol fails closed: no numeric conclusion is emitted until all",
                "ten cells score the exact frozen record set under raw, constrained,",
                "and repaired decode with matching repeated-initialization hashes.",
                "",
                "## Exact command",
                "",
                f"```bash\n{command}\n```",
                "",
            ]
        )
    if status == "complete_local_grid" and "protocol" in payload:
        protocol = payload["protocol"]
        paired = payload["paired"]["bootstrap_ci"]
        return "\n".join(
            [
                "# SLM-287: locked five-seed × two-config local baseline",
                "",
                "**Claim class:** local diagnostic; not a ship or promotion result.",
                "",
                f"- locked manifest: `{protocol['locked_eval_manifest_sha256']}`",
                f"- locked test records: `{len(protocol['record_ids'])}`",
                f"- cells: `{len(payload['cells'])}` (five seeds × two local configurations)",
                f"- primary paired delta (design-on minus design-off): `{paired['estimate']:.6f}`",
                f"- paired 95% bootstrap CI: `[{paired['low']:.6f}, {paired['high']:.6f}]`",
                "- no best seed was selected; human ratings are not a gate.",
                "",
                "All cells used the canonical evaluator's raw, constrained, and repaired",
                "variants. This result is diagnostic only when a bounded locked prefix is",
                "used; a full 226-record sweep is required before any promotion decision.",
                "",
                "## Exact command",
                "",
                f"```bash\n{command}\n```",
                "",
            ]
        )
    if status == "interrupted_not_evidence" and "protocol" in payload:
        protocol = payload["protocol"]
        return "\n".join(
            [
                "# SLM-287: interrupted local locked-power execution",
                "",
                "**Claim class:** interrupted diagnostic; no numeric result is evidence.",
                "",
                f"- locked manifest: `{protocol['locked_eval_manifest_sha256']}`",
                f"- requested locked records: `{len(protocol['record_ids'])}`",
                f"- reason: `{payload['reason']}`",
                "- ship gate: not evaluated; human ratings are not a gate.",
                "",
                "The runner rejects a timed-out cell before aggregation. No MDE, CI,",
                "or candidate-selection conclusion was emitted.",
                "",
                "## Exact command",
                "",
                f"```bash\n{command}\n```",
                "",
            ]
        )
    if status == "plan_only":
        manifest = ConfirmationSuiteManifest.from_dict(payload["manifest"])
        lines = [
            "# SLM-183 (PQR): powered cluster-aware confirmation protocol plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            "**Run date:** 2026-07-20",
            "",
            "**Machine-readable result:** ["
            "`iter-slm183-power-protocol-20260720.json`"
            "](iter-slm183-power-protocol-20260720.json)",
            "",
            "This is a plan-only manifest. The power-protocol statistical "
            "utilities, cluster bootstrap, ICC, and MDE simulation are wired; "
            "run `--mode fixture` to execute the CPU-only simulation.",
            "",
            "## Hypothesis",
            "",
            "A powered confirmation protocol can separate seed variance from "
            "target variance and estimate the minimum detectable effect size.",
            "",
            "## Falsifier",
            "",
            "The protocol collapses seed and target variance into a single "
            "pooled estimate and cannot produce a calibrated MDE curve.",
            "",
            "## Manifest",
            "",
            f"- suite_role: `{manifest.suite_role}`",
            f"- generator_version: `{manifest.generator_version}`",
            f"- target_cluster_id: `{manifest.target_cluster_id}`",
            f"- primary_endpoint: `{manifest.primary_endpoint}`",
            f"- primary_contrast: `{manifest.primary_contrast}`",
            f"- mde: `{manifest.mde}`",
            f"- alpha: `{manifest.alpha}`",
            f"- power: `{manifest.power}`",
            f"- multiplicity_family: `{manifest.multiplicity_family}`",
            "",
            "## Exact command",
            "",
            f"```bash\n{command}\n```",
            "",
        ]
        return "\n".join(lines)

    if status == "analysis":
        analysis = payload.get("analysis", {})
        lines = [
            "# SLM-183 (PQR): existing-iter power-protocol analysis",
            "",
            f"Source: `{analysis.get('source', 'unknown')}`",
            "",
            f"- n_records: **{analysis.get('n_records', 0)}**",
            f"- n_successes: **{analysis.get('n_successes', 0)}**",
            f"- success_rate: **{analysis.get('success_rate', 0.0):.3f}**",
            f"- seed_variance: **{analysis.get('seed_variance', 0.0):.4f}**",
            "",
            "## Exact command",
            "",
            f"```bash\n{command}\n```",
            "",
        ]
        return "\n".join(lines)

    report = PowerProtocolReport.from_dict(payload)
    return (
        render_markdown(report)
        + "\n## Exact command\n\n"
        + f"```bash\n{command}\n```\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-183 PQR powered cluster-aware confirmation protocol fixture",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={
            "plan-only", "fixture", "analyze-existing", "locked-plan", "locked-analyze",
            "locked-run", "locked-shard", "locked-merge",
        },
        default="plan-only",
        help="Run mode: plan-only writes the manifest; fixture runs the CPU simulation; "
        "analyze-existing reads an iter JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts (default: outputs/runs/slm183-power-protocol-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_int_tuple,
        default="0,1,2,3,4",
        help="Comma-separated random seeds for fixture mode (default: 0,1,2,3,4).",
    )
    parser.add_argument(
        "--n-targets",
        type=int,
        default=50,
        help="Number of synthetic targets (default: 50).",
    )
    parser.add_argument(
        "--paths-per-target",
        type=int,
        default=3,
        help="Independent paths per target (default: 3).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of seeds per target (default: 5).",
    )
    parser.add_argument(
        "--iter-json",
        type=Path,
        help="Path to existing iter JSON or SLM-287 locked cells JSON.",
    )
    parser.add_argument(
        "--locked-manifest",
        type=Path,
        default=LOCKED_MANIFEST,
        help="Immutable SLM-285 promotion manifest used by SLM-287.",
    )
    parser.add_argument(
        "--locked-limit",
        type=int,
        help="Bounded diagnostic prefix of locked_test; never a promotion result.",
    )
    parser.add_argument("--locked-seed", type=int, help="Declared SLM-287 seed for one shard.")
    parser.add_argument("--locked-backend", help="Declared SLM-287 backend for one shard.")
    parser.add_argument("--shard-index", type=int, help="Zero-based deterministic locked shard index.")
    parser.add_argument("--shard-count", type=int, help="Number of deterministic locked shards.")
    parser.add_argument(
        "--write-design-docs",
        action="store_true",
        help="Publish fixture JSON/Markdown under docs/design (off in tests).",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = args.output_dir or Path(
        f"outputs/runs/slm183-power-protocol-{_today_yyyymmdd()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "fixture" and args.n_seeds != len(args.seeds):
        print(
            "error: --n-seeds must equal the number of declared --seeds",
            file=sys.stderr,
        )
        return 2

    try:
        payload, command = _build_payload(
            args.mode,
            output_dir,
            args.seeds,
            args.n_targets,
            args.paths_per_target,
            args.n_seeds,
            args.iter_json,
            args.locked_manifest,
            args.locked_limit,
            args.locked_seed,
            args.locked_backend,
            args.shard_index,
            args.shard_count,
        )
    except TimeoutError as exc:
        protocol = load_locked_protocol(args.locked_manifest, limit=args.locked_limit)
        payload = {
            "schema": "Slm287LockedPowerProtocolAttemptV1",
            "experiment_id": "slm287-locked-power-protocol",
            "status": "interrupted_not_evidence",
            "claim_class": "diagnostic",
            "reason": str(exc),
            "protocol": protocol.to_dict(),
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "evals.power_protocol",
                "data.locked_eval_manifest",
            ),
        }
        command = "python -m scripts.run_flow_power_protocol --mode locked-run"
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.mode == "fixture" and args.write_design_docs:
        command = (
            "python -m scripts.run_flow_power_protocol --mode fixture "
            f"--output-dir {output_dir} "
            f"--n-targets {args.n_targets} "
            f"--paths-per-target {args.paths_per_target} "
            f"--n-seeds {args.n_seeds} "
            f"--seeds {','.join(str(seed) for seed in args.seeds)}"
        )
        if args.write_design_docs:
            command += " --write-design-docs"

    if args.mode in {"plan-only", "fixture", "analyze-existing"}:
        payload["schema"] = "Slm183PowerProtocolReportV1"
        payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / (
        "slm287_locked_power_protocol_report.json"
        if args.mode.startswith("locked-")
        else "slm183_power_protocol_report.json"
    )
    run_json.write_text(report_text, encoding="utf-8")

    if args.mode in {"fixture", "locked-plan", "locked-analyze", "locked-run", "locked-merge"} and args.write_design_docs:
        root = Path(__file__).resolve().parents[1]
        json_path = root / (
            "docs/design/iter-slm287-locked-power-protocol-20260724.json"
            if args.mode.startswith("locked-")
            else _DESIGN_JSON
        )
        md_path = root / (
            "docs/design/iter-slm287-locked-power-protocol-20260724.md"
            if args.mode.startswith("locked-")
            else _DESIGN_MD
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(report_text, encoding="utf-8")

        md_path.write_text(_build_markdown(payload, command), encoding="utf-8")

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
