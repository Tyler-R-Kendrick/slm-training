#!/usr/bin/env python3
"""Collect a bounded SLM-219 checkpoint trajectory into durable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import torch
import torch.nn as nn

from slm_training.harnesses.experiments.slm219_correlation_traps import (
    native_trap_metrics,
)
from slm_training.versioning import build_version_stamp

DEFAULT_OUTPUT = "docs/design/iter-slm219-correlation-trap-evidence-20260723.json"
DEFAULT_AGENTV = "docs/design/iter-slm219-correlation-trap-agentv-20260723"
CHECKPOINT_ROLES = {
    "mlp_out": "denoiser.layers.0.mlp.2.weight",
    "cross_attn_out": "denoiser.layers.0.cross_attn.out_proj.weight",
    "lm_head": "denoiser.lm_head.weight",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_state(path: Path, expected_sha: str) -> dict[str, torch.Tensor]:
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"checkpoint hash mismatch for {path}: {actual_sha} != {expected_sha}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no state_dict: {path}")
    return state


def _repetition_rate(evaluation: dict[str, Any]) -> float:
    details = evaluation.get("details", [])
    repeated = sum(
        "duplicate_subtree_spam"
        in detail.get("semantic_meaning_report_v2", {}).get("reason_codes", [])
        for detail in details
    )
    return repeated / len(details) if details else 0.0


def _copy_agentv_bundle(
    eval_dir: Path,
    destination: Path,
    label: str,
    repo_root: Path,
) -> dict[str, Any]:
    agentv = eval_dir / "agentv"
    spec = next(agentv.glob("*.eval.jsonl"))
    run_dir = next(path for path in agentv.iterdir() if path.is_dir())
    sources = {
        "spec": spec,
        "benchmark": run_dir / "benchmark.json",
        "index": run_dir / "index.jsonl",
        "timing": run_dir / "timing.json",
    }
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    normalized_spec = str((destination / f"{label}-spec.jsonl").relative_to(repo_root))
    for kind, source in sources.items():
        suffix = ".jsonl" if source.suffix == ".jsonl" else ".json"
        target = destination / f"{label}-{kind}{suffix}"
        content = source.read_text(encoding="utf-8")
        content = content.replace(str(spec), normalized_spec)
        content = content.replace(
            quote(str(spec), safe=""),
            quote(normalized_spec, safe=""),
        )
        target.write_text(content, encoding="utf-8")
        artifacts[kind] = {
            "path": str(target.relative_to(repo_root)),
            "sha256": _sha256(target),
            "bytes": target.stat().st_size,
        }
    return artifacts


def _weightwatcher_statistics(matrix: torch.Tensor) -> dict[str, Any]:
    import weightwatcher as weightwatcher

    layer = nn.Linear(matrix.shape[1], matrix.shape[0], bias=False)
    layer.weight.data.copy_(matrix)
    frame = weightwatcher.WeightWatcher(model=nn.Sequential(layer)).analyze(
        min_evals=10,
        mp_fit=False,
        plot=False,
        randomize=False,
    )
    row = frame.iloc[0]
    return {
        "backend": f"weightwatcher-{weightwatcher.__version__}",
        "status": str(row["status"]),
        "alpha": float(row["alpha"]),
        "alpha_weighted": float(row["alpha_weighted"]),
        "stable_rank": float(row["stable_rank"]),
        "spectral_norm": float(row["spectral_norm"]),
        "num_pl_spikes": int(row["num_pl_spikes"]),
    }


def collect(
    *,
    repo_root: Path,
    parent_dir: Path,
    trajectory_root: Path,
    eval_root: Path,
    output_path: Path,
    agentv_dir: Path,
    null_draws: int,
    run_weightwatcher: bool,
) -> dict[str, Any]:
    source_rows = [
        {
            "label": "t0000",
            "step": 0,
            "tokens": 0,
            "checkpoint": parent_dir / "last.pt",
            "eval_dir": eval_root / "slm219-e396-parent-t0-exact-context",
            "train_summary": None,
            "source_locator": (
                "hf://buckets/TKendrick/OpenUI/checkpoints/"
                "e396-balanced-type-head-continuation-r1/last.pt"
            ),
        },
        *[
            {
                "label": f"t{tokens:04d}",
                "step": None,
                "tokens": tokens,
                "checkpoint": (
                    trajectory_root
                    / f"slm219-e502-uniform-t{tokens}"
                    / "checkpoints"
                    / "last.pt"
                ),
                "eval_dir": (
                    eval_root / f"slm219-e502-uniform-t{tokens}-exact-context"
                ),
                "train_summary": trajectory_root
                / f"slm219-e502-uniform-t{tokens}"
                / "train_summary.json",
                "source_locator": (
                    "scratch:no-sync/deterministic-prefix-rerun/"
                    f"slm219-e502-uniform-t{tokens}/checkpoints/last.pt"
                ),
            }
            for tokens in (1000, 2000, 3000, 4000, 5000)
        ],
    ]

    rows: list[dict[str, Any]] = []
    parent_tensors: dict[str, torch.Tensor] = {}
    previous_tensors: dict[str, torch.Tensor] = {}
    for index, source in enumerate(source_rows):
        evaluation_path = source["eval_dir"] / "eval_smoke.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        expected_sha = str(evaluation["checkpoint_sha256"])
        state = _checkpoint_state(source["checkpoint"], expected_sha)
        summary = (
            json.loads(source["train_summary"].read_text(encoding="utf-8"))
            if source["train_summary"]
            else None
        )
        role_rows = {}
        for role, matrix_path in CHECKPOINT_ROLES.items():
            tensor = state[matrix_path].detach().cpu()
            if index == 0:
                parent_tensors[role] = tensor
            parent = parent_tensors[role]
            previous = previous_tensors.get(role, parent)
            role_rows[role] = {
                "matrix_path": matrix_path,
                "shape": list(tensor.shape),
                "trap": native_trap_metrics(
                    tensor,
                    null_draws=null_draws,
                    seed=21_900 + index,
                ).to_dict(),
                "weightwatcher": (
                    _weightwatcher_statistics(tensor)
                    if run_weightwatcher
                    else {
                        "status": "not_run",
                        "reason": "collector invoked without --weightwatcher",
                    }
                ),
                "rms_drift_from_parent": float(
                    torch.sqrt(torch.mean((tensor - parent).double().square()))
                ),
                "update_norm_from_previous": float(
                    torch.linalg.vector_norm((tensor - previous).double())
                ),
            }
            previous_tensors[role] = tensor

        label = str(source["label"])
        artifact_rows = _copy_agentv_bundle(
            source["eval_dir"],
            agentv_dir,
            label,
            repo_root,
        )
        rows.append(
            {
                "label": label,
                "run_id": (
                    summary["run_id"]
                    if summary
                    else "e396-balanced-type-head-continuation-r1"
                ),
                "step": int(summary["steps"]) if summary else 0,
                "tokens": int(summary["seen_target_tokens"]) if summary else 0,
                "checkpoint_sha256": expected_sha,
                "source_locator": source["source_locator"],
                "source_resolution": "verified_local_copy",
                "train_loss_proxy": summary.get("last_loss") if summary else None,
                "heldout_nll": None,
                "heldout_nll_unavailable_reason": (
                    "historical continuation did not emit a held-out NLL snapshot"
                ),
                "gradient_norm": None,
                "gradient_norm_unavailable_reason": (
                    "historical train telemetry did not persist gradient norms"
                ),
                "elapsed_wall_seconds": (
                    float(summary["elapsed_wall_seconds"]) if summary else 0.0
                ),
                "evaluation": {
                    "suite": evaluation["suite"],
                    "n": evaluation["n"],
                    "structural_similarity": evaluation["structural_similarity"],
                    "duplicate_subtree_rate": _repetition_rate(evaluation),
                    "meaningful_program_rate": evaluation["meaningful_program_rate"],
                    "component_type_recall": evaluation["component_type_recall"],
                    "placeholder_fidelity": evaluation["placeholder_fidelity"],
                    "reward_score": evaluation["reward_score"],
                    "eval_json_sha256": _sha256(evaluation_path),
                    "agentv": evaluation["agentv"]["summary"],
                    "agentv_artifacts": artifact_rows,
                },
                "roles": role_rows,
                "training": (
                    {
                        "device": summary["device"],
                        "context_backend": summary["telemetry"]["meta"][
                            "context_backend"
                        ],
                        "data_manifest_sha": summary["data_manifest_sha"],
                        "code_commit": summary["version_stamp"]["code_commit"],
                        "max_wall_minutes": summary["max_wall_minutes"],
                        "target_token_budget": summary["target_token_budget"],
                        "seen_prompt_tokens": summary["seen_prompt_tokens"],
                        "seen_target_tokens": summary["seen_target_tokens"],
                        "recipe": summary["recipe"],
                        "resumed_from_prior_stage": False,
                        "deterministic_prefix_rerun": True,
                        "shared_parent_seed_and_uniform_order": True,
                        "checkpoint_sync": "disabled_scratch_rejected_continuation",
                    }
                    if summary
                    else None
                ),
            }
        )

    payload = {
        "schema": "CorrelationTrapCheckpointEvidenceV1",
        "run_id": "slm219-e396-e500-prefix-trajectory-20260723",
        "matrix_set": "slm219_correlation_traps",
        "matrix_version": "ncs1-03-v2",
        "historical_code_commit": "f2ab01f8ae6af6be49db3f294cd166fe034b67a5",
        "data_manifest_sha": (
            "bc256915c79c6a07ff9ef253d5b6eb70fade30c5422ae8015c23f739c463bc62"
        ),
        "seed": 0,
        "family": "e396_parent_e500_auxiliary_continuation",
        "honesty_mode": "no-design-md-context",
        "null_draws": null_draws,
        "weightwatcher_comparison": {
            "requested": run_weightwatcher,
            "version": "0.7.5" if run_weightwatcher else None,
            "install_scope": "ephemeral project virtualenv; not a runtime dependency",
        },
        "roles": CHECKPOINT_ROLES,
        "points": rows,
        "checkpoint_disposition": (
            "scratch diagnostic only; not promoted, not synced, and not registered "
            "in the serving model card"
        ),
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm214_spectral_snapshot",
            "harness.experiments.slm219_correlation_traps",
        ),
    }
    payload["evidence_hash"] = _canonical_sha(
        {key: value for key, value in payload.items() if key != "version_stamp"}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--agentv-dir", type=Path, default=Path(DEFAULT_AGENTV))
    parser.add_argument("--null-draws", type=int, default=24)
    parser.add_argument("--weightwatcher", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else repo_root / args.output
    agentv = (
        args.agentv_dir
        if args.agentv_dir.is_absolute()
        else repo_root / args.agentv_dir
    )
    payload = collect(
        repo_root=repo_root,
        parent_dir=args.parent_dir,
        trajectory_root=args.trajectory_root,
        eval_root=args.eval_root,
        output_path=output,
        agentv_dir=agentv,
        null_draws=args.null_draws,
        run_weightwatcher=args.weightwatcher,
    )
    print(payload["evidence_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
