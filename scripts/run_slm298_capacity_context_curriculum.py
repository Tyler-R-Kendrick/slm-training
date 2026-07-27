#!/usr/bin/env python3
"""Run one bounded, preregistered local SLM-298 factorial cell at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from slm_training.autoresearch.experiment_campaign import (
    CampaignDeviationV1,
    CampaignLockV1,
    campaign_manifest_sha256,
)
from slm_training.harnesses.experiments.slm298_capacity_context_curriculum import (
    DERIVED_TRAIN_DIR,
    LOCKED_MANIFEST,
    build_campaign,
    load_protocol,
    write_train_view,
)
from slm_training.harnesses.model_build import ModelBuildConfig, build_model, train
from slm_training.harnesses.model_build.data import load_train_records
from slm_training.harnesses.model_build.eval_runner import evaluate_grammar_leakage_audit


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_sha(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _write_lock(*, output_dir: Path, protocol: Any) -> Path:
    campaign = build_campaign(protocol)
    lock = CampaignLockV1(
        manifest_sha256=campaign_manifest_sha256(campaign), manifest=campaign
    )
    path = output_dir / "campaign.lock.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("manifest_sha256") != lock.manifest_sha256:
            deviation = CampaignDeviationV1(
                campaign_id=campaign.campaign_id,
                experiment_id=campaign.experiment_id,
                manifest_sha256=str(previous.get("manifest_sha256")),
                changed_field="campaign_manifest_sha256",
                old_value=str(previous.get("manifest_sha256")),
                new_value=lock.manifest_sha256,
                reason="A local preflight outcome required an append-only protocol revision before further cells could run.",
                author="slm-training",
                outcome_accessed=True,
            )
            with (output_dir / "campaign.deviations.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(deviation.model_dump_json() + "\n")
    path.write_text(lock.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _write_locked_suite(
    *, locked_manifest: Path, output_dir: Path, limit: int
) -> tuple[Path, str, list[str]]:
    if limit < 1:
        raise ValueError("--locked-limit must be positive for a bounded local diagnostic")
    payload = json.loads(locked_manifest.read_text(encoding="utf-8"))
    rows = [row["record"] for row in payload["rows"] if row.get("partition") == "locked_test"]
    rows = rows[:limit]
    if not rows:
        raise ValueError("locked manifest has no selected rows")
    suite = "slm298_locked_diagnostic"
    suite_dir = output_dir / "locked_suite" / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    records_path = suite_dir / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = output_dir / "locked_suite" / "manifest.json"
    manifest.write_text(json.dumps({"suites": {suite: str(records_path)}}, indent=2) + "\n", encoding="utf-8")
    return manifest.parent, suite, [str(row["id"]) for row in rows]


def _config(
    *,
    train_dir: Path,
    output_dir: Path,
    cell: Any,
    target_token_budget: int,
    suite_dir: Path | None = None,
    suite: str = "smoke",
) -> ModelBuildConfig:
    return ModelBuildConfig(
        train_dir=train_dir,
        test_dir=suite_dir,
        suite=suite,
        run_root=output_dir / "cells" / cell.cell_id,
        run_id=cell.cell_id,
        seed=cell.seed,
        device="cpu",
        model_name="twotower",
        # The strict derived snapshot is admitted by the current symbol-only
        # contract; its broader document forms are encoded by the canonical
        # lexer path, not the narrower LangCore choice codec.
        output_tokenizer="lexer",
        d_model=cell.d_model,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        context_backend="scratch",
        freeze_context=cell.context == "scratch_frozen",
        local_files_only=True,
        optimizer_name="adamw",
        steps=10_000,
        batch_size=4,
        target_token_budget=target_token_budget,
        grammar_ltr_max_tokens=8,
        grammar_ltr_primary=False,
        gen_steps=1,
        run_class="scratch_matrix",
    )


def run_cell(*, output_dir: Path, train_dir: Path, locked_manifest: Path, cell_id: str, locked_limit: int) -> Path:
    protocol = load_protocol(train_dir=train_dir, locked_manifest=locked_manifest)
    lock_path = _write_lock(output_dir=output_dir, protocol=protocol)
    cell = next((item for item in protocol.cells if item.cell_id == cell_id), None)
    if cell is None:
        raise ValueError(f"unknown SLM-298 cell: {cell_id}")
    flat_dir = write_train_view(protocol, train_dir=train_dir, output_dir=output_dir, curriculum="flat")
    cell_dir = write_train_view(protocol, train_dir=train_dir, output_dir=output_dir, curriculum=cell.curriculum)
    suite_dir, suite, record_ids = _write_locked_suite(
        locked_manifest=locked_manifest, output_dir=output_dir, limit=locked_limit
    )
    # Build every curriculum arm from the same canonical vocabulary/order.  Training
    # alone sees its selected curriculum view, so paired initial hashes are meaningful.
    initial_records = load_train_records(flat_dir)
    config = _config(
        train_dir=cell_dir,
        output_dir=output_dir,
        cell=cell,
        target_token_budget=protocol.target_token_budget,
    )
    model = build_model(config, initial_records)
    initial = _state_sha(model)
    repeat_initial = _state_sha(build_model(config, initial_records))
    if initial != repeat_initial:
        raise ValueError("cell initialization is not bit exact")
    summary = train(config, model=model)
    checkpoint = Path(str(summary["checkpoint"]))
    eval_config = _config(
        train_dir=cell_dir,
        output_dir=output_dir,
        cell=cell,
        target_token_budget=protocol.target_token_budget,
        suite_dir=suite_dir,
        suite=suite,
    )
    eval_config = replace(eval_config, decode_timeout_seconds=5.0)
    audit = evaluate_grammar_leakage_audit(
        eval_config,
        model=build_model(eval_config, [], checkpoint=checkpoint),
        publish_agentv=True,
        variant_names=("constrained_native", "constrained_compiler"),
    )
    if any(int(values.get("decode_timeout_count") or 0) for values in audit["variants"].values()):
        raise TimeoutError("decode timeout makes the SLM-298 cell non-evidence")
    result = {
        "schema": "Slm298LocalCellResultV1",
        "cell": cell.to_dict(),
        "protocol_sha256": protocol.to_dict()["protocol_sha256"],
        "campaign_manifest_sha256": json.loads(lock_path.read_text(encoding="utf-8"))["manifest_sha256"],
        "train_manifest_sha256": protocol.train_manifest_sha256,
        "locked_eval_manifest_sha256": protocol.locked_eval_manifest_sha256,
        "locked_limit": locked_limit,
        "record_ids": record_ids,
        "initial_tensor_sha256": initial,
        "repeat_initial_tensor_sha256": repeat_initial,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha_file(checkpoint),
        "train_summary": summary,
        "grammar_audit": audit,
        "claim_class": "bounded_local_diagnostic",
        "human_rating_gate": "not_required",
    }
    path = output_dir / "cells" / f"{cell.cell_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "cell"), default="plan")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs/slm298_local_factorial"))
    parser.add_argument("--train-dir", type=Path, default=DERIVED_TRAIN_DIR)
    parser.add_argument("--locked-manifest", type=Path, default=LOCKED_MANIFEST)
    parser.add_argument("--cell")
    parser.add_argument("--locked-limit", type=int)
    args = parser.parse_args(argv)
    protocol = load_protocol(train_dir=args.train_dir, locked_manifest=args.locked_manifest)
    if args.mode == "plan":
        print(_write_lock(output_dir=args.output_dir, protocol=protocol))
        return 0
    if not args.cell or args.locked_limit is None:
        parser.error("cell mode requires --cell and --locked-limit")
    print(run_cell(output_dir=args.output_dir, train_dir=args.train_dir, locked_manifest=args.locked_manifest, cell_id=args.cell, locked_limit=args.locked_limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
