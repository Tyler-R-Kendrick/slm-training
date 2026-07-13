#!/usr/bin/env python3
"""Run the quality experiment matrix (docs/design/quality-experiment-matrix.md)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from slm_training.harnesses.model_build import ModelBuildConfig, train
from slm_training.harnesses.model_build.eval_runner import evaluate_suites
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
    write_ship_gates,
)


SUITES = ["smoke", "held_out", "adversarial", "ood", "rico_held"]


def _copy_checkpoint(src: Path, dest: Path) -> Path:
    """Copy checkpoint + tokenizer/meta sidecars used by TwoTower.from_checkpoint."""
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    for side in (src.with_suffix(".tokenizer.json"), src.with_suffix(".meta.json")):
        if side.is_file():
            shutil.copy2(side, dest.with_suffix(side.suffix))
    return dest


@dataclass(frozen=True)
class Experiment:
    eid: str
    run_id: str
    description: str
    train_dir: Path
    # Train overrides
    fidelity_loss_weight: float = 0.0
    schema_in_context: bool = False
    retrieval_k: int = 0
    best_of_n: int = 1
    use_curriculum: bool = False
    grammar_ltr_repair: bool = False
    d_model: int = 128
    n_heads: int = 4
    context_layers: int = 2
    denoiser_layers: int = 4
    grammar_ltr_max_tokens: int = 64
    # Eval-only overlay on a prior run (skip train if set)
    eval_from_run: str | None = None
    preference: bool = False


def _base_experiments(train_v1: Path, train_cur: Path) -> list[Experiment]:
    return [
        Experiment(
            "E0",
            "qx_e0_baseline",
            "Ship-recipe baseline (v1, LTR primary, scratch)",
            train_v1,
        ),
        Experiment(
            "E1",
            "qx_e1_repair",
            "Constrained LTR repair at eval (decode lever on E0)",
            train_v1,
            grammar_ltr_repair=True,
            eval_from_run="qx_e0_baseline",
        ),
        Experiment(
            "E2",
            "qx_e2_curriculum",
            "Curriculum A→B→C sampling",
            train_cur,
            use_curriculum=True,
        ),
        Experiment(
            "E3",
            "qx_e3_fidelity",
            "Fidelity aux loss on placeholder tokens",
            train_v1,
            fidelity_loss_weight=1.0,
        ),
        Experiment(
            "E4",
            "qx_e4_schema",
            "Schema-conditioned context",
            train_v1,
            schema_in_context=True,
        ),
        Experiment(
            "E5",
            "qx_e5_pref_bon",
            "Soft preference pairs + best-of-N decode",
            train_v1,
            best_of_n=4,
            preference=True,
            eval_from_run="qx_e0_baseline",
        ),
        Experiment(
            "E6",
            "qx_e6_retrieve",
            "Retrieval skeleton bank (k=1)",
            train_v1,
            retrieval_k=1,
        ),
        Experiment(
            "E7",
            "qx_e7_capacity",
            "Capacity upgrade (wider/deeper + longer LTR)",
            train_v1,
            d_model=192,
            n_heads=6,
            context_layers=3,
            denoiser_layers=6,
            grammar_ltr_max_tokens=96,
        ),
        Experiment(
            "E8",
            "qx_e8_combo",
            "Stacked combo of all levers",
            train_cur,
            fidelity_loss_weight=1.0,
            schema_in_context=True,
            retrieval_k=1,
            best_of_n=4,
            use_curriculum=True,
            grammar_ltr_repair=True,
            d_model=192,
            n_heads=6,
            context_layers=3,
            denoiser_layers=6,
            grammar_ltr_max_tokens=96,
            preference=True,
        ),
    ]


def _train_cfg(exp: Experiment, args: argparse.Namespace) -> ModelBuildConfig:
    return ModelBuildConfig(
        train_dir=exp.train_dir,
        test_dir=args.test_dir,
        suite="smoke",
        run_root=args.run_root,
        run_id=exp.run_id,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        model_name="twotower",
        d_model=exp.d_model,
        n_heads=exp.n_heads,
        context_layers=exp.context_layers,
        denoiser_layers=exp.denoiser_layers,
        context_backend=args.context_backend,
        local_files_only=args.local_files_only,
        grammar_constrained=True,
        grammar_ltr_primary=True,
        grammar_ltr_repair=exp.grammar_ltr_repair,
        grammar_ltr_max_tokens=exp.grammar_ltr_max_tokens,
        design_md_in_context=True,
        ltr_loss_weight=1.0,
        fidelity_loss_weight=exp.fidelity_loss_weight,
        schema_in_context=exp.schema_in_context,
        retrieval_k=exp.retrieval_k,
        best_of_n=1,  # train without BoN cost; apply at eval
        use_curriculum=exp.use_curriculum,
        eval_every=args.eval_every,
        eval_suite="smoke",
        structural_bias=2.5,
    )


def _eval_cfg(exp: Experiment, args: argparse.Namespace) -> ModelBuildConfig:
    cfg = _train_cfg(exp, args)
    return replace(
        cfg,
        best_of_n=exp.best_of_n,
        grammar_ltr_repair=exp.grammar_ltr_repair or exp.eid in {"E1", "E8"},
        rico_eval_limit=args.rico_limit,
        run_id=exp.run_id,
    )


def _maybe_preference(exp: Experiment, ckpt: Path, args: argparse.Namespace) -> Path:
    if not exp.preference:
        return ckpt
    from slm_training.dsl.schema import load_jsonl
    from slm_training.preference import collect_pairs_with_generator, write_pairs
    from slm_training.preference.train import train_preference_from_paths
    from slm_training.quality import soft_corrupt_openui

    pairs_path = args.run_root / exp.run_id / "pairs.jsonl"
    records = load_jsonl(exp.train_dir / "records.jsonl")[: args.pref_limit]
    pairs = collect_pairs_with_generator(
        records,
        lambda r: [r.openui, soft_corrupt_openui(r.openui)],
    )
    write_pairs(pairs_path, pairs)
    out_dir = args.run_root / exp.run_id / "pref"
    summary = train_preference_from_paths(
        ckpt,
        pairs_path,
        out_dir=out_dir,
        steps=args.pref_steps,
        device=args.device,
    )
    pref_ckpt = Path(summary.get("checkpoint") or (out_dir / "checkpoints" / "last.pt"))
    if pref_ckpt.is_file():
        dest = args.run_root / exp.run_id / "checkpoints" / "last.pt"
        return _copy_checkpoint(pref_ckpt, dest)
    return ckpt


def _summarize_board(board: dict[str, Any]) -> dict[str, Any]:
    suites = board.get("suites") or {}
    gates = evaluate_ship_gates(suites)
    slim = {
        name: {
            "parse_rate": m.get("parse_rate"),
            "placeholder_fidelity": m.get("placeholder_fidelity"),
            "structural_similarity": m.get("structural_similarity"),
            "reward_score": m.get("reward_score"),
            "n": m.get("n"),
        }
        for name, m in suites.items()
    }
    return {
        "pass": gates.get("pass"),
        "failures": gates.get("failures"),
        "suites": slim,
    }


def run_one(exp: Experiment, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_root / exp.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if exp.eval_from_run and not exp.preference:
        src = args.run_root / exp.eval_from_run / "checkpoints" / "last.pt"
        dest = run_dir / "checkpoints" / "last.pt"
        if not src.is_file():
            raise FileNotFoundError(f"{exp.eid} needs {src}")
        ckpt = _copy_checkpoint(src, dest)
    else:
        train_cfg = _train_cfg(exp, args)
        if exp.eval_from_run and exp.preference:
            # Start from prior checkpoint weights, then preference-train.
            src = args.run_root / exp.eval_from_run / "checkpoints" / "last.pt"
            dest = run_dir / "checkpoints" / "last.pt"
            ckpt = _copy_checkpoint(src, dest)
        else:
            summary = train(train_cfg)
            ckpt = Path(summary["checkpoint"])

    ckpt = _maybe_preference(exp, ckpt, args)
    eval_cfg = _eval_cfg(exp, args)
    board = evaluate_suites(
        eval_cfg,
        SUITES,
        checkpoint=ckpt,
        write_gates=True,
    )
    result = {
        "id": exp.eid,
        "run_id": exp.run_id,
        "description": exp.description,
        "checkpoint": str(ckpt),
        **_summarize_board(board),
    }
    (run_dir / "matrix_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("outputs/train_data/v1"))
    parser.add_argument(
        "--curriculum-dir",
        type=Path,
        default=Path("outputs/train_data/v1_curriculum"),
    )
    parser.add_argument("--test-dir", type=Path, default=Path("outputs/test_data/v1"))
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context-backend", choices=("scratch", "hf"), default="scratch")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--rico-limit", type=int, default=32)
    parser.add_argument("--pref-steps", type=int, default=30)
    parser.add_argument("--pref-limit", type=int, default=40)
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated experiment ids (e.g. E0,E1,E8).",
    )
    parser.add_argument(
        "--build-curriculum",
        action="store_true",
        help="Build curriculum train corpus before running.",
    )
    args = parser.parse_args(argv)

    if args.build_curriculum or (
        not args.curriculum_dir.exists()
        and (args.only is None or any(x in (args.only or "") for x in ("E2", "E8")))
    ):
        from slm_training.harnesses.train_data import TrainDataConfig, build_train_data

        build_train_data(
            TrainDataConfig(
                source="all",
                output_root=args.curriculum_dir.parent,
                version=args.curriculum_dir.name,
                synthesizer="quality",
                curriculum=True,
            )
        )

    experiments = _base_experiments(args.train_dir, args.curriculum_dir)
    if args.only:
        wanted = {x.strip().upper() for x in args.only.split(",") if x.strip()}
        experiments = [e for e in experiments if e.eid in wanted]

    # Ensure E0 runs before dependents when selected together.
    order = {e.eid: i for i, e in enumerate(experiments)}
    experiments = sorted(
        experiments,
        key=lambda e: (
            0 if e.eid == "E0" else 1 if e.eval_from_run else 2,
            order.get(e.eid, 99),
        ),
    )

    results: list[dict[str, Any]] = []
    for exp in experiments:
        print(json.dumps({"status": "start", "id": exp.eid, "run_id": exp.run_id}))
        results.append(run_one(exp, args))
        print(json.dumps({"status": "done", "id": exp.eid, "pass": results[-1]["pass"]}))

    out = {
        "matrix": "quality-experiment-matrix",
        "reference": "docs/design/quality-experiment-matrix.md",
        "gate_policy": {k: v for k, v in DEFAULT_SHIP_GATES.items()},
        "rico_eval_limit": args.rico_limit,
        "steps": args.steps,
        "context_backend": args.context_backend,
        "results": results,
    }
    out_path = args.run_root / "quality_matrix_summary.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    # Also mirror under docs artifacts path for the PR.
    docs_out = Path("docs/design/quality-matrix-results.json")
    docs_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out_path), "n": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
