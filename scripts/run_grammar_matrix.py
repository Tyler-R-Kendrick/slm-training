#!/usr/bin/env python3
"""Run grammar topology diffusion experiments (X9-X15).

Staged ablations with 3 seeds and successive halving on smoke → held_out → adversarial.
See docs/design/quality-experiment-matrix.md (X matrix section).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from slm_training.harnesses.model_build import ModelBuildConfig, train
from slm_training.harnesses.model_build.eval_runner import evaluate_suites
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
)

HALVING_SUITES = ["smoke", "held_out", "adversarial"]
FULL_SUITES = ["smoke", "held_out", "adversarial", "ood", "rico_held"]
DEFAULT_SEEDS = [0, 1, 2]
LEGACY_FIXED_IDS = {"X2", "X3", "X4", "X5", "X7", "X8"}


def _args_with_seed(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    ns.seed = seed
    return ns


def _copy_checkpoint(src: Path, dest: Path) -> Path:
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    for side in (src.with_suffix(".tokenizer.json"), src.with_suffix(".meta.json")):
        if side.is_file():
            shutil.copy2(side, dest.parent / side.name)
    return dest


@dataclass(frozen=True)
class GrammarExperiment:
    xid: str
    run_id: str
    description: str
    train_dir: Path
    model_name: str = "twotower"
    # Shared recipe knobs
    fidelity_loss_weight: float = 0.0
    schema_in_context: bool = False
    slot_contract_in_context: bool = False
    slot_contract_constrained_decode: bool = False
    honest_slot_contract: bool = True
    grammar_ltr_repair: bool = True
    grammar_ltr_primary: bool = True
    ltr_loss_weight: float = 1.0
    fuse_ltr_loss: bool = True
    grammar_fastpath: bool = True
    grammar_fastpath_mode: str = "hybrid"
    fastpath_aux_weight: float = 0.0
    parallel_unmask: str = "adaptive"
    grammar_ltr_stages: tuple[int, ...] | None = None
    grammar_ltr_max_tokens: int = 96
    use_curriculum: bool = False
    mix_curriculum: bool = True
    d_model: int = 128
    n_heads: int = 4
    context_layers: int = 2
    denoiser_layers: int = 4
    design_md_in_context: bool = True
    best_of_n: int = 1
    preference: bool = False
    rl: bool = False
    skip_rl_no_variance: bool = False
    eval_from_run: str | None = None
    seed_checkpoint: str | None = None
    cache_context: bool = True
    use_compile: bool = False
    block_size: int = 4
    production_loss_weight: float = 1.0
    slot_loss_weight: float = 0.5
    confidence_loss_weight: float = 0.25
    topology_actions: bool = True
    topology_structural_embeddings: bool = True
    topology_heterogeneous_noise: bool = True
    topology_critic_decode: bool = True
    topology_bounded_buffer: bool = True


def _x_experiments(
    train_v1: Path,
    train_cur: Path,
    *,
    design_md_in_context: bool = True,
) -> list[GrammarExperiment]:
    """Runnable honest controls plus X9-X15 topology ablations."""
    base = dict(
        train_dir=train_v1,
        design_md_in_context=design_md_in_context,
        grammar_ltr_repair=True,
        grammar_ltr_primary=True,
        honest_slot_contract=True,
    )
    return [
        GrammarExperiment(
            "X0",
            "gx_x0_baseline",
            "Corrected baseline (twotower + honest DESIGN.md eval)",
            **base,
            model_name="twotower",
        ),
        GrammarExperiment(
            "X1",
            "gx_x1_contract",
            "Data/contract: slot inventory in context + constrained decode",
            **base,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            fidelity_loss_weight=1.0,
        ),
        GrammarExperiment(
            "X6",
            "gx_x6_curriculum",
            "Grammar curriculum: soft A/B/C mix (anti-leak)",
            train_cur,
            design_md_in_context=design_md_in_context,
            use_curriculum=True,
            mix_curriculum=True,
        ),
        GrammarExperiment(
            "X9",
            "gx_x9_topology_base",
            "Typed subtree collapse + synchronous expansion",
            **base,
            model_name="grammar_diffusion",
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            topology_actions=False,
            topology_structural_embeddings=False,
            topology_heterogeneous_noise=False,
            topology_critic_decode=False,
            topology_bounded_buffer=False,
        ),
        GrammarExperiment(
            "X10",
            "gx_x10_topology_actions",
            "X9 + insertion/deletion/contraction actions",
            **base,
            model_name="grammar_diffusion",
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            topology_structural_embeddings=False,
            topology_heterogeneous_noise=False,
            topology_critic_decode=False,
            topology_bounded_buffer=False,
        ),
        GrammarExperiment(
            "X11",
            "gx_x11_tree_embeddings",
            "X10 + tree structural embeddings",
            **base,
            model_name="grammar_diffusion",
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            topology_heterogeneous_noise=False,
            topology_critic_decode=False,
            topology_bounded_buffer=False,
        ),
        GrammarExperiment(
            "X12",
            "gx_x12_heterogeneous_noise",
            "X11 + heterogeneous node noise",
            **base,
            model_name="grammar_diffusion",
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            topology_critic_decode=False,
            topology_bounded_buffer=False,
        ),
        GrammarExperiment(
            "X13",
            "gx_x13_critic",
            "X12 + critic-guided accept/defer/contract",
            **base,
            model_name="grammar_diffusion",
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            topology_bounded_buffer=False,
        ),
        GrammarExperiment(
            "X14",
            "gx_x14_buffer",
            "X13 + bounded active buffer and global sync",
            **base,
            model_name="grammar_diffusion",
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
        ),
        GrammarExperiment(
            "X15",
            "gx_x15_topology_champion",
            "Full grammar-topology diffusion stack",
            train_cur,
            model_name="grammar_diffusion",
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            use_curriculum=True,
            mix_curriculum=True,
            design_md_in_context=design_md_in_context,
            d_model=192,
            n_heads=6,
            context_layers=3,
            denoiser_layers=6,
        ),
    ]


def _train_cfg(exp: GrammarExperiment, args: argparse.Namespace) -> ModelBuildConfig:
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
        model_name=exp.model_name,
        d_model=exp.d_model,
        n_heads=exp.n_heads,
        context_layers=exp.context_layers,
        denoiser_layers=exp.denoiser_layers,
        context_backend=args.context_backend,
        local_files_only=args.local_files_only,
        grammar_constrained=True,
        grammar_ltr_primary=exp.grammar_ltr_primary,
        grammar_ltr_repair=exp.grammar_ltr_repair,
        grammar_ltr_max_tokens=exp.grammar_ltr_max_tokens,
        grammar_ltr_stages=exp.grammar_ltr_stages,
        design_md_in_context=exp.design_md_in_context,
        ltr_loss_weight=exp.ltr_loss_weight,
        fidelity_loss_weight=exp.fidelity_loss_weight,
        schema_in_context=exp.schema_in_context,
        slot_contract_in_context=exp.slot_contract_in_context,
        slot_contract_constrained_decode=exp.slot_contract_constrained_decode,
        honest_slot_contract=bool(getattr(exp, "honest_slot_contract", True)),
        best_of_n=1,
        use_curriculum=exp.use_curriculum,
        mix_curriculum=exp.mix_curriculum,
        use_compile=bool(exp.use_compile or getattr(args, "compile", False)),
        parallel_unmask=exp.parallel_unmask,
        grad_accum_steps=max(1, int(getattr(args, "grad_accum", 1) or 1)),
        cache_context=exp.cache_context,
        fuse_ltr_loss=exp.fuse_ltr_loss,
        grammar_fastpath=exp.grammar_fastpath,
        grammar_fastpath_mode=exp.grammar_fastpath_mode,
        fastpath_aux_weight=exp.fastpath_aux_weight,
        block_size=exp.block_size,
        production_loss_weight=exp.production_loss_weight,
        slot_loss_weight=exp.slot_loss_weight,
        confidence_loss_weight=exp.confidence_loss_weight,
        topology_actions=exp.topology_actions,
        topology_structural_embeddings=exp.topology_structural_embeddings,
        topology_heterogeneous_noise=exp.topology_heterogeneous_noise,
        topology_critic_decode=exp.topology_critic_decode,
        topology_bounded_buffer=exp.topology_bounded_buffer,
        eval_every=args.eval_every,
        eval_suite="smoke",
        eval_suites="smoke,held_out" if args.eval_every else "",
        structural_bias=2.5,
        telemetry=True,
    )


def _eval_cfg(exp: GrammarExperiment, args: argparse.Namespace) -> ModelBuildConfig:
    cfg = _train_cfg(exp, args)
    return replace(
        cfg,
        best_of_n=exp.best_of_n,
        gen_steps=args.gen_steps,
        grammar_ltr_repair=True,
        rico_eval_limit=args.rico_limit,
        run_id=exp.run_id,
    )


def _halving_score(suites: dict[str, Any], suite: str) -> float:
    m = suites.get(suite) or {}
    topology = m.get("topology_composite")
    if topology is not None:
        return float(topology)
    return (
        2.0 * float(m.get("parse_rate") or 0.0)
        + 2.0 * float(m.get("placeholder_fidelity") or 0.0)
        + 1.0 * float(m.get("structural_similarity") or 0.0)
        + 0.5 * float(m.get("reward_score") or 0.0)
    )


def _summarize_board(board: dict[str, Any]) -> dict[str, Any]:
    suites = board.get("suites") or {}
    gates = evaluate_ship_gates(suites)
    slim = {
        name: {
            "parse_rate": m.get("parse_rate"),
            "placeholder_fidelity": m.get("placeholder_fidelity"),
            "structural_similarity": m.get("structural_similarity"),
            "reward_score": m.get("reward_score"),
            "ast_node_f1": m.get("ast_node_f1"),
            "ast_edge_f1": m.get("ast_edge_f1"),
            "tree_edit_similarity": m.get("tree_edit_similarity"),
            "topology_quality_score": m.get("topology_quality_score"),
            "topology_structure_score": m.get("topology_structure_score"),
            "topology_trace_score": m.get("topology_trace_score"),
            "topology_efficiency_score": m.get("topology_efficiency_score"),
            "topology_composite": m.get("topology_composite"),
            "topology_telemetry": m.get("topology_telemetry"),
            "n": m.get("n"),
        }
        for name, m in suites.items()
    }
    return {
        "pass": gates.get("pass"),
        "failures": gates.get("failures"),
        "suites": slim,
    }


def _maybe_preference(
    exp: GrammarExperiment, ckpt: Path, args: argparse.Namespace
) -> Path:
    if not exp.preference:
        return ckpt
    from slm_training.dsl.schema import load_jsonl
    from slm_training.harnesses.preference import (
        collect_pairs_with_generator,
        write_pairs,
    )
    from slm_training.harnesses.preference.train import train_preference_from_paths
    from slm_training.harnesses.quality import soft_corrupt_openui

    pairs_path = args.run_root / exp.run_id / f"pairs_s{args.seed}.jsonl"
    records = load_jsonl(exp.train_dir / "records.jsonl")[: args.pref_limit]
    pairs = collect_pairs_with_generator(
        records,
        lambda r: [r.openui, soft_corrupt_openui(r.openui)],
        prefer_valid_rejects=True,
        structure_only=True,
    )
    write_pairs(pairs_path, pairs)
    out_dir = args.run_root / exp.run_id / f"pref_s{args.seed}"
    summary = train_preference_from_paths(
        ckpt,
        pairs_path,
        out_dir=out_dir,
        steps=args.pref_steps,
        device=args.device,
    )
    pref_ckpt = Path(summary.get("checkpoint") or (out_dir / "checkpoints" / "last.pt"))
    if pref_ckpt.is_file():
        dest = args.run_root / exp.run_id / f"checkpoints_s{args.seed}" / "last.pt"
        return _copy_checkpoint(pref_ckpt, dest)
    return ckpt


def _maybe_rl(exp: GrammarExperiment, ckpt: Path, args: argparse.Namespace) -> Path:
    if not exp.rl:
        return ckpt
    from slm_training.harnesses.rl import train_grpo_from_paths
    from slm_training.autoresearch.rl_gate import assert_rl_ready

    readiness = assert_rl_ready(getattr(args, "rl_readiness_report", None))

    out_dir = args.run_root / exp.run_id / f"rl_s{args.seed}"
    summary = train_grpo_from_paths(
        ckpt,
        exp.train_dir / "records.jsonl",
        out_dir=out_dir,
        steps=max(10, int(getattr(args, "rl_steps", 30) or 30)),
        group_size=max(2, int(getattr(args, "rl_group_size", 4) or 4)),
        device=args.device,
        ref_checkpoint=ckpt,
        limit=int(getattr(args, "pref_limit", 32) or 32),
        kl_beta=0.05,
        readiness_report=readiness,
    )
    skipped = int(summary.get("skipped_groups") or 0)
    rl_ckpt = Path(summary.get("checkpoint") or (out_dir / "model.pt"))
    if (
        exp.skip_rl_no_variance
        and skipped > 0
        and summary.get("last_reward_mean", 0) <= 0
    ):
        return ckpt
    if rl_ckpt.is_file():
        dest = args.run_root / exp.run_id / f"checkpoints_s{args.seed}" / "last.pt"
        return _copy_checkpoint(rl_ckpt, dest)
    return ckpt


def run_one(
    exp: GrammarExperiment,
    args: argparse.Namespace,
    *,
    suites: list[str] | None = None,
    skip_train: bool = False,
) -> dict[str, Any]:
    run_id = f"{exp.run_id}_s{args.seed}"
    run_dir = args.run_root / exp.run_id
    ckpt_dir = run_dir / f"checkpoints_s{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "last.pt"

    if skip_train and ckpt.is_file():
        pass
    elif not skip_train and exp.seed_checkpoint:
        src = Path(exp.seed_checkpoint)
        dest = ckpt_dir / "last.pt"
        if not src.is_file():
            raise FileNotFoundError(f"{exp.xid} needs seed checkpoint {src}")
        ckpt = _copy_checkpoint(src, dest)
    elif not skip_train and exp.eval_from_run and not exp.preference:
        src = (
            args.run_root / exp.eval_from_run / f"checkpoints_s{args.seed}" / "last.pt"
        )
        if not src.is_file():
            src = args.run_root / exp.eval_from_run / "checkpoints" / "last.pt"
        dest = ckpt_dir / "last.pt"
        if not src.is_file():
            raise FileNotFoundError(f"{exp.xid} needs {src}")
        ckpt = _copy_checkpoint(src, dest)
    elif not skip_train:
        train_cfg = _train_cfg(exp, args)
        train_cfg = replace(train_cfg, run_id=run_id)
        if exp.eval_from_run and exp.preference:
            src = (
                args.run_root
                / exp.eval_from_run
                / f"checkpoints_s{args.seed}"
                / "last.pt"
            )
            if not src.is_file():
                src = args.run_root / exp.eval_from_run / "checkpoints" / "last.pt"
            dest = ckpt_dir / "last.pt"
            ckpt = _copy_checkpoint(src, dest)
        else:
            summary = train(train_cfg)
            ckpt = Path(summary["checkpoint"])
            _copy_checkpoint(ckpt, ckpt_dir / "last.pt")
            ckpt = ckpt_dir / "last.pt"
    elif not ckpt.is_file():
        raise FileNotFoundError(
            f"{exp.xid} seed {args.seed}: missing checkpoint for skip_train"
        )

    ckpt = _maybe_preference(exp, ckpt, args)
    ckpt = _maybe_rl(exp, ckpt, args)
    eval_cfg = _eval_cfg(exp, args)
    eval_cfg = replace(eval_cfg, run_id=run_id)
    eval_suites = suites or FULL_SUITES
    board = evaluate_suites(
        eval_cfg,
        eval_suites,
        checkpoint=ckpt,
        write_gates=eval_suites == FULL_SUITES,
    )
    result = {
        "id": exp.xid,
        "run_id": exp.run_id,
        "seed": args.seed,
        "model_name": exp.model_name,
        "description": exp.description,
        "checkpoint": str(ckpt),
        **_summarize_board(board),
    }
    out_path = run_dir / f"matrix_result_s{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def successive_halving(
    candidates: list[tuple[GrammarExperiment, int]],
    args: argparse.Namespace,
) -> tuple[list[tuple[GrammarExperiment, int]], list[dict[str, Any]]]:
    """Train all candidates once; prune by smoke → held_out → adversarial scores."""
    survivors = list(candidates)
    all_results: list[dict[str, Any]] = []
    trained: dict[tuple[str, int], dict[str, Any]] = {}

    for suite in HALVING_SUITES:
        round_results: list[tuple[tuple[GrammarExperiment, int], float]] = []
        for exp, seed in survivors:
            key = (exp.xid, seed)
            ns = _args_with_seed(args, seed)
            if key not in trained:
                trained[key] = run_one(
                    exp,
                    ns,
                    suites=[suite],
                    skip_train=args.reuse_checkpoints,
                )
            else:
                eval_cfg = _eval_cfg(exp, ns)
                eval_cfg = replace(eval_cfg, run_id=f"{exp.run_id}_s{seed}")
                ckpt = Path(trained[key]["checkpoint"])
                board = evaluate_suites(
                    eval_cfg,
                    [suite],
                    checkpoint=ckpt,
                    write_gates=False,
                )
                summary = _summarize_board(board)
                prev_suites = dict(trained[key].get("suites") or {})
                merged_suites = {**prev_suites, **(summary.get("suites") or {})}
                trained[key] = {
                    **trained[key],
                    **summary,
                    "suites": merged_suites,
                }
            score = _halving_score(trained[key].get("suites") or {}, suite)
            round_results.append(((exp, seed), score))

        grouped: dict[str, list[float]] = {}
        experiments_by_id: dict[str, GrammarExperiment] = {}
        for (experiment, _seed), score in round_results:
            grouped.setdefault(experiment.xid, []).append(score)
            experiments_by_id[experiment.xid] = experiment
        experiment_scores = sorted(
            (
                (experiments_by_id[xid], statistics.median(scores))
                for xid, scores in grouped.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        keep = max(
            1,
            min(
                len(experiment_scores),
                max(2, math.ceil(len(experiment_scores) / 2)),
            ),
        )
        kept_ids = {experiment.xid for experiment, _ in experiment_scores[:keep]}
        survivors = [pair for pair in survivors if pair[0].xid in kept_ids]
        print(
            json.dumps(
                {
                    "halving_round": suite,
                    "kept": keep,
                    "total": len(experiment_scores),
                    "top": [
                        {"id": experiment.xid, "median_score": score}
                        for experiment, score in experiment_scores[:keep]
                    ],
                }
            )
        )

    for exp, seed in survivors:
        key = (exp.xid, seed)
        if key in trained:
            ns = _args_with_seed(args, seed)
            eval_cfg = _eval_cfg(exp, ns)
            eval_cfg = replace(eval_cfg, run_id=f"{exp.run_id}_s{seed}")
            ckpt = Path(trained[key]["checkpoint"])
            board = evaluate_suites(
                eval_cfg,
                FULL_SUITES,
                checkpoint=ckpt,
                write_gates=True,
            )
            final = {
                **trained[key],
                **_summarize_board(board),
            }
            trained[key] = final
            all_results.append(final)

    # Include non-survivor results for the summary artifact.
    for key, result in trained.items():
        if not any(
            r.get("id") == key[0] and r.get("seed") == key[1] for r in all_results
        ):
            all_results.append(result)

    return survivors, all_results


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
    parser.add_argument(
        "--context-backend", choices=("scratch", "hf"), default="scratch"
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0, help="Single-seed override.")
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seeds (default 0,1,2).",
    )
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--rico-limit", type=int, default=32)
    parser.add_argument("--pref-steps", type=int, default=30)
    parser.add_argument("--pref-limit", type=int, default=40)
    parser.add_argument("--rl-steps", type=int, default=30)
    parser.add_argument("--rl-group-size", type=int, default=4)
    parser.add_argument("--rl-readiness-report", type=Path, default=None)
    parser.add_argument("--gen-steps", type=int, default=8)
    parser.add_argument(
        "--confirm-steps",
        type=int,
        default=0,
        help="After halving, retrain the surviving experiment rows at this step count.",
    )
    parser.add_argument(
        "--confirm-top",
        type=int,
        default=2,
        help="Number of surviving experiment rows to confirm (default 2).",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated experiment ids (e.g. X0,X7).",
    )
    parser.add_argument(
        "--no-halving",
        action="store_true",
        help="Run all experiment×seed combos without successive halving.",
    )
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Resume halving from existing per-seed checkpoints without retraining.",
    )
    parser.add_argument(
        "--training-source-commit",
        default=None,
        help="Commit that produced reused checkpoints (recorded separately from eval).",
    )
    parser.add_argument(
        "--no-design-md-context",
        action="store_true",
        help="Train/eval without DESIGN.md in context.",
    )
    parser.add_argument(
        "--build-curriculum",
        action="store_true",
        help="Build curriculum train corpus before running.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile on train experiments.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
    )
    args = parser.parse_args(argv)

    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seed_list:
        seed_list = list(DEFAULT_SEEDS)

    needs_curriculum = args.only is None or any(
        x in (args.only or "") for x in ("X6", "X15")
    )
    if args.build_curriculum or (not args.curriculum_dir.exists() and needs_curriculum):
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

    design_md = not args.no_design_md_context
    experiments = _x_experiments(
        args.train_dir,
        args.curriculum_dir,
        design_md_in_context=design_md,
    )
    if args.only:
        wanted = {x.strip().upper() for x in args.only.split(",") if x.strip()}
        legacy = sorted(wanted & LEGACY_FIXED_IDS)
        if legacy:
            raise ValueError(
                f"{legacy} are frozen fixed-canvas rows; rerun them from the "
                "source_commit recorded in docs/design/grammar-matrix-results.json"
            )
        experiments = [e for e in experiments if e.xid in wanted]

    candidates = [(exp, seed) for exp in experiments for seed in seed_list]

    if args.no_halving:
        results: list[dict[str, Any]] = []
        for exp, seed in candidates:
            ns = _args_with_seed(args, seed)
            print(json.dumps({"status": "start", "id": exp.xid, "seed": seed}))
            result = run_one(exp, ns)
            print(
                json.dumps(
                    {
                        "status": "done",
                        "id": exp.xid,
                        "seed": seed,
                        "pass": result["pass"],
                    }
                )
            )
            results.append(result)
        survivors = candidates
    else:
        survivors, results = successive_halving(candidates, args)
        if args.confirm_steps > 0:
            survivor_ids = list(dict.fromkeys(exp.xid for exp, _seed in survivors))[
                : max(1, args.confirm_top)
            ]
            for exp, seed in survivors:
                if exp.xid not in survivor_ids:
                    continue
                confirmed = replace(
                    exp,
                    run_id=f"{exp.run_id}_confirm_{args.confirm_steps}",
                )
                ns = _args_with_seed(args, seed)
                ns.steps = args.confirm_steps
                result = run_one(confirmed, ns)
                result["stage"] = "confirmation"
                results.append(result)

    results.sort(key=lambda r: (r.get("id") or "", r.get("seed") or 0))

    evaluation_source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    out = {
        "matrix": "grammar-experiment-matrix-x",
        "reference": "docs/design/quality-experiment-matrix.md",
        "gate_policy": {k: v for k, v in DEFAULT_SHIP_GATES.items()},
        "halving_suites": HALVING_SUITES,
        "seeds": seed_list,
        "rico_eval_limit": args.rico_limit,
        "steps": args.steps,
        "gen_steps": args.gen_steps,
        "confirmation_steps": args.confirm_steps,
        "context_backend": args.context_backend,
        "halving_enabled": not args.no_halving,
        "survivors": [{"id": e.xid, "seed": s} for e, s in survivors],
        "results": results,
        "source_commit": args.training_source_commit or evaluation_source_commit,
        "training_source_commit": args.training_source_commit
        or evaluation_source_commit,
        "evaluation_source_commit": evaluation_source_commit,
    }
    out_path = args.run_root / "grammar_matrix_summary.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    docs_out = Path("docs/design/grammar-matrix-results.json")
    if docs_out.exists():
        prior = json.loads(docs_out.read_text(encoding="utf-8"))
        legacy = prior.get("legacy_fixed_canvas")
        if legacy is None and any(
            result.get("id") in LEGACY_FIXED_IDS for result in prior.get("results", [])
        ):
            legacy = prior
        if legacy is not None:
            out["legacy_fixed_canvas"] = legacy
    docs_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"summary": str(out_path), "n": len(results), "survivors": len(survivors)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
