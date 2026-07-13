#!/usr/bin/env python3
"""
Run Phases A→B→C to completion (scratch CPU path).

A: fidelity anti-leak SFT (soft curriculum + fidelity aux + schema + LTR repair)
B: soft-corrupt preference pairs + DPO surrogate
C: GRPO-lite online RL
Then evaluate honest multi-suite ship gates + write telemetry.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.model_build import ModelBuildConfig, train
from slm_training.harnesses.model_build.eval_runner import evaluate_suites
from slm_training.harnesses.model_build.ship_gates import evaluate_ship_gates
from slm_training.preference import collect_pairs_with_generator, write_pairs
from slm_training.preference.train import train_preference_from_paths
from slm_training.quality import soft_corrupt_openui
from slm_training.rl import train_grpo_from_paths
from slm_training.dsl.schema import load_jsonl


def _copy_ckpt(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = Path(src).resolve()
    dest = Path(dest).resolve()
    if src != dest:
        shutil.copy2(src, dest)
    for side in (src.with_suffix(".tokenizer.json"), src.with_suffix(".meta.json")):
        if not side.is_file():
            continue
        target = dest.parent / side.name
        # When dest is last.pt, sidecars already live beside it.
        if side.resolve() == target.resolve():
            continue
        # Prefer dest-stem sidecars (sft.tokenizer.json) when dest != last.pt
        stem_target = dest.with_suffix(side.suffix)
        if side.resolve() != stem_target.resolve():
            shutil.copy2(side, stem_target)
        if side.name.startswith("last.") and dest.name != "last.pt":
            # Also keep last.* next to dest for from_checkpoint loaders that look beside dest.
            pass
    # TwoTower.from_checkpoint expects dest.with_suffix('.tokenizer.json')
    tok_src = src.with_suffix(".tokenizer.json")
    meta_src = src.with_suffix(".meta.json")
    if tok_src.is_file():
        tok_dst = dest.with_suffix(".tokenizer.json")
        if tok_src.resolve() != tok_dst.resolve():
            shutil.copy2(tok_src, tok_dst)
    if meta_src.is_file():
        meta_dst = dest.with_suffix(".meta.json")
        if meta_src.resolve() != meta_dst.resolve():
            shutil.copy2(meta_src, meta_dst)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("outputs/train_data/v1_curriculum"),
    )
    parser.add_argument("--test-dir", type=Path, default=Path("outputs/test_data/v1"))
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--run-id", default="phase_abc_complete")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--context-backend",
        choices=("scratch", "hf"),
        default="scratch",
        help="HF preferred when cached; default scratch for offline CPU.",
    )
    parser.add_argument("--sft-steps", type=int, default=250)
    parser.add_argument("--pref-steps", type=int, default=20)
    parser.add_argument("--rl-steps", type=int, default=15)
    parser.add_argument("--rl-group-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rico-limit", type=int, default=32)
    parser.add_argument("--pref-limit", type=int, default=48)
    parser.add_argument("--rl-limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Reuse existing SFT checkpoint under this run-id.",
    )
    parser.add_argument(
        "--seed-checkpoint",
        type=Path,
        default=None,
        help="Optional warm-start SFT checkpoint (skips random init).",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_root / args.run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    phases: dict = {"started_at": datetime.now(timezone.utc).isoformat()}

    # --- Phase A: SFT ---
    sft_ckpt = ckpt_dir / "sft.pt"
    if args.skip_sft and sft_ckpt.is_file():
        phases["A"] = {
            "skipped": True,
            "checkpoint": str(sft_ckpt),
            "note": "reused existing sft.pt",
        }
    elif args.skip_sft and (ckpt_dir / "last.pt").is_file() and not sft_ckpt.is_file():
        sft_ckpt = _copy_ckpt(ckpt_dir / "last.pt", sft_ckpt)
        phases["A"] = {
            "skipped": True,
            "checkpoint": str(sft_ckpt),
            "note": "bootstrapped sft.pt from last.pt",
        }
    else:
        if args.seed_checkpoint and args.seed_checkpoint.is_file():
            _copy_ckpt(args.seed_checkpoint, ckpt_dir / "last.pt")
        cfg = ModelBuildConfig(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            suite="smoke",
            run_root=args.run_root,
            run_id=args.run_id,
            steps=args.sft_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            model_name="twotower",
            d_model=192,
            n_heads=6,
            context_layers=3,
            denoiser_layers=6,
            context_backend=args.context_backend,
            local_files_only=True,
            grammar_constrained=True,
            grammar_ltr_primary=True,
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=96,
            design_md_in_context=True,
            ltr_loss_weight=1.0,
            fidelity_loss_weight=1.5,
            schema_in_context=True,
            use_curriculum=True,
            mix_curriculum=True,
            cache_context=True,
            fuse_ltr_loss=True,
            structural_bias=2.5,
            eval_every=max(50, args.sft_steps // 5),
            eval_suite="smoke",
            eval_suites="smoke,held_out,adversarial",
            rico_eval_limit=args.rico_limit,
            telemetry=True,
            best_of_n=1,
        )
        if args.seed_checkpoint and args.seed_checkpoint.is_file():
            from slm_training.harnesses.model_build.factory import build_model
            from slm_training.harnesses.model_build.data import load_train_records

            records = load_train_records(args.train_dir)
            model = build_model(cfg, records, checkpoint=args.seed_checkpoint)
            summary = train(cfg, model=model)
        else:
            summary = train(cfg)
        src = Path(summary["checkpoint"])
        sft_ckpt = _copy_ckpt(src, sft_ckpt)
        _copy_ckpt(src, ckpt_dir / "last.pt")
        phases["A"] = {
            "steps": summary.get("steps"),
            "last_loss": summary.get("last_loss"),
            "checkpoint": str(sft_ckpt),
            "telemetry": summary.get("telemetry", {}).get("bottlenecks"),
            "final_eval": summary.get("final_eval"),
        }

    # --- Phase B: preference (low LR surrogate; may not beat SFT) ---
    if not sft_ckpt.is_file() and (ckpt_dir / "last.pt").is_file():
        sft_ckpt = _copy_ckpt(ckpt_dir / "last.pt", sft_ckpt)
    records = load_jsonl(args.train_dir / "records.jsonl")[: args.pref_limit]
    pairs_path = run_dir / "pairs.jsonl"
    pairs = collect_pairs_with_generator(
        records,
        lambda r: [r.openui, soft_corrupt_openui(r.openui)],
        prefer_valid_rejects=True,
        structure_only=True,
    )
    write_pairs(pairs_path, pairs)
    pref_dir = run_dir / "pref"
    pref_summary = train_preference_from_paths(
        sft_ckpt,
        pairs_path,
        out_dir=pref_dir,
        steps=args.pref_steps,
        device=args.device,
    )
    pref_ckpt = Path(pref_summary["checkpoint"])
    phases["B"] = {
        "pairs": len(pairs),
        "steps": pref_summary.get("steps"),
        "last_loss": pref_summary.get("last_loss"),
        "checkpoint": str(pref_ckpt),
    }

    # --- Phase C: GRPO (conservative; restores best-reward weights) ---
    rl_dir = run_dir / "rl"
    rl_summary = train_grpo_from_paths(
        sft_ckpt,  # start from SFT (pref often hurts); KL to SFT
        args.train_dir / "records.jsonl",
        out_dir=rl_dir,
        steps=args.rl_steps,
        group_size=args.rl_group_size,
        device=args.device,
        ref_checkpoint=sft_ckpt,
        limit=args.rl_limit,
        kl_beta=0.02,
        lr=1e-5,
    )
    rl_ckpt = Path(rl_summary["checkpoint"])
    phases["C"] = {
        "steps": rl_summary.get("steps"),
        "last_loss": rl_summary.get("last_loss"),
        "last_reward_mean": rl_summary.get("last_reward_mean"),
        "best_reward_mean": rl_summary.get("best_reward_mean"),
        "skipped_groups": rl_summary.get("skipped_groups"),
        "checkpoint": str(rl_ckpt),
        "telemetry": (rl_summary.get("telemetry") or {}).get("bottlenecks"),
    }

    def _score_board(suites: dict) -> float:
        smoke = suites.get("smoke") or {}
        adv = suites.get("adversarial") or {}
        held = suites.get("held_out") or {}
        return (
            2.0 * float(adv.get("parse_rate") or 0.0)
            + 2.0 * float(adv.get("placeholder_fidelity") or 0.0)
            + 1.5 * float(held.get("placeholder_fidelity") or 0.0)
            + 1.0 * float(smoke.get("structural_similarity") or 0.0)
            + 0.5 * float(smoke.get("parse_rate") or 0.0)
        )

    # --- Eval each phase; keep champion ---
    eval_cfg = ModelBuildConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        suite="smoke",
        run_root=args.run_root,
        run_id=args.run_id,
        device=args.device,
        model_name="twotower",
        grammar_constrained=True,
        grammar_ltr_primary=True,
        grammar_ltr_repair=True,
        best_of_n=1,  # match mid-train; BoN applied only if it improves champion score
        design_md_in_context=True,
        schema_in_context=True,
        rico_eval_limit=args.rico_limit,
        structural_bias=2.5,
    )
    candidates = {
        "A_sft": sft_ckpt,
        "B_pref": pref_ckpt,
        "C_rl": rl_ckpt,
    }
    phase_boards: dict[str, Any] = {}
    best_name = "A_sft"
    best_score = -1.0
    best_ckpt = sft_ckpt
    for name, ckpt in candidates.items():
        if not Path(ckpt).is_file():
            continue
        board = evaluate_suites(
            eval_cfg,
            ["smoke", "held_out", "adversarial", "ood", "rico_held"],
            checkpoint=Path(ckpt),
            write_gates=False,
        )
        suites = board.get("suites") or {}
        score = _score_board(suites)
        phase_boards[name] = {
            "score": score,
            "checkpoint": str(ckpt),
            "suites": {
                s: {
                    "parse_rate": (suites.get(s) or {}).get("parse_rate"),
                    "placeholder_fidelity": (suites.get(s) or {}).get(
                        "placeholder_fidelity"
                    ),
                    "structural_similarity": (suites.get(s) or {}).get(
                        "structural_similarity"
                    ),
                    "reward_score": (suites.get(s) or {}).get("reward_score"),
                }
                for s in ("smoke", "held_out", "adversarial", "ood", "rico_held")
            },
        }
        if score >= best_score:
            best_score = score
            best_name = name
            best_ckpt = Path(ckpt)

    final_ckpt = _copy_ckpt(best_ckpt, ckpt_dir / "last.pt")
    board = evaluate_suites(
        eval_cfg,
        ["smoke", "held_out", "adversarial", "ood", "rico_held"],
        checkpoint=final_ckpt,
        write_gates=True,
    )
    suites = board.get("suites") or {}
    gates = evaluate_ship_gates(suites)
    phases["phase_boards"] = phase_boards
    phases["champion"] = {
        "phase": best_name,
        "score": best_score,
        "checkpoint": str(final_ckpt),
    }
    phases["eval"] = {
        "pass": gates.get("pass"),
        "failures": gates.get("failures"),
        "suites": {
            name: {
                "parse_rate": m.get("parse_rate"),
                "placeholder_fidelity": m.get("placeholder_fidelity"),
                "structural_similarity": m.get("structural_similarity"),
                "reward_score": m.get("reward_score"),
                "n": m.get("n"),
            }
            for name, m in suites.items()
        },
    }
    phases["finished_at"] = datetime.now(timezone.utc).isoformat()
    phases["checkpoint"] = str(final_ckpt)
    phases["note"] = (
        "HF long-train deferred (no local SmolLM2 cache / hub egress). "
        "Scratch Phase A anti-leak SFT is the fidelity lever; B/C keep SFT if they regress."
    )

    out = run_dir / "phase_pipeline_summary.json"
    out.write_text(json.dumps(phases, indent=2) + "\n", encoding="utf-8")
    docs = Path("docs/design/phase-abc-results.json")
    docs.write_text(json.dumps(phases, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(phases, indent=2))
    print(f"wrote {out} and {docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
