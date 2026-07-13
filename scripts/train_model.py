#!/usr/bin/env python3
"""Train a ModelPlugin (default: TwoTower) on train-data artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.model_build import ModelBuildConfig, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("outputs/train_data/v1"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--run-id", default="latest")
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Optional test dir for periodic eval (--eval-every).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run smoke eval every N steps (0 disables).",
    )
    parser.add_argument("--eval-suite", default="smoke")
    parser.add_argument(
        "--model",
        choices=("twotower", "stub"),
        default="twotower",
        help="Model plug-in to train (default: twotower).",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto|cpu|cuda|npu:0 (auto picks cuda→npu→cpu).",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--context-layers", type=int, default=2)
    parser.add_argument("--denoiser-layers", type=int, default=4)
    parser.add_argument("--mask-min", type=float, default=0.15)
    parser.add_argument("--mask-max", type=float, default=0.85)
    parser.add_argument("--gen-steps", type=int, default=8)
    parser.add_argument(
        "--context-backend",
        choices=("scratch", "hf"),
        default="hf",
        help="Context tower backend (default: hf; use scratch for offline CI).",
    )
    parser.add_argument(
        "--hf-model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HF model id when --context-backend hf.",
    )
    parser.add_argument(
        "--freeze-context",
        action="store_true",
        help="Freeze context tower weights.",
    )
    parser.add_argument(
        "--no-freeze-context",
        action="store_true",
        help="Allow context tower gradients (overrides HF default freeze).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load HF weights only from local cache.",
    )
    parser.add_argument(
        "--no-grammar",
        action="store_true",
        help="Disable streaming/grammar-constrained decode at generate time.",
    )
    parser.add_argument(
        "--grammar-dsl",
        default="openui",
        help=(
            "Grammar backend id for parse/stream/constrained decode "
            "(openui | openui-lark | openui-langcore | toy-layout)."
        ),
    )
    parser.add_argument("--grammar-top-k", type=int, default=16)
    parser.add_argument("--structural-bias", type=float, default=1.25)
    parser.add_argument(
        "--ltr-loss-weight",
        type=float,
        default=0.5,
        help="Auxiliary prefix-LM loss weight (helps LTR generate).",
    )
    parser.add_argument(
        "--no-design-md-context",
        action="store_true",
        help="Do not concatenate DESIGN.md into the context tower prompt.",
    )
    parser.add_argument(
        "--grammar-ltr-primary",
        action="store_true",
        help="Prefer greedy LTR decode at generate time.",
    )
    parser.add_argument(
        "--grammar-ltr-repair",
        action="store_true",
        help="Re-decode failing LTR outputs with streaming grammar constraints.",
    )
    parser.add_argument(
        "--grammar-ltr-max-tokens",
        type=int,
        default=256,
        help="Max tokens for LTR / constrained repair canvases.",
    )
    parser.add_argument(
        "--fidelity-loss-weight",
        type=float,
        default=0.5,
        help="Extra CE weight on gold placeholder tokens during training.",
    )
    parser.add_argument(
        "--schema-in-context",
        action="store_true",
        help="Inject compact OpenUI component schema into the context tower.",
    )
    parser.add_argument(
        "--slot-contract-in-context",
        action="store_true",
        help="Inject record placeholder inventory (SLOT_CONTRACT) into context.",
    )
    parser.add_argument(
        "--slot-contract-constrained-decode",
        action="store_true",
        help="Restrict placeholder decode to the slot contract inventory.",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=0,
        help="Retrieve K nearest train OpenUI skeletons into context (0 disables).",
    )
    parser.add_argument(
        "--best-of-n",
        type=int,
        default=1,
        help="Generate N candidates and pick by composite reward.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Sample train batches by curriculum stage A→B→C (soft mix by default).",
    )
    parser.add_argument(
        "--hard-curriculum",
        action="store_true",
        help="Use hard A→B→C stage cutovers instead of soft mix (can leak C into smoke).",
    )
    parser.add_argument(
        "--grammar-trust-model",
        action="store_true",
        help="Trust-the-model decode: no structural bias or structural reordering.",
    )
    parser.add_argument(
        "--eval-suites",
        default="",
        help="Comma-separated mid-train eval suites (e.g. smoke,held_out,adversarial).",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable train cycle telemetry JSON.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable autocast AMP on cuda/npu.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the denoiser (Inductor / CUDA graphs when available).",
    )
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="torch.compile mode (reduce-overhead uses CUDA graphs on GPU).",
    )
    parser.add_argument(
        "--fast-train",
        action="store_true",
        help=(
            "Enable train-speed bundle: cache context, fuse LTR loss, "
            "AMP+compile when the device supports them."
        ),
    )
    parser.add_argument(
        "--no-cache-context",
        action="store_true",
        help="Disable frozen HF / DESIGN.md context caching.",
    )
    parser.add_argument(
        "--no-fuse-ltr",
        action="store_true",
        help="Disable fused mask+LTR (use second denoiser forward for LTR).",
    )
    parser.add_argument(
        "--fastpath-aux-weight",
        type=float,
        default=0.0,
        help="Cheap structural force-align aux loss weight (0 disables).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (larger effective batch).",
    )
    parser.add_argument(
        "--parallel-unmask",
        default="adaptive",
        choices=("topk", "confidence", "adaptive"),
        help="MaskGIT parallel unmask policy (adaptive = mean-field-lite spacing).",
    )
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.0,
        help="Stub-only: rate of intentional broken generations.",
    )
    args = parser.parse_args(argv)

    from slm_training.accel import detect_device

    accel = detect_device(args.device)
    device = accel.device if args.device in {"auto", "best"} else args.device

    freeze = args.freeze_context
    if args.context_backend == "hf" and not args.no_freeze_context:
        freeze = True
    if args.no_freeze_context:
        freeze = False

    use_amp = bool(args.amp)
    use_compile = bool(args.compile)
    cache_context = not bool(args.no_cache_context)
    fuse_ltr = not bool(args.no_fuse_ltr)
    if args.fast_train:
        cache_context = True
        fuse_ltr = True
        # AMP only when accel advertises it (cuda/npu); compile still useful on CPU.
        if accel.amp:
            use_amp = True
        use_compile = True

    summary = train(
        ModelBuildConfig(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            suite=args.eval_suite,
            run_root=args.run_root,
            run_id=args.run_id,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=device,
            model_name=args.model,
            d_model=args.d_model,
            n_heads=args.n_heads,
            context_layers=args.context_layers,
            denoiser_layers=args.denoiser_layers,
            mask_min=args.mask_min,
            mask_max=args.mask_max,
            gen_steps=args.gen_steps,
            context_backend=args.context_backend,
            hf_model_name=args.hf_model,
            freeze_context=freeze,
            local_files_only=args.local_files_only,
            grammar_constrained=not args.no_grammar,
            grammar_dsl=args.grammar_dsl,
            grammar_top_k=args.grammar_top_k,
            structural_bias=args.structural_bias,
            design_md_in_context=not args.no_design_md_context,
            ltr_loss_weight=args.ltr_loss_weight,
            fidelity_loss_weight=args.fidelity_loss_weight,
            grammar_ltr_primary=args.grammar_ltr_primary,
            grammar_ltr_repair=args.grammar_ltr_repair,
            grammar_ltr_max_tokens=args.grammar_ltr_max_tokens,
            schema_in_context=args.schema_in_context,
            slot_contract_in_context=args.slot_contract_in_context,
            slot_contract_constrained_decode=args.slot_contract_constrained_decode,
            retrieval_k=args.retrieval_k,
            best_of_n=args.best_of_n,
            use_curriculum=args.curriculum,
            mix_curriculum=not bool(args.hard_curriculum),
            use_amp=use_amp,
            use_compile=use_compile,
            compile_mode=args.compile_mode,
            grad_accum_steps=args.grad_accum,
            parallel_unmask=args.parallel_unmask,
            cache_context=cache_context,
            fuse_ltr_loss=fuse_ltr,
            grammar_fastpath=True,
            fastpath_aux_weight=args.fastpath_aux_weight,
            grammar_trust_model=args.grammar_trust_model,
            noise_rate=args.noise_rate,
            eval_every=args.eval_every,
            eval_suite=args.eval_suite,
            eval_suites=args.eval_suites,
            telemetry=not bool(args.no_telemetry),
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
