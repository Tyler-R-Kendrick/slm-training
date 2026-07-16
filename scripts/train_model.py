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
    parser.add_argument(
        "--train-version",
        default=None,
        help="Use a published source-controlled corpus version from src/slm_training/resources/train_data.",
    )
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
    parser.add_argument(
        "--loss-eval-every",
        type=int,
        default=0,
        help="Run deterministic denoising-NLL suites every N steps (0 disables).",
    )
    parser.add_argument(
        "--loss-suite-version",
        default="v1",
        help="Frozen loss-suite JSON version (default: v1).",
    )
    parser.add_argument(
        "--loss-mask-seed",
        type=int,
        default=0,
        help="Seed mixed into fixed-mask NLL draws (default: 0).",
    )
    parser.add_argument(
        "--target-token-budget",
        type=int,
        default=None,
        help="Stop training once seen target tokens reach this budget.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from a last_full_state.pt checkpoint (bit-exact).",
    )
    parser.add_argument(
        "--no-full-state-checkpoint",
        action="store_true",
        help="Skip writing last_full_state.pt (serving last.pt still written).",
    )
    parser.add_argument(
        "--mixture-manifest",
        type=Path,
        default=None,
        help="P1b: JSON mixture weights for online family-weighted sampling.",
    )
    parser.add_argument(
        "--mixture-min-quality-score",
        type=float,
        default=0.0,
        help="Exclude judged records below this score when sampling a mixture.",
    )
    parser.add_argument(
        "--register-promoted",
        action="store_true",
        help="P1d: write promoted.pt from best_weighted_nll / best_ship / last.",
    )
    parser.add_argument("--eval-suite", default="smoke")
    parser.add_argument(
        "--model",
        choices=("twotower", "grammar_diffusion", "stub"),
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
        help=(
            "Device: auto|cpu|cuda|npu:0|directml "
            "(auto picks CUDA→Ascend NPU→DirectML→CPU)."
        ),
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--context-layers", type=int, default=2)
    parser.add_argument("--denoiser-layers", type=int, default=4)
    parser.add_argument("--mask-min", type=float, default=0.15)
    parser.add_argument("--mask-max", type=float, default=0.85)
    parser.add_argument(
        "--mask-pattern",
        choices=("random", "mixed", "diffusion"),
        default="random",
        help="Training corruption family (diffusion enables the online adapter).",
    )
    parser.add_argument(
        "--output-tokenizer",
        choices=("compositional", "lexer"),
        default="compositional",
    )
    parser.add_argument(
        "--runtime-symbol-features",
        choices=("none", "surface", "role_gated"),
        default="none",
    )
    parser.add_argument(
        "--symbol-slot-augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--semantic-candidate-masks",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--constraint-graph-mode",
        choices=("off", "grammar", "hybrid"),
        default="off",
    )
    parser.add_argument(
        "--grammar-completion-bounds",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--grammar-equivalence-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--grammar-active-symbol-bitsets",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--compact-active-canvas",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--diffusion-policies",
        default=",".join(ModelBuildConfig.diffusion_policies),
        help="Comma-separated online policies used with --mask-pattern diffusion.",
    )
    parser.add_argument(
        "--diffusion-length-buckets",
        default=",".join(map(str, ModelBuildConfig.diffusion_length_buckets)),
        help="Comma-separated target-length bucket upper bounds.",
    )
    parser.add_argument("--diffusion-overallocate", type=int, default=8)
    parser.add_argument("--diffusion-length-loss-weight", type=float, default=0.1)
    parser.add_argument("--gen-steps", type=int, default=8)
    parser.add_argument(
        "--topology-actions", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--topology-structural-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--topology-heterogeneous-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--topology-critic-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--topology-bounded-buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--scope-contracts", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--scope-independent-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--scope-local-oracle", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--scope-contract-negatives",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--topology-max-nodes", type=int, default=256)
    parser.add_argument("--topology-max-active", type=int, default=64)
    parser.add_argument("--topology-max-arity", type=int, default=8)
    parser.add_argument("--topology-max-depth", type=int, default=32)
    parser.add_argument("--topology-max-phases", type=int, default=32)
    parser.add_argument("--topology-global-sync-interval", type=int, default=4)
    parser.add_argument("--topology-accept-threshold", type=float, default=0.5)
    parser.add_argument("--topology-contract-threshold", type=float, default=0.25)
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
        "--hf-revision",
        default=None,
        help="Immutable Hugging Face commit revision for the context model.",
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
        "--ltr-prefix-loss-weight",
        type=float,
        default=0.0,
        help="Extra weight for the first three LTR positions (root/early structure).",
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
        "--compiler-decode-mode",
        choices=("off", "forced", "restricted", "tree"),
        default="off",
        help="Compiler-drafted decode hierarchy used by in-run evaluations.",
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
        default=None,
        choices=("default", "reduce-overhead", "max-autotune"),
        help=(
            "torch.compile mode (reduce-overhead uses CUDA graphs on GPU). "
            "Default: reduce-overhead under --fast-train on CUDA, else default."
        ),
    )
    parser.add_argument(
        "--fast-train",
        action="store_true",
        help=(
            "Enable train-speed bundle: cache context, fuse LTR loss, "
            "AMP+compile when the device supports them. "
            "Also auto-enabled on HF Jobs / SLM_FAST_TRAIN=1."
        ),
    )
    parser.add_argument(
        "--no-fast-train",
        action="store_true",
        help=(
            "Disable the train-speed bundle even when HF Jobs / "
            "SLM_FAST_TRAIN would enable it."
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
    parser.add_argument(
        "--checkpoint-bucket",
        default=None,
        help=(
            "HF Bucket URI or id for durable checkpoints "
            "(default: hf://buckets/TKendrick/OpenUI when sync is on). "
            "Pass empty string to disable auto bucket selection."
        ),
    )
    parser.add_argument(
        "--sync-checkpoints",
        action="store_true",
        help="Force upload checkpoints to the HF Bucket after training.",
    )
    parser.add_argument(
        "--no-sync-checkpoints",
        action="store_true",
        help="Keep checkpoints local-only (matrix/CI/scratch).",
    )
    parser.add_argument(
        "--checkpoint-bucket-dry-run",
        action="store_true",
        help="Plan bucket sync without uploading (debug / no-write environments).",
    )
    args = parser.parse_args(argv)
    if args.train_version:
        args.train_dir = (
            Path("src/slm_training/resources/train_data") / args.train_version
        )
    if (args.eval_every > 0 or args.loss_eval_every > 0) and not args.test_dir:
        parser.error(
            "--test-dir is required when --eval-every or --loss-eval-every is enabled"
        )
    if args.sync_checkpoints and args.no_sync_checkpoints:
        parser.error("use only one of --sync-checkpoints / --no-sync-checkpoints")
    if args.fast_train and args.no_fast_train:
        parser.error("use only one of --fast-train / --no-fast-train")

    from slm_training.harnesses.model_build.checkpoint_bucket import (
        DEFAULT_CHECKPOINT_BUCKET_URI,
    )
    from slm_training.runtime.accel import detect_device, prefer_fast_train_env

    accel = detect_device(args.device)
    device = (
        accel.device
        if args.device in {"auto", "best", "dml", "directml"}
        else args.device
    )

    freeze = args.freeze_context
    if args.context_backend == "hf" and not args.no_freeze_context:
        freeze = True
    if args.no_freeze_context:
        freeze = False

    use_amp = bool(args.amp)
    use_compile = bool(args.compile)
    cache_context = not bool(args.no_cache_context)
    fuse_ltr = not bool(args.no_fuse_ltr)
    want_fast = bool(args.fast_train) or (
        prefer_fast_train_env() and not bool(args.no_fast_train)
    )
    if want_fast:
        cache_context = True
        fuse_ltr = True
        # AMP only when accel advertises it (cuda/npu); compile still useful on CPU.
        if accel.amp:
            use_amp = True
        use_compile = True
    compile_mode = args.compile_mode
    if compile_mode is None:
        compile_mode = (
            "reduce-overhead" if want_fast and accel.backend == "cuda" else "default"
        )

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
            mask_pattern=args.mask_pattern,
            output_tokenizer=args.output_tokenizer,
            runtime_symbol_features=args.runtime_symbol_features,
            symbol_slot_augmentation=args.symbol_slot_augmentation,
            semantic_candidate_masks=args.semantic_candidate_masks,
            constraint_graph_mode=args.constraint_graph_mode,
            grammar_completion_bounds=args.grammar_completion_bounds,
            grammar_equivalence_cache=args.grammar_equivalence_cache,
            grammar_active_symbol_bitsets=args.grammar_active_symbol_bitsets,
            compact_active_canvas=args.compact_active_canvas,
            diffusion_policies=tuple(
                value.strip()
                for value in args.diffusion_policies.split(",")
                if value.strip()
            ),
            diffusion_length_buckets=tuple(
                int(value.strip())
                for value in args.diffusion_length_buckets.split(",")
                if value.strip()
            ),
            diffusion_overallocate=args.diffusion_overallocate,
            diffusion_length_loss_weight=args.diffusion_length_loss_weight,
            gen_steps=args.gen_steps,
            topology_actions=args.topology_actions,
            topology_structural_embeddings=args.topology_structural_embeddings,
            topology_heterogeneous_noise=args.topology_heterogeneous_noise,
            topology_critic_decode=args.topology_critic_decode,
            topology_bounded_buffer=args.topology_bounded_buffer,
            scope_contracts=args.scope_contracts,
            scope_independent_noise=args.scope_independent_noise,
            scope_local_oracle=args.scope_local_oracle,
            scope_contract_negatives=args.scope_contract_negatives,
            topology_max_nodes=args.topology_max_nodes,
            topology_max_active=args.topology_max_active,
            topology_max_arity=args.topology_max_arity,
            topology_max_depth=args.topology_max_depth,
            topology_max_phases=args.topology_max_phases,
            topology_global_sync_interval=args.topology_global_sync_interval,
            topology_accept_threshold=args.topology_accept_threshold,
            topology_contract_threshold=args.topology_contract_threshold,
            context_backend=args.context_backend,
            hf_model_name=args.hf_model,
            hf_model_revision=args.hf_revision,
            freeze_context=freeze,
            local_files_only=args.local_files_only,
            grammar_constrained=not args.no_grammar,
            grammar_dsl=args.grammar_dsl,
            grammar_top_k=args.grammar_top_k,
            structural_bias=args.structural_bias,
            design_md_in_context=not args.no_design_md_context,
            ltr_loss_weight=args.ltr_loss_weight,
            ltr_prefix_loss_weight=args.ltr_prefix_loss_weight,
            fidelity_loss_weight=args.fidelity_loss_weight,
            grammar_ltr_primary=args.grammar_ltr_primary,
            grammar_ltr_repair=args.grammar_ltr_repair,
            compiler_decode_mode=args.compiler_decode_mode,
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
            compile_mode=compile_mode,
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
            loss_eval_every=args.loss_eval_every,
            loss_suite_version=args.loss_suite_version,
            loss_mask_seed=args.loss_mask_seed,
            target_token_budget=args.target_token_budget,
            resume_from=args.resume_from,
            full_state_checkpoint=not bool(args.no_full_state_checkpoint),
            mixture_manifest=args.mixture_manifest,
            mixture_min_quality_score=args.mixture_min_quality_score,
            register_promoted=bool(args.register_promoted),
            telemetry=not bool(args.no_telemetry),
            checkpoint_bucket=(
                args.checkpoint_bucket
                if args.checkpoint_bucket is not None
                else (
                    DEFAULT_CHECKPOINT_BUCKET_URI
                    if (
                        not args.no_sync_checkpoints
                        and (args.sync_checkpoints or args.context_backend == "hf")
                    )
                    else None
                )
            ),
            sync_checkpoints=(
                False
                if args.no_sync_checkpoints
                else True
                if args.sync_checkpoints or args.context_backend == "hf"
                else False
            ),
            checkpoint_bucket_dry_run=bool(args.checkpoint_bucket_dry_run),
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
