#!/usr/bin/env python3
"""Train a ModelPlugin (default: TwoTower) on train-data artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.data.store import DataStore
from slm_training.harnesses.model_build import ModelBuildConfig, train


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def resolve_published_train_version(
    version: str,
    *,
    root: Path | None = None,
    store: DataStore | None = None,
) -> tuple[Path, Path | None]:
    """Resolve a committed corpus and its canonical online-sampling policy."""
    train_dir = (
        root / version
        if root is not None
        else (store or DataStore()).resolve("train", version).path
    )
    mixture = train_dir / "mixture.json"
    return train_dir, mixture if mixture.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("outputs/data/train/v1"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument(
        "--train-version",
        default=None,
        help="Use a published source-controlled corpus version from src/slm_training/resources/data/train.",
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
        "--initialize-from",
        type=Path,
        default=None,
        help=(
            "Warm-start weights/tokenizers from a serving checkpoint while "
            "resetting optimizer, RNG, step, and token counters."
        ),
    )
    parser.add_argument(
        "--replay-train-dir",
        type=Path,
        default=None,
        help="Optional immutable parent corpus mixed into continuation batches.",
    )
    parser.add_argument(
        "--replay-fraction",
        type=_probability,
        default=0.0,
        help="Fraction of each replay sampling window drawn from --replay-train-dir.",
    )
    parser.add_argument(
        "--initialization-weight-retention",
        type=_probability,
        default=0.0,
        help=(
            "Contract trainable weights toward --initialize-from after each "
            "optimizer step (0=off, 1=exact retention)."
        ),
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
        "--mixture-sampling-policy",
        choices=(
            "with_replacement",
            "capacity_aware",
            "quota_capacity_aware",
            "exposure_targeted",
        ),
        default="with_replacement",
        help="Draw mixture rows with replacement, capacity-aware, or exposure-targeted.",
    )
    parser.add_argument(
        "--mixture-exposure-target-profile",
        default=None,
        help="SDE2-03: named exposure-target profile (audit label).",
    )
    parser.add_argument(
        "--mixture-total-decision-budget",
        type=int,
        default=None,
        help="SDE2-03: total decision budget for exposure-targeted sampling.",
    )
    parser.add_argument(
        "--mixture-per-root-cap",
        type=int,
        default=None,
        help="SDE2-03: max records per root parent for exposure-targeted sampling.",
    )
    parser.add_argument(
        "--mixture-per-template-cap",
        type=int,
        default=None,
        help="SDE2-03: max records per prompt template for exposure-targeted sampling.",
    )
    parser.add_argument(
        "--mixture-max-importance-weight",
        type=float,
        default=None,
        help="SDE2-03: cap on per-action importance weights.",
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
        choices=("compositional", "lexer", "choice"),
        default="compositional",
    )
    parser.add_argument(
        "--bind-encoding",
        choices=("absolute", "relative"),
        default="absolute",
        help="C1: binder reference representation (relative = De Bruijn refs).",
    )
    parser.add_argument(
        "--denoiser-backend",
        choices=("scratch", "hf"),
        default="scratch",
        help="B4: from-scratch DenoiserTower or AR-adapted HF backbone.",
    )
    parser.add_argument(
        "--denoiser-arch",
        choices=("stacked", "shared_recursive"),
        default="stacked",
        help="SLM-138: stacked independent blocks or shared recursive transition.",
    )
    parser.add_argument(
        "--recursive-steps",
        type=int,
        default=1,
        help="SLM-138: recurrence depth for shared_recursive denoiser.",
    )
    parser.add_argument(
        "--recursive-transition-layers",
        type=int,
        default=0,
        help="SLM-138: shared transition blocks (0 = use --denoiser-layers).",
    )
    parser.add_argument(
        "--recursive-depth-supervision-weights",
        default="",
        help="SLM-138: comma-separated depth CE weights (empty = off).",
    )
    parser.add_argument(
        "--decode-min-content",
        type=int,
        default=0,
        help="A4: minimum components before EOS (0 off, -1 auto-from-inventory).",
    )
    parser.add_argument(
        "--asap-decode",
        action="store_true",
        help="A2: ASAp-style constraint-mass removal in MaskGIT decode.",
    )
    parser.add_argument(
        "--runtime-symbol-features",
        choices=("none", "surface", "role_gated", "replace"),
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
        "--compiler-alignment-loss-weight",
        type=float,
        default=0.0,
        help="Train Lark-derived branch decisions on compiler-style masked suffixes.",
    )
    parser.add_argument(
        "--compiler-alignment-margin",
        type=float,
        default=0.0,
        help="Require the gold legal branch to beat every legal alternative by this margin.",
    )
    parser.add_argument(
        "--compiler-alignment-stratified",
        action="store_true",
        help="Sample one compiler-alignment state per grammar-derived decision kind.",
    )
    parser.add_argument(
        "--compiler-alignment-semantic-exhaustive",
        action="store_true",
        help="Align every grammar-derived AST-role decision; stratify structural states.",
    )
    parser.add_argument(
        "--component-inventory-loss-weight",
        type=float,
        default=0.0,
        help="Multi-label prompt-to-gold-component inventory loss weight.",
    )
    parser.add_argument(
        "--component-inventory-decode-weight",
        type=float,
        default=0.0,
        help="Bias only compiler-legal component candidates with inventory logits.",
    )
    parser.add_argument(
        "--component-plan-loss-weight",
        type=float,
        default=0.0,
        help="Grammar-role root and bound-component count plan loss weight.",
    )
    parser.add_argument(
        "--component-plan-decode-weight",
        type=float,
        default=0.0,
        help="Bias compiler-legal components by role and remaining planned count.",
    )
    parser.add_argument(
        "--slot-component-loss-weight",
        type=float,
        default=0.0,
        help="Per-visible-slot containing-component classification loss weight.",
    )
    parser.add_argument(
        "--slot-component-focal-gamma",
        type=float,
        default=0.0,
        help="Focal exponent for slot-owner loss (0 preserves cross-entropy).",
    )
    parser.add_argument(
        "--slot-component-class-balance-power",
        type=float,
        default=0.0,
        help="Power applied to inverse corpus owner frequency (0 disables).",
    )
    parser.add_argument(
        "--slot-component-owner-rare-threshold",
        type=int,
        default=0,
        help="Treat slot-owner classes with at most this many labels as rare (0 disables).",
    )
    parser.add_argument(
        "--slot-component-owner-rare-multiplier",
        type=_positive_int,
        default=1,
        help="Sampler copies for records containing a rare visible slot owner.",
    )
    parser.add_argument(
        "--slot-component-decode-weight",
        type=float,
        default=0.0,
        help="Bias legal bound components for the next unfilled visible slot.",
    )
    parser.add_argument(
        "--slot-component-prompt-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add pooled whole-prompt context to each slot-owner prediction.",
    )
    parser.add_argument(
        "--slot-component-next-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Encode the next visible slot with each slot-owner query.",
    )
    parser.add_argument(
        "--slot-component-pair-interaction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Multiply current and next slot vectors for owner prediction.",
    )
    parser.add_argument(
        "--slot-component-lexeme-prior-weight",
        type=float,
        default=0.0,
        help="Add corpus-derived slot-lexeme owner log odds.",
    )
    parser.add_argument(
        "--slot-component-span-prior-weight",
        type=float,
        default=0.0,
        help="Bias multi-slot owners from ordered role spans.",
    )
    parser.add_argument(
        "--slot-component-content-arity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Average owner evidence across schema-derived content slots.",
    )
    parser.add_argument(
        "--component-edge-loss-weight",
        type=float,
        default=0.0,
        help="Prompt-to-resolved-AST component-edge loss weight.",
    )
    parser.add_argument(
        "--component-edge-alignment-loss-weight",
        type=float,
        default=0.0,
        help="Parent-conditioned CE over compiler-legal bound components.",
    )
    parser.add_argument(
        "--component-edge-decode-weight",
        type=float,
        default=0.0,
        help="Bias legal bound components by known parent edge logits.",
    )
    parser.add_argument(
        "--binder-component-plan-loss-weight",
        type=float,
        default=0.0,
        help="Grammar-binder instance component-plan CE weight.",
    )
    parser.add_argument(
        "--binder-component-plan-decode-weight",
        type=float,
        default=0.0,
        help="Bias legal bound components by active binder instance plan.",
    )
    parser.add_argument(
        "--binder-topology-loss-weight",
        type=float,
        default=0.0,
        help="Parent-conditioned CE over compiler-legal binder references.",
    )
    parser.add_argument(
        "--binder-topology-decode-weight",
        type=float,
        default=0.0,
        help="Bias legal binder references by active declaration topology.",
    )
    parser.add_argument(
        "--binder-arity-loss-weight",
        type=float,
        default=0.0,
        help="Grammar-binder reference-count plan CE weight.",
    )
    parser.add_argument(
        "--binder-arity-decode-weight",
        type=float,
        default=0.0,
        help="Bias legal reference-list continue/stop paths by planned arity.",
    )
    parser.add_argument(
        "--root-reference-arity-loss-weight",
        type=float,
        default=0.0,
        help="Choice-codec final-root reference-count CE weight.",
    )
    parser.add_argument(
        "--root-reference-arity-decode-weight",
        type=float,
        default=0.0,
        help="Bias terminal root-list continue/stop choices by learned arity.",
    )
    parser.add_argument(
        "--root-reference-identity-loss-weight",
        type=float,
        default=0.0,
        help="Choice-codec terminal-root reference-inclusion BCE weight.",
    )
    parser.add_argument(
        "--root-reference-identity-negative-weight",
        type=float,
        default=1.0,
        help="Relative BCE weight for generated sections excluded from the root.",
    )
    parser.add_argument(
        "--root-reference-identity-strict-subset-multiplier",
        type=_positive_int,
        default=1,
        help="Sampler copies for records whose root references a strict section subset.",
    )
    parser.add_argument(
        "--root-reference-identity-decode-weight",
        type=float,
        default=0.0,
        help="Bias terminal root-list references by learned inclusion.",
    )
    parser.add_argument(
        "--no-design-md-context",
        action="store_true",
        help="Do not concatenate DESIGN.md into the context tower prompt.",
    )
    parser.add_argument(
        "--design-md-dropout",
        type=_probability,
        default=0.0,
        help="Deterministically omit DESIGN.md for this fraction of training records.",
    )
    parser.add_argument(
        "--emit-record-nll",
        action="store_true",
        help=(
            "After training, score every train record's NLL under the final "
            "model and write record_nll.jsonl (difficulty evidence for "
            "build_train_data --difficulty-from)."
        ),
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
        "--compiler-search-mode",
        choices=("greedy", "lattice", "ptrm", "gram"),
        default="greedy",
    )
    parser.add_argument(
        "--compiler-search-trigger",
        choices=("bottom", "stagnation", "always"),
        default="stagnation",
    )
    parser.add_argument("--compiler-search-width", type=int, default=1)
    parser.add_argument("--compiler-search-noise", type=float, default=0.0)
    parser.add_argument("--compiler-search-stagnation-patience", type=int, default=2)
    parser.add_argument("--compiler-search-backtrack-limit", type=int, default=8)
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
        "--honest-slot-contract",
        action="store_true",
        help=(
            "Derive slot inventory only from user-visible prompt/DESIGN input; "
            "never inject gold record placeholders."
        ),
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
        "--action-embedding-init",
        default="none",
        choices=(
            "none",
            "current_stub",
            "schema_description",
            "expanded_description",
            "shuffled",
            "alias_aware_description",
            "alias_aware_signature_only",
            "alias_aware_shuffled",
            "description_without_canonical_name",
            "canonical_name_plus_description",
            "signature_only",
        ),
        help="SLM-163/174: source text used to initialize action embedding rows.",
    )
    parser.add_argument(
        "--action-embedding-train",
        default="frozen",
        choices=("frozen", "trainable"),
        help="SLM-163: whether action embeddings are frozen or trainable.",
    )
    parser.add_argument(
        "--action-alias-mode",
        default="canonical",
        choices=("canonical", "off", "fixed", "held_out"),
        help="SLM-174: use anonymized action aliases for description-mediated generalization.",
    )
    parser.add_argument(
        "--action-alias-manifest",
        type=Path,
        default=None,
        help="SLM-174: optional persisted alias map JSON.",
    )
    parser.add_argument(
        "--action-description-name-mode",
        default="schema",
        choices=(
            "schema",
            "alias_aware_description",
            "alias_aware_signature_only",
            "alias_aware_shuffled",
            "description_without_canonical_name",
            "canonical_name_plus_description",
            "signature_only",
        ),
        help="SLM-174: how canonical names are rendered in action-description sources.",
    )
    parser.add_argument(
        "--action-shortlist-mode",
        default="off",
        choices=("off", "description_retrieval"),
        help="SLM-176: description-based retrieve-then-rerank over live legal action sets.",
    )
    parser.add_argument(
        "--action-shortlist-k",
        type=int,
        default=8,
        help="SLM-176: top-k actions retained by description retrieval.",
    )
    parser.add_argument(
        "--action-shortlist-min-legal-size",
        type=int,
        default=16,
        help="SLM-176: minimum legal set size before shortlisting is allowed.",
    )
    parser.add_argument(
        "--action-shortlist-score-margin",
        type=float,
        default=0.0,
        help="SLM-176: include actions within this margin of the k-th score.",
    )
    parser.add_argument(
        "--action-shortlist-fallback-policy",
        default="confidence_and_coverage",
        choices=("confidence_and_coverage",),
        help="SLM-176: fallback policy when retrieval confidence is too low.",
    )
    parser.add_argument(
        "--action-shortlist-shadow-full-score",
        action="store_true",
        help="SLM-176: also score the full legal set for diagnostic comparison.",
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
    parser.add_argument(
        "--adapter-spec",
        type=Path,
        default=None,
        help="LDI2-01: load a removable TwoTower low-rank adapter directory.",
    )
    parser.add_argument(
        "--adapter-frozen",
        action="store_true",
        help="Load the adapter as frozen (no adapter parameters train).",
    )
    args = parser.parse_args(argv)
    data_store = DataStore()
    if args.train_version:
        args.train_dir, version_mixture = resolve_published_train_version(
            args.train_version, store=data_store
        )
        if args.mixture_manifest is None:
            args.mixture_manifest = version_mixture
    else:
        args.train_dir = data_store.resolve_path("train", args.train_dir)
    if args.test_dir is not None:
        args.test_dir = data_store.resolve_path("eval", args.test_dir)
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

    config = ModelBuildConfig(
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
        bind_encoding=args.bind_encoding,
        denoiser_backend=args.denoiser_backend,
        denoiser_arch=args.denoiser_arch,
        recursive_steps=args.recursive_steps,
        recursive_transition_layers=args.recursive_transition_layers,
        recursive_depth_supervision_weights=tuple(
            float(v.strip())
            for v in args.recursive_depth_supervision_weights.split(",")
            if v.strip()
        ),
        decode_min_content=max(-1, args.decode_min_content),
        asap_decode=bool(args.asap_decode),
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
        design_md_dropout=args.design_md_dropout,
        emit_record_nll=bool(args.emit_record_nll),
        ltr_loss_weight=args.ltr_loss_weight,
        ltr_prefix_loss_weight=args.ltr_prefix_loss_weight,
        compiler_alignment_loss_weight=args.compiler_alignment_loss_weight,
        compiler_alignment_margin=args.compiler_alignment_margin,
        compiler_alignment_stratified=args.compiler_alignment_stratified,
        compiler_alignment_semantic_exhaustive=(
            args.compiler_alignment_semantic_exhaustive
        ),
        component_inventory_loss_weight=args.component_inventory_loss_weight,
        component_inventory_decode_weight=args.component_inventory_decode_weight,
        component_plan_loss_weight=args.component_plan_loss_weight,
        component_plan_decode_weight=args.component_plan_decode_weight,
        slot_component_loss_weight=args.slot_component_loss_weight,
        slot_component_focal_gamma=args.slot_component_focal_gamma,
        slot_component_class_balance_power=(args.slot_component_class_balance_power),
        slot_component_owner_rare_threshold=(args.slot_component_owner_rare_threshold),
        slot_component_owner_rare_multiplier=(
            args.slot_component_owner_rare_multiplier
        ),
        slot_component_decode_weight=args.slot_component_decode_weight,
        slot_component_prompt_context=args.slot_component_prompt_context,
        slot_component_next_context=args.slot_component_next_context,
        slot_component_pair_interaction=args.slot_component_pair_interaction,
        slot_component_lexeme_prior_weight=(args.slot_component_lexeme_prior_weight),
        slot_component_span_prior_weight=(args.slot_component_span_prior_weight),
        slot_component_content_arity=args.slot_component_content_arity,
        component_edge_loss_weight=args.component_edge_loss_weight,
        component_edge_alignment_loss_weight=(
            args.component_edge_alignment_loss_weight
        ),
        component_edge_decode_weight=args.component_edge_decode_weight,
        binder_component_plan_loss_weight=(args.binder_component_plan_loss_weight),
        binder_component_plan_decode_weight=(args.binder_component_plan_decode_weight),
        binder_topology_loss_weight=args.binder_topology_loss_weight,
        binder_topology_decode_weight=args.binder_topology_decode_weight,
        binder_arity_loss_weight=args.binder_arity_loss_weight,
        binder_arity_decode_weight=args.binder_arity_decode_weight,
        root_reference_arity_loss_weight=args.root_reference_arity_loss_weight,
        root_reference_arity_decode_weight=args.root_reference_arity_decode_weight,
        root_reference_identity_loss_weight=(args.root_reference_identity_loss_weight),
        root_reference_identity_negative_weight=(
            args.root_reference_identity_negative_weight
        ),
        root_reference_identity_strict_subset_multiplier=(
            args.root_reference_identity_strict_subset_multiplier
        ),
        root_reference_identity_decode_weight=(
            args.root_reference_identity_decode_weight
        ),
        fidelity_loss_weight=args.fidelity_loss_weight,
        grammar_ltr_primary=args.grammar_ltr_primary,
        grammar_ltr_repair=args.grammar_ltr_repair,
        compiler_decode_mode=args.compiler_decode_mode,
        compiler_search_mode=args.compiler_search_mode,
        compiler_search_trigger=args.compiler_search_trigger,
        compiler_search_width=max(1, args.compiler_search_width),
        compiler_search_noise=max(0.0, args.compiler_search_noise),
        compiler_search_stagnation_patience=max(
            1, args.compiler_search_stagnation_patience
        ),
        compiler_search_backtrack_limit=max(0, args.compiler_search_backtrack_limit),
        grammar_ltr_max_tokens=args.grammar_ltr_max_tokens,
        schema_in_context=args.schema_in_context,
        slot_contract_in_context=args.slot_contract_in_context,
        slot_contract_constrained_decode=args.slot_contract_constrained_decode,
        honest_slot_contract=args.honest_slot_contract,
        retrieval_k=args.retrieval_k,
        best_of_n=args.best_of_n,
        action_embedding_init=args.action_embedding_init,
        action_embedding_train=args.action_embedding_train,
        action_alias_mode=args.action_alias_mode,
        action_alias_manifest=args.action_alias_manifest,
        action_description_name_mode=args.action_description_name_mode,
        action_shortlist_mode=args.action_shortlist_mode,
        action_shortlist_k=args.action_shortlist_k,
        action_shortlist_min_legal_size=args.action_shortlist_min_legal_size,
        action_shortlist_score_margin=args.action_shortlist_score_margin,
        action_shortlist_fallback_policy=args.action_shortlist_fallback_policy,
        action_shortlist_shadow_full_score=args.action_shortlist_shadow_full_score,
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
        initialize_from=args.initialize_from,
        replay_train_dir=args.replay_train_dir,
        replay_fraction=args.replay_fraction,
        initialization_weight_retention=args.initialization_weight_retention,
        full_state_checkpoint=not bool(args.no_full_state_checkpoint),
        mixture_manifest=args.mixture_manifest,
        mixture_min_quality_score=args.mixture_min_quality_score,
        mixture_sampling_policy=args.mixture_sampling_policy,
        mixture_exposure_target_profile=args.mixture_exposure_target_profile,
        mixture_total_decision_budget=args.mixture_total_decision_budget,
        mixture_per_root_cap=args.mixture_per_root_cap,
        mixture_per_template_cap=args.mixture_per_template_cap,
        mixture_max_importance_weight=args.mixture_max_importance_weight,
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
        adapter_spec=args.adapter_spec,
        adapter_trainable=not bool(args.adapter_frozen),
    )
    from slm_training.runtime.telemetry import run_trace

    with run_trace(
        args.run_id,
        "train",
        run_dir=config.run_dir,
        attributes={"slm.data.path": args.train_dir.as_posix()},
    ) as trace:
        summary = train(config)
        summary["trace_id"] = trace.trace_id
        summary_path = config.run_dir / "train_summary.json"
        if summary_path.is_file():
            summary_path.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
