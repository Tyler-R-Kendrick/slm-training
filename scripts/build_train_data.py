#!/usr/bin/env python3
"""Build versioned training-data artifacts (high-quality, deterministic)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.train_data import TrainDataConfig, build_train_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="all",
        choices=[
            "rico",
            "fixture",
            "both",
            "awwwards",
            "rico+awwwards",
            "existing",
            "programspec",
            "language_contract",
            "deconstruct",
            "render",
            "integrated",
            "all",
        ],
        help="Training data source (default: all).",
    )
    parser.add_argument(
        "--seed-path",
        type=Path,
        default=Path("src/slm_training/resources/train_seeds.jsonl"),
        help="JSONL seed fixtures (used when source includes fixtures).",
    )
    parser.add_argument(
        "--derive-from",
        type=Path,
        default=None,
        help="Existing records.jsonl to use as roots when --source existing.",
    )
    parser.add_argument(
        "--rico-path",
        type=Path,
        default=Path("src/slm_training/resources/rico/semantic_train.jsonl"),
        help="Local RICO semantic JSONL (HF-exported screens).",
    )
    parser.add_argument(
        "--rico-hf-split",
        default=None,
        help="Optional live Hugging Face RICO split (train/validation/test).",
    )
    parser.add_argument("--rico-limit", type=int, default=None)
    parser.add_argument("--max-children", type=int, default=6)
    parser.add_argument(
        "--min-verification-tier",
        choices=("Bronze", "Silver", "Gold"),
        default=None,
        help="Require this independent verification tier before quality filtering.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/train_data"),
    )
    parser.add_argument("--version", default="v1")
    parser.add_argument(
        "--immutable",
        action="store_true",
        help="Fail instead of overwriting an existing versioned snapshot.",
    )
    parser.add_argument(
        "--synthesizer",
        default="quality",
        choices=[
            "quality",
            "template",
            "layout",
            "frontier",
            "none",
            "noop",
            "off",
        ],
        help="Deterministic synthesizer (default: quality = layout aug + templates).",
    )
    parser.add_argument(
        "--programspec-path",
        type=Path,
        default=Path("outputs/progspec/programs.jsonl"),
        help="Optional JSONL ProgramSpec roots; deterministic generation is the fallback.",
    )
    parser.add_argument("--programspec-count", type=int, default=16)
    parser.add_argument("--programspec-seed", type=int, default=0)
    parser.add_argument(
        "--language-contract",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deconstruct-path",
        type=Path,
        default=Path("src/slm_training/resources/deconstruct/pipeline.jsonl"),
    )
    parser.add_argument(
        "--render-path",
        type=Path,
        default=Path("src/slm_training/resources/render/sample_program.json"),
    )
    parser.add_argument(
        "--frontier-artifact-root",
        type=Path,
        default=Path("src/slm_training/resources/frontier"),
    )
    parser.add_argument(
        "--frontier-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--repairs-per-program", type=int, default=1)
    parser.add_argument(
        "--edit-derivatives",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--design-md-contrastive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--diffusion-online",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record train-time diffusion policies; never materialize noisy targets.",
    )
    parser.add_argument(
        "--governance-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mixture-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.55,
        help="Drop records below this quality score after validation.",
    )
    parser.add_argument(
        "--allow-missing-design-md",
        action="store_true",
        help="Do not require DESIGN.md on every kept record.",
    )
    parser.add_argument(
        "--max-openui-chars",
        type=int,
        default=None,
        help="Drop layouts longer than this many characters (compact core sets).",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="Drop layouts with more than this many component calls.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help=(
            "Tag train records with curriculum stages A/B/C and inject stress "
            "adversarial examples without importing eval fixtures."
        ),
    )
    parser.add_argument(
        "--namespace-augment",
        action="store_true",
        help="Emit namespace-augmented train variants (:acme.* re-prefix).",
    )
    parser.add_argument(
        "--prompt-slot-contract",
        action="store_true",
        help="Append each record's declared placeholder inventory to its prompt.",
    )
    parser.add_argument(
        "--max-records-per-parent",
        type=int,
        default=None,
        help=(
            "Exposure cap: keep at most N records per root parent "
            "(original + synth variants). Default: uncapped."
        ),
    )
    parser.add_argument(
        "--fuzzy-dedup",
        action="store_true",
        help="P1a: MinHash fuzzy dedup within structure clusters (Jaccard≥0.92).",
    )
    parser.add_argument(
        "--fuzzy-jaccard",
        type=float,
        default=0.92,
        help="Fuzzy MinHash Jaccard threshold (default: 0.92).",
    )
    parser.add_argument(
        "--semantic-cluster-cap",
        type=int,
        default=None,
        help="P1a: max representatives per semantic cluster (default: uncapped).",
    )
    args = parser.parse_args(argv)

    result = build_train_data(
        TrainDataConfig(
            seed_path=args.seed_path
            if args.source in {"fixture", "both", "all"}
            else None,
            rico_path=args.rico_path
            if args.source in {"rico", "both", "rico+awwwards", "all"}
            else None,
            source=args.source,
            derive_from=args.derive_from,
            output_root=args.output_root,
            version=args.version,
            immutable=args.immutable,
            synthesizer=args.synthesizer,
            rico_hf_split=args.rico_hf_split,
            rico_limit=args.rico_limit,
            max_children=args.max_children,
            min_quality_score=args.min_quality_score,
            min_verification_tier=args.min_verification_tier,
            require_design_md=not args.allow_missing_design_md,
            max_openui_chars=args.max_openui_chars,
            max_components=args.max_components,
            curriculum=args.curriculum,
            namespace_augment=args.namespace_augment,
            prompt_slot_contract=args.prompt_slot_contract,
            max_records_per_parent=args.max_records_per_parent,
            fuzzy_dedup=bool(args.fuzzy_dedup),
            fuzzy_jaccard=float(args.fuzzy_jaccard),
            semantic_cluster_cap=args.semantic_cluster_cap,
            programspec_path=args.programspec_path,
            programspec_count=args.programspec_count,
            programspec_seed=args.programspec_seed,
            include_language_contract=args.language_contract,
            deconstruct_path=args.deconstruct_path,
            render_path=args.render_path,
            frontier_artifact_root=args.frontier_artifact_root,
            include_frontier_artifacts=args.frontier_artifacts,
            repairs_per_program=args.repairs_per_program,
            include_edit_derivatives=args.edit_derivatives,
            include_design_md_contrastive=args.design_md_contrastive,
            diffusion_online=args.diffusion_online,
            governance_artifacts=args.governance_artifacts,
            mixture_manifest=args.mixture_manifest,
        )
    )
    print(json.dumps(result["stats"], indent=2))
    print(f"wrote {result['output_dir']}")
    print(f"content_fingerprint={result['manifest'].get('content_fingerprint')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
