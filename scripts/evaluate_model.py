#!/usr/bin/env python3
"""Evaluate a ModelPlugin checkpoint on a test suite (eval-driven gates)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.evals.eval_cache import EvalCache, EvalCacheConfig, EvalCacheMode
from slm_training.evals.record_schema import RUN_CLASSES
from slm_training.harnesses.model_build import ModelBuildConfig, evaluate
from slm_training.harnesses.model_build.eval_runner import evaluate_suites
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
    write_ship_gates,
)


def _check_fail_unders(metrics: dict, args: argparse.Namespace) -> int:
    if args.fail_under_parse_rate is not None:
        if float(metrics.get("parse_rate") or 0) < args.fail_under_parse_rate:
            return 2
    if args.fail_under_placeholder_fidelity is not None:
        if (
            float(metrics.get("placeholder_fidelity") or 0)
            < args.fail_under_placeholder_fidelity
        ):
            return 4
    if args.fail_under_placeholder_validity is not None:
        if (
            float(metrics.get("placeholder_validity") or 0)
            < args.fail_under_placeholder_validity
        ):
            return 7
    if args.fail_under_structural_similarity is not None:
        if (
            float(metrics.get("structural_similarity") or 0)
            < args.fail_under_structural_similarity
        ):
            return 5
    if args.fail_under_reward_score is not None:
        if float(metrics.get("reward_score") or 0) < args.fail_under_reward_score:
            return 6
    if args.fail_under_design_lint is not None:
        # Gold DESIGN.md diagnostic only — never a ship-gate / style eval.
        # Ignored when --ship-gates is set so unused-color warnings cannot fail readiness.
        if getattr(args, "ship_gates", False):
            return 0
        score = metrics.get("gold_design_lint_score")
        if score is None:
            score = metrics.get("design_lint_score")
        if score is None or float(score) < args.fail_under_design_lint:
            return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("outputs/data/eval/v1"),
    )
    parser.add_argument("--suite", default="smoke")
    parser.add_argument(
        "--suites",
        default=None,
        help="Comma-separated suites for a scoreboard (overrides --suite).",
    )
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--run-id", default="latest")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to outputs/runs/<run-id>/checkpoints/last.pt",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("outputs/data/train/v1"),
        help="Used to rebuild vocab when loading TwoTower without sidecar tokenizer.",
    )
    parser.add_argument(
        "--train-version",
        default=None,
        help="Use a published source-controlled corpus from src/slm_training/resources/data/train.",
    )
    parser.add_argument(
        "--model",
        choices=("twotower", "grammar_diffusion", "stub"),
        default="twotower",
        help="Must match the checkpoint kind.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-tokenizer",
        choices=("compositional", "lexer"),
        default=None,
        help="Override the checkpoint output tokenizer during evaluation.",
    )
    parser.add_argument(
        "--ship-gates",
        action="store_true",
        help=(
            "Apply honest multi-suite ship gates (see docs/design/adversarial-review.md) "
            "and write gates.json. Implies checking every suite in the policy."
        ),
    )
    parser.add_argument(
        "--run-class",
        choices=RUN_CLASSES,
        default=None,
        help=(
            "Honesty label stamped into the eval payload. Defaults to ship_eval "
            "under --ship-gates, else scratch_matrix; pass fixture_demo for "
            "committed-fixture wiring runs."
        ),
    )
    parser.add_argument(
        "--fail-under-parse-rate",
        type=float,
        default=None,
        help="Exit non-zero if parse_rate is below this threshold (single/primary suite).",
    )
    parser.add_argument(
        "--fail-under-placeholder-fidelity",
        type=float,
        default=None,
        help="Exit non-zero if placeholder_fidelity is below this threshold.",
    )
    parser.add_argument(
        "--fail-under-placeholder-validity",
        type=float,
        default=None,
        help="Diagnostic soft metric; prefer fidelity + --ship-gates for readiness.",
    )
    parser.add_argument(
        "--fail-under-structural-similarity",
        type=float,
        default=None,
        help="Exit non-zero if structural_similarity is below this threshold.",
    )
    parser.add_argument(
        "--fail-under-reward-score",
        type=float,
        default=None,
        help="Exit non-zero if mean composite reward_score is below this threshold.",
    )
    parser.add_argument(
        "--fail-under-design-lint",
        type=float,
        default=None,
        help=(
            "Gold DESIGN.md context lint only — not model skill / not style eval. "
            "Ignored when --ship-gates is set (warnings must not fail readiness)."
        ),
    )
    parser.add_argument(
        "--grammar-ltr-primary",
        action="store_true",
        default=None,
        help="Override checkpoint: prefer greedy LTR decode.",
    )
    parser.add_argument(
        "--grammar-ltr-repair",
        action="store_true",
        default=None,
        help="Override checkpoint: constrained LTR repair on failed parses.",
    )
    parser.add_argument(
        "--compiler-decode-mode",
        choices=("off", "forced", "restricted", "tree"),
        default="off",
        help="Compiler-drafted decode hierarchy (decode-only; default: off).",
    )
    parser.add_argument(
        "--component-inventory-decode-weight",
        type=float,
        default=None,
        help="Override the checkpoint's compiler-legal component inventory bias.",
    )
    parser.add_argument(
        "--component-plan-decode-weight",
        type=float,
        default=None,
        help="Override the checkpoint's grammar-role component-plan bias.",
    )
    parser.add_argument(
        "--slot-component-decode-weight",
        type=float,
        default=None,
        help="Override the checkpoint's slot-to-component role bias.",
    )
    parser.add_argument(
        "--semantic-role-decode-weight",
        type=float,
        default=None,
        help="Bias legal component choices using only visible semantic-role candidates.",
    )
    parser.add_argument(
        "--visible-reference-decode-weight",
        type=float,
        default=None,
        help="Prefer unused legal generated element references in root/list aggregation.",
    )
    parser.add_argument(
        "--component-edge-decode-weight",
        type=float,
        default=None,
        help="Override the checkpoint's resolved-AST component-edge bias.",
    )
    parser.add_argument(
        "--binder-component-plan-decode-weight",
        type=float,
        default=None,
        help="Override the checkpoint's grammar-binder component-plan bias.",
    )
    parser.add_argument(
        "--binder-topology-decode-weight",
        type=float,
        default=None,
        help="Override the checkpoint's grammar-binder topology bias.",
    )
    parser.add_argument(
        "--binder-arity-decode-weight",
        type=float,
        default=None,
        help="Override the checkpoint's grammar-binder arity bias.",
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
        "--verified-solver-decode",
        action="store_true",
        help="VSS1-03: prune the compiler forest via certified exact closure before ranking",
    )
    parser.add_argument("--solver-max-nodes", type=int, default=512)
    parser.add_argument(
        "--solver-unknown-policy", choices=("keep_and_rank",), default="keep_and_rank"
    )
    parser.add_argument(
        "--solver-certificate-mode",
        choices=("none", "summary", "full"),
        default="summary",
    )
    parser.add_argument(
        "--schema-in-context",
        action="store_true",
        help="Override: inject compact schema into context.",
    )
    parser.add_argument(
        "--slot-contract-in-context",
        action="store_true",
        help="Override: inject placeholder inventory (SLOT_CONTRACT) into context.",
    )
    parser.add_argument(
        "--semantic-role-contract-in-context",
        action="store_true",
        help=(
            "Normalize prompt-mentioned components and visible slots into the "
            "semantic-role contract used by E530 training."
        ),
    )
    parser.add_argument(
        "--slot-contract-constrained-decode",
        action="store_true",
        help="Override: constrain placeholder decode to slot contract.",
    )
    parser.add_argument(
        "--honest-slot-contract",
        action="store_true",
        help="Forbid hidden gold placeholder inventory during evaluation.",
    )
    parser.add_argument(
        "--contract-template-fastpath",
        action="store_true",
        help="Use the certified slot-contract template fast path during evaluation.",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=0,
        help="Override: retrieve K train skeletons into context.",
    )
    parser.add_argument(
        "--best-of-n",
        type=int,
        default=1,
        help="Override: best-of-N decode by composite reward.",
    )
    parser.add_argument(
        "--rico-limit",
        type=int,
        default=None,
        help="Cap rico_held eval size (CPU/matrix).",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Diagnostic-only cap for every selected suite; omit for full eval.",
    )
    parser.add_argument(
        "--gen-steps",
        type=int,
        default=8,
        help="Decode denoising steps; lower values are diagnostic-only.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum decode retries per record; lower values are diagnostic-only.",
    )
    parser.add_argument(
        "--skip-exact-stream-probe",
        action="store_true",
        help="Diagnostic override: skip the blocking exact grammar stream probe.",
    )
    parser.add_argument(
        "--grammar-constrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the checkpoint's grammar-constrained decode setting.",
    )
    parser.add_argument(
        "--verify-chosen-only",
        action="store_true",
        help="Diagnostic override: verify only the model-chosen token per step.",
    )
    parser.add_argument(
        "--grammar-top-k",
        type=int,
        default=None,
        help="Diagnostic override for constrained candidate breadth.",
    )
    parser.add_argument(
        "--decode-timeout-seconds",
        type=float,
        default=None,
        help="Diagnostic per-record decode timeout; omit for unlimited evaluation.",
    )
    parser.add_argument(
        "--no-design-md-context",
        action="store_true",
        help="Override: do not concatenate DESIGN.md into context.",
    )
    parser.add_argument(
        "--design-md-context",
        action="store_true",
        help="Override: force DESIGN.md into context (default: preserve checkpoint).",
    )
    parser.add_argument(
        "--grammar-dsl",
        default="openui",
        help="Grammar backend id (openui | openui-lark | openui-langcore | toy-layout).",
    )
    parser.add_argument(
        "--context-backend",
        choices=("scratch", "hf"),
        default=None,
        help="Override checkpoint context tower (scratch | hf). Default: preserve checkpoint.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="For HF context evaluation, use only locally cached model files.",
    )
    parser.add_argument(
        "--grammar-trust-model",
        action="store_true",
        help="Trust-the-model decode: no structural bias or structural reordering.",
    )
    parser.add_argument(
        "--grammar-sample-decode",
        action="store_true",
        help="Sample from renormalized legal-token distribution (DINGO-lite).",
    )
    parser.add_argument(
        "--grammar-ltr-max-tokens",
        type=int,
        default=None,
        help="Override LTR decode canvas cap (default: checkpoint or 256).",
    )
    parser.add_argument(
        "--check-decode-feasibility",
        action="store_true",
        help="Exit 9 if decode canvas makes ship parse gates unreachable.",
    )
    parser.add_argument(
        "--no-unconstrained-fallback",
        action="store_true",
        help="Disable unconstrained retries so constrained-decode adherence is measured directly.",
    )
    parser.add_argument(
        "--eval-cache-mode",
        choices=("off", "read", "read_write", "refresh"),
        default="off",
        help="SDE3-01: content-addressed suite-result cache mode (default: off).",
    )
    parser.add_argument(
        "--eval-cache-root",
        type=Path,
        default=Path("outputs/eval_cache"),
        help="SDE3-01: root directory for the evaluation cache.",
    )
    parser.add_argument(
        "--eval-shards",
        type=int,
        default=1,
        help="SDE3-01: deterministic suite sharding (currently wiring-only flag).",
    )

    args = parser.parse_args(argv)
    from slm_training.data.store import DataStore

    data_store = DataStore()
    if args.train_version:
        args.train_dir = data_store.resolve("train", args.train_version).path
    else:
        args.train_dir = data_store.resolve_path("train", args.train_dir)
    args.test_dir = data_store.resolve_path("eval", args.test_dir)

    if args.no_design_md_context and args.design_md_context:
        raise SystemExit(
            "pass only one of --design-md-context / --no-design-md-context"
        )
    design_md_override: bool | None = None
    if args.no_design_md_context:
        design_md_override = False
    elif args.design_md_context:
        design_md_override = True

    config = ModelBuildConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        suite=args.suite,
        run_class=args.run_class
        or ("ship_eval" if args.ship_gates else "scratch_matrix"),
        run_root=args.run_root,
        run_id=args.run_id,
        model_name=args.model,
        device=args.device,
        output_tokenizer=args.output_tokenizer,
        context_backend=args.context_backend or "hf",
        local_files_only=args.local_files_only,
        grammar_ltr_primary=(
            True if args.compiler_decode_mode != "off" else args.grammar_ltr_primary
        ),
        grammar_ltr_repair=args.grammar_ltr_repair,
        schema_in_context=args.schema_in_context,
        slot_contract_in_context=args.slot_contract_in_context,
        semantic_role_contract_in_context=(
            args.semantic_role_contract_in_context
        ),
        slot_contract_constrained_decode=(
            args.slot_contract_constrained_decode or args.ship_gates
        ),
        honest_slot_contract=(args.honest_slot_contract or args.ship_gates),
        contract_template_fastpath=args.contract_template_fastpath,
        retrieval_k=args.retrieval_k,
        best_of_n=args.best_of_n,
        design_md_in_context=design_md_override,
        rico_eval_limit=args.rico_limit,
        eval_limit=args.eval_limit,
        gen_steps=args.gen_steps,
        generate_max_attempts=max(1, args.max_attempts),
        allow_unconstrained_fallback=not args.no_unconstrained_fallback,
        # Preserve checkpoint settings unless an explicit override is supplied.
        grammar_skip_exact_stream_probe=(
            True if args.skip_exact_stream_probe else None
        ),
        grammar_constrained=args.grammar_constrained,
        grammar_verify_chosen_only=(True if args.verify_chosen_only else None),
        grammar_top_k=args.grammar_top_k,
        compiler_decode_mode=args.compiler_decode_mode,
        verified_solver_decode=args.verified_solver_decode,
        solver_max_nodes=args.solver_max_nodes,
        solver_unknown_policy=args.solver_unknown_policy,
        solver_certificate_mode=args.solver_certificate_mode,
        component_inventory_decode_weight=args.component_inventory_decode_weight,
        component_plan_decode_weight=args.component_plan_decode_weight,
        slot_component_decode_weight=args.slot_component_decode_weight,
        semantic_role_decode_weight=args.semantic_role_decode_weight,
        visible_reference_decode_weight=args.visible_reference_decode_weight,
        component_edge_decode_weight=args.component_edge_decode_weight,
        binder_component_plan_decode_weight=(
            args.binder_component_plan_decode_weight
        ),
        binder_topology_decode_weight=args.binder_topology_decode_weight,
        binder_arity_decode_weight=args.binder_arity_decode_weight,
        compiler_search_mode=args.compiler_search_mode,
        compiler_search_trigger=args.compiler_search_trigger,
        compiler_search_width=max(1, args.compiler_search_width),
        compiler_search_noise=max(0.0, args.compiler_search_noise),
        compiler_search_stagnation_patience=max(
            1, args.compiler_search_stagnation_patience
        ),
        compiler_search_backtrack_limit=max(0, args.compiler_search_backtrack_limit),
        decode_timeout_seconds=args.decode_timeout_seconds,
        grammar_dsl=args.grammar_dsl,
        grammar_trust_model=args.grammar_trust_model,
        grammar_sample_decode=args.grammar_sample_decode,
        grammar_ltr_max_tokens=args.grammar_ltr_max_tokens or 256,
        eval_cache_mode=args.eval_cache_mode,
        eval_cache_root=args.eval_cache_root,
        eval_shards=args.eval_shards,
    )

    if args.check_decode_feasibility and config.test_dir is not None:
        from slm_training.harnesses.model_build.decode_feasibility import (
            evaluate_decode_feasibility,
        )

        feas = evaluate_decode_feasibility(
            config.test_dir,
            canvas_cap=config.grammar_ltr_max_tokens,
            rico_limit=config.rico_eval_limit,
        )
        print(json.dumps(feas, indent=2))
        if not feas.get("pass"):
            return 9

    if args.ship_gates and not args.suites:
        args.suites = ",".join(DEFAULT_SHIP_GATES.keys())

    cache = EvalCache(
        EvalCacheConfig(
            mode=EvalCacheMode(args.eval_cache_mode),
            root=args.eval_cache_root,
        )
    )

    if args.suites:
        from slm_training.runtime.telemetry import run_trace

        suites = [s.strip() for s in args.suites.split(",") if s.strip()]
        with run_trace(args.run_id, "eval", run_dir=config.run_dir):
            scoreboard = evaluate_suites(
                config,
                suites,
                checkpoint=args.checkpoint,
                write_gates=args.ship_gates,
                cache=cache,
            )
        print(json.dumps({k: v for k, v in scoreboard.items()}, indent=2))
        if args.ship_gates:
            gates = scoreboard.get("gates") or write_ship_gates(
                config.run_dir, scoreboard["suites"]
            )
            # Re-read full payload when only summary was embedded.
            if "pass" not in gates or "failures" not in gates:
                gates = evaluate_ship_gates(scoreboard["suites"])
                write_ship_gates(config.run_dir, scoreboard["suites"])
            return 0 if gates.get("pass") else 8
        # Legacy: fail-under applies to every listed suite (not smoke-only).
        for suite_name in suites:
            metrics = scoreboard["suites"][suite_name]
            code = _check_fail_unders(metrics, args)
            if code:
                return code
        return 0

    from slm_training.runtime.telemetry import run_trace

    with run_trace(args.run_id, "eval", run_dir=config.run_dir):
        metrics = evaluate(config, checkpoint=args.checkpoint, cache=cache)
    summary = {k: v for k, v in metrics.items() if k != "details"}
    print(json.dumps(summary, indent=2))
    return _check_fail_unders(metrics, args)


if __name__ == "__main__":
    raise SystemExit(main())
