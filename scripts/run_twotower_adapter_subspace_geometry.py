"""CLI for the LDI2-02 (SLM-125) TwoTower adapter-subspace geometry diagnostic.

Produces a canonical JSON report that includes an explicit fail-closed
authorization decision for the downstream LDI2-03 (SLM-126) adapter-vs-full-update
campaign matrix.

Examples::

    # Fixture/scratch wiring evidence (no external checkpoint or corpus):
    python scripts/run_twotower_adapter_subspace_geometry.py --fixture \
        --ranks 2,4,8 --target-modules attn_q,attn_v --out outputs/ldi2_geometry/report.json

    # Canonical run against a frozen checkpoint and an admitted V2 corpus:
    python scripts/run_twotower_adapter_subspace_geometry.py \
        --checkpoint outputs/runs/e228/model.pt \
        --events outputs/data/e283_admitted/events.jsonl \
        --materializer pareto \
        --ranks 2,4,8,16 --target-modules attn_q,attn_v \
        --out outputs/ldi2_geometry/report.json

No training, checkpoint, or quality claim is produced here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.distill.trace_store import checkpoint_sha
from slm_training.harnesses.preference.adapter_subspace_geometry import (
    profile_adapter_subspace_geometry,
    write_geometry_report,
)
from slm_training.harnesses.preference.decision_diagnostics import DiagnosticBudget
from slm_training.harnesses.preference.decision_events_v2 import (
    DecisionStateV2,
    ObjectiveView,
    materialize_pareto,
    materialize_set_valued,
    materialize_single_best_worst,
    materialize_thresholded,
)
from slm_training.harnesses.preference.local_decisions import (
    DecisionEventV2,
    admit_semantic_corpus,
    load_decision_events_v2,
    split_for_group,
)
from slm_training.models.adapters import TwoTowerAdapterSpec
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

_MATERIALIZERS: dict[str, Any] = {
    "pareto": materialize_pareto,
    "set_valued": materialize_set_valued,
    "single_best_worst": materialize_single_best_worst,
    "thresholded": materialize_thresholded,
}


def _parse_ranks(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_modules(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _fixture_model(device: str = "cpu") -> TwoTowerModel:
    record = ExampleRecord(
        id="a",
        prompt="Card",
        openui='root = TextContent(":card.title")',
        split="train",
        placeholders=[":card.title"],
    )
    return TwoTowerModel.from_records(
        [record],
        config=TwoTowerConfig(
            d_model=16,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            max_target_len=8,
            seed=0,
        ),
        device=device,
    )


def _group_for_split(split: str, seed: str) -> str:
    group = seed
    while split_for_group(group) != split:
        group += "x"
    return group


def _fixture_corpus() -> list[tuple[DecisionStateV2, ObjectiveView]]:
    def state(group: str, kind: str, role: str) -> DecisionStateV2:
        return DecisionStateV2(
            group_id=group,
            architecture="twotower",
            context_text="root=Stack([",
            canvas_ids=(1, 2, 3),
            decision_position=1,
            legal_action_ids=(4, 9, 10),
            decision_kind=kind,
            abstract_state_role=role,
            grammar_state_hash="gsh",
            policy_checkpoint_sha="pcs",
            tokenizer_sha="tsha",
            decode_config_hash="dch",
            verifier_bundle_hash="vbh",
            split=split_for_group(group),
        )

    def view(
        *, good: tuple[int, ...] = (4,), bad: tuple[int, ...] = (9,)
    ) -> ObjectiveView:
        return ObjectiveView(
            good_action_ids=good,
            bad_action_ids=bad,
            ambiguous_action_ids=(),
            unobserved_action_ids=(),
            weights=(),
            materializer_id="pareto",
            materializer_config_hash="cfg",
            trainable=True,
        )

    train_a = _group_for_split("train", "obj-a")
    train_b = _group_for_split("train", "obj-b")
    held = _group_for_split("held_out", "obj-h")
    return [
        (state(train_a, "component", "component_slot"), view(good=(4,), bad=(9,))),
        (
            state(train_b, "grammar_comma", "grammar_slot"),
            view(good=(9,), bad=(4,)),
        ),
        (state(held, "component", "component_slot"), view(good=(4,), bad=(9,))),
    ]


def _materialize_corpus(
    events: list[DecisionEventV2], materializer_name: str
) -> list[tuple[DecisionStateV2, ObjectiveView]]:
    materializer = _MATERIALIZERS.get(materializer_name)
    if materializer is None:
        raise ValueError(
            f"unknown materializer {materializer_name!r}; "
            f"expected one of {sorted(_MATERIALIZERS)}"
        )
    corpus: list[tuple[DecisionStateV2, ObjectiveView]] = []
    for event in events:
        view = materializer(event.state, event.outcomes)
        corpus.append((event.state, view))
    return corpus


def _spec_factory(
    model: TwoTowerModel,
    cell: dict[str, Any],
    *,
    base_checkpoint_sha: str,
) -> TwoTowerAdapterSpec:
    rank = int(cell["rank"])
    return TwoTowerAdapterSpec(
        method="low_rank",
        rank=rank,
        alpha=float(2 * rank),
        dropout=0.0,
        target_modules=cell["target_modules"],
        base_compatibility_fingerprint=model.compatibility_fingerprint(),
        base_checkpoint_sha=base_checkpoint_sha,
        tokenizer_sha=model.artifact_identity()["tokenizer_sha"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="frozen parent checkpoint path"
    )
    parser.add_argument(
        "--events", type=Path, default=None, help="V2 decision events JSONL path"
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="use the committed 3-state fixture corpus and tiny model (wiring evidence only)",
    )
    parser.add_argument(
        "--materializer",
        choices=sorted(_MATERIALIZERS),
        default="pareto",
        help="objective materializer for V2 events",
    )
    parser.add_argument(
        "--ranks",
        type=_parse_ranks,
        default="2,4,8,16",
        help="comma-separated adapter ranks to profile",
    )
    parser.add_argument(
        "--target-modules",
        type=_parse_modules,
        default="attn_q,attn_v",
        help="comma-separated target module names",
    )
    parser.add_argument(
        "--module-restricted",
        action="store_true",
        help="add a second cell set restricted to the last denoiser layer",
    )
    parser.add_argument(
        "--objective",
        default="ftpo_set",
        help="local objective passed to the legal-token guard",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=3.0,
        help="cumulative wall budget in minutes (default 3, hard cap 3)",
    )
    parser.add_argument(
        "--admit",
        action="store_true",
        help="fail closed if the corpus does not pass objective-support admission",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for the model",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="path to write the canonical JSON report",
    )
    args = parser.parse_args(argv)

    if args.fixture:
        corpus = _fixture_corpus()
        base_checkpoint_sha = "fixture"
    else:
        if args.checkpoint is None or args.events is None:
            print(
                "error: --checkpoint and --events are required unless --fixture is set",
                file=__import__("sys").stderr,
            )
            return 2
        events = load_decision_events_v2(args.events)
        corpus = _materialize_corpus(events, args.materializer)
        base_checkpoint_sha = checkpoint_sha(args.checkpoint)

    if args.admit:
        admit_semantic_corpus(
            corpus,
            materializer_id=args.materializer,
            min_train_support=1,
        )

    matrix: list[dict[str, Any]] = [
        {"rank": rank, "target_modules": args.target_modules} for rank in args.ranks
    ]
    if args.module_restricted:
        matrix.extend(
            {
                "rank": rank,
                "target_modules": args.target_modules,
                "target_layer_indices": (-1,),
            }
            for rank in args.ranks
        )

    def model_factory() -> TwoTowerModel:
        if args.fixture:
            return _fixture_model(device=args.device)
        return TwoTowerModel.from_checkpoint(args.checkpoint, device=args.device)

    def spec_factory(model: TwoTowerModel, cell: dict[str, Any]) -> TwoTowerAdapterSpec:
        return _spec_factory(model, cell, base_checkpoint_sha=base_checkpoint_sha)

    report = profile_adapter_subspace_geometry(
        model_factory,
        corpus,
        spec_factory,
        matrix,
        objective=args.objective,
        budget=DiagnosticBudget(max_wall_minutes=args.budget),
    )
    report["base_checkpoint_sha"] = base_checkpoint_sha
    report["materializer"] = args.materializer
    if not args.fixture and args.events is not None:
        report["events_path"] = str(args.events)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_geometry_report(args.out, report)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
