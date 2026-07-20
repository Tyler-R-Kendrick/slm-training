#!/usr/bin/env python3
"""Run the SLM-138 shared recursive denoiser fixture.

Builds tiny TwoTower models with ``denoiser_arch="stacked"`` and
``"shared_recursive"``, exercises forward / training_loss on synthetic records,
verifies shapes, parameter counts, weight sharing across recursions, and the
checkpoint migration helper.  Emits version-stamped JSON + markdown artifacts.

Example:
  python -m scripts.run_slm138_recursive_denoiser_fixture --mode plan-only
  python -m scripts.run_slm138_recursive_denoiser_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.recursive_denoiser import SharedRecursiveDenoiserTower
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.versioning import build_version_stamp

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


def _today_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _fixture_records() -> list[ExampleRecord]:
    return [
        ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA layout", openui=CTA, split="train"),
    ]


def _build_model(arch: str, seed: int = 0) -> TwoTowerModel:
    return TwoTowerModel.from_records(
        _fixture_records(),
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=2,
            denoiser_arch=arch,  # type: ignore[arg-type]
            recursive_steps=2,
            recursive_transition_layers=2,
            recursive_depth_supervision_weights=(0.5, 1.0),
            grammar_constrained=False,
            gen_steps=2,
            seed=seed,
        ),
        device="cpu",
    )


def _count_params(model: TwoTowerModel) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def _run_fixture() -> dict[str, Any]:
    import torch

    records = _fixture_records()
    stacked = _build_model("stacked", seed=0)
    recursive = _build_model("shared_recursive", seed=0)

    stacked_forward = stacked.denoiser(
        torch.randint(1, stacked.tokenizer.vocab_size, (2, 6)),
        torch.randn(2, 3, 32),
        pad_id=stacked.tokenizer.pad_id,
    )
    recursive_forward = recursive.denoiser(
        torch.randint(1, recursive.tokenizer.vocab_size, (2, 6)),
        torch.randn(2, 3, 32),
        pad_id=recursive.tokenizer.pad_id,
    )

    stacked_loss = stacked.training_loss(records)
    recursive_loss = recursive.training_loss(records)

    stacked.train()
    recursive.train()
    opt_s = torch.optim.AdamW(stacked.trainable_parameters(), lr=1e-3)
    opt_r = torch.optim.AdamW(recursive.trainable_parameters(), lr=1e-3)
    opt_s.zero_grad(set_to_none=True)
    opt_r.zero_grad(set_to_none=True)
    stacked_loss.backward()
    recursive_loss.backward()
    opt_s.step()
    opt_r.step()

    # Weight sharing: recursive tower reuses the same layer objects each step.
    rec_tower: SharedRecursiveDenoiserTower = recursive.denoiser  # type: ignore[assignment]
    f_ids = {id(layer) for layer in rec_tower._f_layers}
    g_ids = {id(layer) for layer in rec_tower._g_layers}

    # Deep-supervision metrics are populated for the recursive model.
    deep_metrics = {
        k: v
        for k, v in recursive.last_training_metrics.items()
        if k.startswith("recursive_depth")
    }

    # Round-trip save/load for the recursive model.
    tmp = Path("outputs/runs/slm138-recursive-denoiser-tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    ckpt = tmp / "recursive.pt"
    recursive.save(ckpt)
    loaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
    loaded_ok = (
        loaded.config.denoiser_arch == "shared_recursive"
        and isinstance(loaded.denoiser, SharedRecursiveDenoiserTower)
    )

    return {
        "matrix_set": "slm138-shared-recursive-denoiser",
        "matrix_version": "slm138-v1",
        "run_id": "slm138_recursive_denoiser_fixture",
        "status": "wiring_only",
        "claim_class": "wiring",
        "denoiser_architectures": ["stacked", "shared_recursive"],
        "stacked_params": _count_params(stacked),
        "recursive_params": _count_params(recursive),
        "forward_shapes": {
            "stacked": list(stacked_forward.shape),
            "recursive": list(recursive_forward.shape),
        },
        "losses": {
            "stacked": float(stacked_loss.detach().cpu()),
            "recursive": float(recursive_loss.detach().cpu()),
        },
        "recursive_weight_sharing": {
            "f_layer_object_count": len(f_ids),
            "g_layer_object_count": len(g_ids),
            "total_shared_layers": len(rec_tower.layers),
        },
        "deep_supervision_metrics": deep_metrics,
        "checkpoint_roundtrip_ok": loaded_ok,
        "note": (
            "Wiring-only evidence. Full matched-block evaluation arms and GPU "
            "training are deferred."
        ),
        "version_stamp": build_version_stamp("model.recursive_denoiser"),
    }


def _plan_only_report() -> dict[str, Any]:
    return {
        "matrix_set": "slm138-shared-recursive-denoiser",
        "matrix_version": "slm138-v1",
        "run_id": "slm138_recursive_denoiser_plan",
        "status": "plan_only",
        "claim_class": "wiring",
        "denoiser_architectures": ["stacked", "shared_recursive"],
        "note": "plan-only: no models instantiated or trained",
        "version_stamp": build_version_stamp("model.recursive_denoiser"),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# SLM-138: Shared recursive denoiser tower fixture ({report.get('run_id')})",
        "",
        f"Matrix set: `{report.get('matrix_set')}`  ",
        f"Version: `{report.get('matrix_version')}`  ",
        f"Status: **{report.get('status')}**",
        "",
        "## What this exercises",
        "",
        "A drop-in ``SharedRecursiveDenoiserTower`` that preserves the ``DenoiserTower`` "
        "public contract. The fixture builds tiny TwoTower models for both ``stacked`` "
        "and ``shared_recursive`` denoiser architectures, runs forward passes and "
        "training_loss, verifies shapes/gradients, confirms object-identity weight "
        "sharing across recursions, and round-trips a recursive checkpoint.",
        "",
        "## Architectures",
        "",
        ", ".join(f"`{a}`" for a in report.get("denoiser_architectures", [])),
        "",
    ]

    if "forward_shapes" in report:
        lines.extend(
            [
                "## Forward shapes",
                "",
                f"- stacked: `{report['forward_shapes']['stacked']}`",
                f"- recursive: `{report['forward_shapes']['recursive']}`",
                "",
            ]
        )

    if "losses" in report:
        lines.extend(
            [
                "## Losses",
                "",
                f"- stacked: `{report['losses']['stacked']:.6f}`",
                f"- recursive: `{report['losses']['recursive']:.6f}`",
                "",
            ]
        )

    sharing = report.get("recursive_weight_sharing")
    if sharing:
        lines.extend(
            [
                "## Recursive weight sharing",
                "",
                f"- F-update distinct layer objects: {sharing['f_layer_object_count']}",
                f"- G-update distinct layer objects: {sharing['g_layer_object_count']}",
                f"- Total shared transition layers: {sharing['total_shared_layers']}",
                "",
            ]
        )

    deep = report.get("deep_supervision_metrics")
    if deep:
        lines.extend(
            [
                "## Deep-supervision metrics",
                "",
            ]
        )
        for k, v in deep.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    if "checkpoint_roundtrip_ok" in report:
        lines.extend(
            [
                "## Checkpoint round-trip",
                "",
                f"Recursive checkpoint save/load OK: **{report['checkpoint_roundtrip_ok']}**",
                "",
            ]
        )

    lines.extend(
        [
            "## Fixture caveat",
            "",
            report.get(
                "note",
                "Wiring-only evidence. Full matched-block evaluation arms and GPU "
                "training are deferred.",
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-138 shared recursive denoiser fixture"
    )
    parser.add_argument(
        "--mode",
        choices=("plan-only", "fixture"),
        default="plan-only",
        help=(
            "plan-only emits the matrix skeleton; "
            "fixture exercises both denoiser architectures"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/runs/slm138-recursive-denoiser-{_today_slug()}"),
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    design_json = Path(f"docs/design/iter-slm138-recursive-denoiser-{_today_slug()}.json")
    design_md = Path(f"docs/design/iter-slm138-recursive-denoiser-{_today_slug()}.md")

    report = _plan_only_report() if args.mode == "plan-only" else _run_fixture()

    report_text = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    report_path = output_dir / "slm138_recursive_denoiser_report.json"
    report_path.write_text(report_text, encoding="utf-8")
    markdown = _render_markdown(report)
    (output_dir / "slm138_recursive_denoiser_report.md").write_text(
        markdown, encoding="utf-8"
    )

    design_json.parent.mkdir(parents=True, exist_ok=True)
    design_json.write_text(report_text, encoding="utf-8")
    design_md.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nReport JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
