#!/usr/bin/env python3
"""SLM-238 (RSC-A02): bounded 5-arm recursive-depth-aux-mode factorial.

Compares the five arms the issue specifies -- ``off`` / ``intermediate_only``
(uniform, then lambda=0.3) / ``all_depths`` (uniform, then lambda=0.3) -- on
two bounded recipes:

  1. A deterministic synthetic fixture (the same HERO/CTA records used by
     ``run_slm138_recursive_denoiser_fixture.py`` and
     ``tests/test_models/test_recursive_denoiser.py``).
  2. One bounded real-corpus smoke: the first ``--corpus-limit`` records of
     the committed ``train_seeds.jsonl`` fixture corpus.

This is calibration/semantics work: it validates that the objective
decomposition, telemetry, and gradient behavior are correct and comparable
across modes at small scale. It is explicitly **not** a quality or
LOTUS-transfer claim, and it authorizes no promotion -- see SLM-233 for the
future full control-matrix campaign this is meant to inform.

Example:
  python -m scripts.run_rsc_a02_depth_aux_mode_factorial --mode plan-only
  python -m scripts.run_rsc_a02_depth_aux_mode_factorial --mode fixture
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.versioning import build_version_stamp

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'

# SLM-238 preregistered lambda: chosen (not tuned post-hoc) as a value clearly
# < 1 that keeps the auxiliary term subordinate to the primary reconstruction
# term -- the final recursion depth already receives full weight once via
# the primary term (rec_out["logits"] == rec_out["depth_logits"][-1]), so a
# partial (0.3x) extra credit in "all_depths" mode avoids letting the
# auxiliary term dominate or swamp the primary gradient signal, while still
# being large enough to matter. 0.3 is also a simple, easy-to-reproduce
# round number for a first calibration pass.
PREREGISTERED_LAMBDA = 0.3

ARMS: tuple[dict[str, Any], ...] = (
    {"arm": "A", "mode": "off", "weights": (), "aux_weight": 0.0, "schedule": "n/a"},
    {
        "arm": "B",
        "mode": "intermediate_only",
        "weights": (1.0, 1.0),
        "aux_weight": 1.0,
        "schedule": "uniform_normalized",
    },
    {
        "arm": "C",
        "mode": "all_depths",
        "weights": (1.0, 1.0, 1.0),
        "aux_weight": 1.0,
        "schedule": "uniform_normalized",
    },
    {
        "arm": "D",
        "mode": "intermediate_only",
        "weights": (1.0, 1.0),
        "aux_weight": PREREGISTERED_LAMBDA,
        "schedule": "uniform_normalized",
    },
    {
        "arm": "E",
        "mode": "all_depths",
        "weights": (1.0, 1.0, 1.0),
        "aux_weight": PREREGISTERED_LAMBDA,
        "schedule": "uniform_normalized",
    },
)

RECURSIVE_STEPS = 3  # so intermediate_only covers depths 0..1, all_depths 0..2
TRAIN_STEPS = 3
SEED = 0


def _today_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _fixture_records() -> list[ExampleRecord]:
    return [
        ExampleRecord(id="a", prompt="Hero layout", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA layout", openui=CTA, split="train"),
    ]


def _corpus_records(limit: int) -> list[ExampleRecord]:
    path = Path("src/slm_training/resources/train_seeds.jsonl")
    records = load_jsonl(path)
    return list(records[:limit])


def _quality_pipeline_ok() -> bool:
    """True when validate() accepts a known-good OpenUI snippet.

    Mirrors ``scripts/run_perf_matrix.py``'s ``_quality_pipeline_ok``: some
    sandboxes have a broken official-grammar Node bridge (e.g. an
    incompatible ``NODE_OPTIONS``), which is an environment-provisioning gap,
    not a semantics defect in this issue's scope. The syntax/structure/
    strict-semantic smoke below is best-effort and explicitly annotated as
    vacuous when this is False, rather than silently reporting misleading
    all-fail numbers.
    """
    try:
        from slm_training.dsl.lang_core import validate

        validate('root = Card(":t.x")\n')
        return True
    except Exception:  # noqa: BLE001
        return False


def _build_model(
    *, mode: str, weights: tuple[float, ...], aux_weight: float, seed: int = SEED
) -> TwoTowerModel:
    return TwoTowerModel(
        tokenizer=_shared_tokenizer(),
        config=TwoTowerConfig(
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=2,
            denoiser_arch="shared_recursive",
            recursive_steps=RECURSIVE_STEPS,
            recursive_transition_layers=2,
            recursive_depth_supervision_weights=weights,
            recursive_depth_aux_mode=mode,
            recursive_depth_aux_weight=aux_weight,
            grammar_constrained=False,
            fidelity_loss_weight=0.0,
            gen_steps=2,
            seed=seed,
        ),
        device="cpu",
    )


_TOKENIZER_CACHE: Any = None


def _shared_tokenizer() -> Any:
    """One tokenizer built from the union of both recipes' text, shared
    across every arm so results are directly comparable (identical vocab)."""
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        from slm_training.models.tokenizer import OpenUITokenizer

        texts: list[str] = []
        for r in _fixture_records() + _corpus_records(limit=8):
            texts.extend([r.prompt, r.openui])
        _TOKENIZER_CACHE = OpenUITokenizer.build(texts)
    return _TOKENIZER_CACHE


def _grad_norm(model: TwoTowerModel) -> float:
    total = 0.0
    for p in model.trainable_parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def _train_arm(arm: dict[str, Any], records: list[ExampleRecord]) -> dict[str, Any]:
    import torch

    model = _build_model(
        mode=arm["mode"], weights=arm["weights"], aux_weight=arm["aux_weight"]
    )
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    step_records: list[dict[str, Any]] = []
    for step in range(TRAIN_STEPS):
        opt.zero_grad(set_to_none=True)
        loss = model.training_loss(records)
        loss.backward()
        grad_norm = _grad_norm(model)
        opt.step()
        m = dict(model.last_training_metrics)
        step_records.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "grad_norm": grad_norm,
                "primary_final_reconstruction_loss": m.get(
                    "primary_final_reconstruction_loss"
                ),
                "recursive_depth_supervision_loss": m.get(
                    "recursive_depth_supervision_loss"
                ),
                "recursive_intermediate_aux_loss": m.get(
                    "recursive_intermediate_aux_loss"
                ),
                "recursive_final_depth_aux_contribution": m.get(
                    "recursive_final_depth_aux_contribution"
                ),
                "combined_training_loss": m.get("combined_training_loss"),
            }
        )
    final_metrics = dict(model.last_training_metrics)

    first_contribution = final_metrics.get("recursive_depth_weighted_contribution_0")
    last_depth_idx = RECURSIVE_STEPS - 1
    last_contribution = final_metrics.get(
        f"recursive_depth_weighted_contribution_{last_depth_idx}"
    )
    if (
        first_contribution is not None
        and last_contribution is not None
        and last_contribution != 0.0
    ):
        first_last_ratio: float | str = float(first_contribution / last_contribution)
    elif first_contribution is not None and last_contribution is None:
        first_last_ratio = "n/a (final depth structurally excluded -- intermediate_only)"
    else:
        first_last_ratio = "n/a (aux disabled -- no per-depth contributions)"

    return {
        "arm": arm["arm"],
        "mode": arm["mode"],
        "weights": list(arm["weights"]),
        "aux_weight": arm["aux_weight"],
        "schedule": arm["schedule"],
        "recursive_steps": RECURSIVE_STEPS,
        "param_count": sum(int(p.numel()) for p in model.parameters()),
        "steps": step_records,
        "final_objective_decomposition": {
            "primary_final_reconstruction_loss": final_metrics.get(
                "primary_final_reconstruction_loss"
            ),
            "recursive_intermediate_aux_loss": final_metrics.get(
                "recursive_intermediate_aux_loss"
            ),
            "recursive_final_depth_aux_contribution": final_metrics.get(
                "recursive_final_depth_aux_contribution"
            ),
            "recursive_depth_aux_weight": final_metrics.get(
                "recursive_depth_aux_weight"
            ),
            "recursive_depth_supervision_loss": final_metrics.get(
                "recursive_depth_supervision_loss"
            ),
            "combined_training_loss": final_metrics.get("combined_training_loss"),
        },
        "first_last_depth_contribution_ratio": first_last_ratio,
        "final_train_loss_scale": step_records[-1]["loss"],
        "final_update_grad_norm": step_records[-1]["grad_norm"],
        "_model": model,
    }


def _per_depth_gradient_diagnostics(arm: dict[str, Any]) -> dict[str, Any]:
    """Isolated tower-level gradient norm/cosine per recursion depth.

    Deliberately mirrors the pattern already used by this repo's own unit
    tests (e.g. ``test_gradient_reaches_only_positive_weight_depths``,
    ``test_recursive_forward_shapes_and_gradients``): a small deterministic
    synthetic batch fed directly through ``SharedRecursiveDenoiserTower``,
    independent of the tokenizer/masking/context pipeline, so this diagnostic
    isolates the recursion's own gradient geometry rather than conflating it
    with unrelated training_loss plumbing.
    """
    import torch
    import torch.nn.functional as F

    model = _build_model(
        mode=arm["mode"], weights=arm["weights"], aux_weight=arm["aux_weight"]
    )
    torch.manual_seed(SEED)
    vocab = model.tokenizer.vocab_size
    d_model = 32
    noisy = torch.randint(1, vocab, (2, 8))
    ctx = torch.randn(2, 3, d_model)
    targets = torch.randint(0, vocab, (2, 8))

    out = model.denoiser.recursive_outputs(noisy, ctx, pad_id=model.tokenizer.pad_id)
    depth_logits = out["depth_logits"]
    target_param = model.denoiser.tok.weight

    per_depth: list[dict[str, Any]] = []
    grad_vectors: list[Any] = []
    for d, d_logits in enumerate(depth_logits):
        d_flat = d_logits.reshape(-1, d_logits.size(-1))
        d_ce = F.cross_entropy(d_flat, targets.reshape(-1))
        (g,) = torch.autograd.grad(
            d_ce, target_param, retain_graph=True, allow_unused=True
        )
        if g is None:
            g = torch.zeros_like(target_param)
        grad_vectors.append(g.detach().flatten())
        per_depth.append({"depth": d, "grad_norm": float(g.norm().item())})

    final_vec = grad_vectors[-1]
    final_norm = final_vec.norm().item()
    for d, row in enumerate(per_depth):
        v = grad_vectors[d]
        denom = v.norm().item() * final_norm
        cosine = float((v @ final_vec).item() / denom) if denom > 0 else 0.0
        row["cosine_vs_final_depth"] = cosine

    return {"target_param": "denoiser.tok.weight", "per_depth": per_depth}


def _smoke_generation_check(arm_result: dict[str, Any], records: list[ExampleRecord]) -> dict[str, Any]:
    """Bounded syntax/structure/strict-semantic smoke -- safety diagnostics
    only, never a quality claim (models here train for 3 steps on <=8
    records)."""
    model: TwoTowerModel = arm_result["_model"]
    bridge_ok = _quality_pipeline_ok()
    rows: list[dict[str, Any]] = []
    if bridge_ok:
        from slm_training.evals.meaningful_program import binding_aware_meaningful_v2

        for record in records[:3]:
            try:
                pred = model.generate(record.prompt, gold=record)
                report = binding_aware_meaningful_v2(pred, record=record)
                rows.append(
                    {
                        "id": record.id,
                        "verdict": bool(report.verdict),
                        "reason_codes": list(report.reason_codes),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append({"id": record.id, "verdict": False, "error": str(exc)[:200]})
    return {
        "bridge_healthy": bridge_ok,
        "note": (
            "safety diagnostic only -- NOT a quality/promotion claim; models "
            f"here train for {TRAIN_STEPS} steps on <=8 records"
            if bridge_ok
            else "openui grammar bridge unavailable in this environment "
            "(e.g. a NODE_OPTIONS incompatibility) -- syntax/structure/"
            "semantic smoke skipped/vacuous here, not because of this "
            "issue's changes; rerun in a bridge-healthy environment for a "
            "real read"
        ),
        "rows": rows,
    }


def _run_factorial(corpus_limit: int) -> dict[str, Any]:
    fixture_records = _fixture_records()
    corpus_records = _corpus_records(limit=corpus_limit)

    arms_report: list[dict[str, Any]] = []
    for arm in ARMS:
        fixture_result = _train_arm(arm, fixture_records)
        depth_grad = _per_depth_gradient_diagnostics(arm)
        smoke = _smoke_generation_check(fixture_result, fixture_records)

        corpus_arm = _train_arm(arm, corpus_records)
        corpus_result = {
            k: v
            for k, v in corpus_arm.items()
            if k != "_model"
        }

        fixture_public = {k: v for k, v in fixture_result.items() if k != "_model"}
        arms_report.append(
            {
                "arm": arm["arm"],
                "mode": arm["mode"],
                "aux_weight": arm["aux_weight"],
                "schedule": arm["schedule"],
                "deterministic_fixture": fixture_public,
                "per_depth_gradient_diagnostics": depth_grad,
                "bounded_smoke_safety_diagnostics": smoke,
                "bounded_real_corpus_smoke": corpus_result,
            }
        )

    return {
        "matrix_set": "rsc-a02-depth-aux-mode-factorial",
        "matrix_version": "rsc-a02-v1",
        "run_id": "rsc_a02_depth_aux_mode_factorial",
        "issue": "SLM-238 (RSC-A02)",
        "status": "calibration_only",
        "claim_class": "semantics_calibration",
        "quality_claim_made": False,
        "preregistered_lambda": PREREGISTERED_LAMBDA,
        "preregistered_lambda_justification": (
            "0.3 (< 1) is chosen a priori, not tuned post-hoc: the final "
            "recursion depth already receives full weight once via the "
            "primary reconstruction term (rec_out['logits'] == "
            "rec_out['depth_logits'][-1]), so a partial (0.3x) extra credit "
            "in all_depths mode avoids letting the auxiliary term dominate "
            "or swamp the primary gradient signal, while remaining large "
            "enough to matter. It is also a simple, easy-to-reproduce round "
            "number for a first calibration pass."
        ),
        "arms": arms_report,
        "params_and_flops_unchanged_across_arms": True,
        "note": (
            "Bounded calibration/semantics factorial only (deterministic "
            f"{len(fixture_records)}-record fixture + a {corpus_limit}-record "
            "real-corpus smoke, 3 training steps each). No promotion or "
            "broad GPU campaign; no quality or LOTUS-transfer claim is made. "
            "See docs/design/iter-rsc-a02-*.md for the recommended semantic "
            "mode and the SLM-233 control-matrix recommendation."
        ),
        "version_stamp": build_version_stamp("model.recursive_denoiser"),
    }


def _plan_only_report() -> dict[str, Any]:
    return {
        "matrix_set": "rsc-a02-depth-aux-mode-factorial",
        "matrix_version": "rsc-a02-v1",
        "run_id": "rsc_a02_depth_aux_mode_plan",
        "status": "plan_only",
        "claim_class": "semantics_calibration",
        "arms": [
            {k: v for k, v in arm.items()} for arm in ARMS
        ],
        "note": "plan-only: no models instantiated or trained",
        "version_stamp": build_version_stamp("model.recursive_denoiser"),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# SLM-238 (RSC-A02): depth-aux-mode factorial ({report.get('run_id')})",
        "",
        f"Matrix set: `{report.get('matrix_set')}`  ",
        f"Version: `{report.get('matrix_version')}`  ",
        f"Status: **{report.get('status')}**",
        "",
        "**No quality or LOTUS-transfer claim is made from this factorial** "
        "-- it is calibration/semantics work only, ahead of a future SLM-233 "
        "control-matrix campaign.",
        "",
    ]
    if "preregistered_lambda" in report:
        lines.extend(
            [
                f"Preregistered lambda: `{report['preregistered_lambda']}`",
                "",
                report.get("preregistered_lambda_justification", ""),
                "",
            ]
        )
    for arm in report.get("arms", []):
        lines.append(
            f"## Arm {arm['arm']}: mode=`{arm['mode']}` "
            f"aux_weight=`{arm['aux_weight']}` schedule=`{arm['schedule']}`"
        )
        lines.append("")
        fixture = arm.get("deterministic_fixture")
        if fixture:
            dec = fixture["final_objective_decomposition"]
            lines.extend(
                [
                    "Deterministic fixture (final step):",
                    "",
                    f"- primary_final_reconstruction_loss: `{dec['primary_final_reconstruction_loss']:.6f}`",
                    f"- recursive_intermediate_aux_loss: `{dec['recursive_intermediate_aux_loss']:.6f}`",
                    f"- recursive_final_depth_aux_contribution: `{dec['recursive_final_depth_aux_contribution']:.6f}`",
                    f"- recursive_depth_aux_weight: `{dec['recursive_depth_aux_weight']}`",
                    f"- recursive_depth_supervision_loss: `{dec['recursive_depth_supervision_loss']:.6f}`",
                    f"- combined_training_loss: `{dec['combined_training_loss']:.6f}`",
                    f"- first/last-depth contribution ratio: `{fixture['first_last_depth_contribution_ratio']}`",
                    f"- final update grad norm: `{fixture['final_update_grad_norm']:.6f}`",
                    "",
                ]
            )
        depth_grad = arm.get("per_depth_gradient_diagnostics")
        if depth_grad:
            lines.append("Per-depth gradient diagnostics (synthetic isolated tower batch):")
            lines.append("")
            for row in depth_grad["per_depth"]:
                lines.append(
                    f"- depth {row['depth']}: grad_norm=`{row['grad_norm']:.6f}` "
                    f"cosine_vs_final_depth=`{row['cosine_vs_final_depth']:.4f}`"
                )
            lines.append("")
        smoke = arm.get("bounded_smoke_safety_diagnostics")
        if smoke:
            lines.append(f"Bounded safety smoke (bridge_healthy={smoke['bridge_healthy']}): {smoke['note']}")
            lines.append("")
        corpus = arm.get("bounded_real_corpus_smoke")
        if corpus:
            dec = corpus["final_objective_decomposition"]
            lines.extend(
                [
                    "Bounded real-corpus smoke (final step):",
                    "",
                    f"- combined_training_loss: `{dec['combined_training_loss']:.6f}`",
                    f"- final update grad norm: `{corpus['final_update_grad_norm']:.6f}`",
                    "",
                ]
            )
    lines.extend(["## Caveat", "", report.get("note", ""), ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-238 (RSC-A02) bounded 5-arm depth-aux-mode factorial"
    )
    parser.add_argument(
        "--mode",
        choices=("plan-only", "fixture"),
        default="plan-only",
        help="plan-only emits the arm skeleton; fixture runs all 5 arms",
    )
    parser.add_argument(
        "--corpus-limit",
        type=int,
        default=8,
        help="bounded real-corpus smoke record count (train_seeds.jsonl prefix)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/runs/rsc-a02-depth-aux-mode-factorial-{_today_slug()}"),
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    design_json = Path(f"docs/design/iter-rsc-a02-depth-aux-mode-factorial-{_today_slug()}.json")
    design_md = Path(f"docs/design/iter-rsc-a02-depth-aux-mode-factorial-{_today_slug()}.md")

    report = (
        _plan_only_report()
        if args.mode == "plan-only"
        else _run_factorial(corpus_limit=args.corpus_limit)
    )

    report_text = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    report_path = output_dir / "rsc_a02_depth_aux_mode_factorial_report.json"
    report_path.write_text(report_text, encoding="utf-8")
    markdown = _render_markdown(report)
    (output_dir / "rsc_a02_depth_aux_mode_factorial_report.md").write_text(
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
