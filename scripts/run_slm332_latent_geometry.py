#!/usr/bin/env python3
"""Run SLM-332's bounded AP-009 program-latent geometry comparison locally.

The result is intentionally diagnostic and non-promotable: AP-030 has no
verified latent-to-program decoder, so strict semantic and interpolation
claims remain fail-closed rather than being replaced by factor proxies.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import torch

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.cap2_rate_distortion_sweep import FactorReconstructionModel
from slm_training.harnesses.experiments.slm332_latent_geometry import (
    GeometryTrainingConfig,
    load_admitted_binding_contrasts,
    measure_geometry_arm,
    train_geometry_arm,
)
from slm_training.models.continuous_latent import ContinuousLatentCodec, ContinuousLatentConfig
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "docs/design/iter-slm332-latent-geometry-20260725.json"
DEFAULT_MARKDOWN = ROOT / "docs/design/iter-slm332-latent-geometry-20260725.md"
DEFAULT_AGENTV = ROOT / "docs/design/iter-slm332-latent-geometry-agentv-20260725"
COMPONENT = "harness.experiments.slm332_latent_geometry"


def _portable(value: Any, output_dir: Path) -> Any:
    prefix = str(output_dir.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "agentv-dir://" + value[len(prefix) :].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    """Keep committed AgentV evidence independent of this worktree location."""
    replacements = {
        str(output_dir.resolve()): "agentv-dir://",
        quote(str(output_dir.resolve()), safe=""): quote("agentv-dir://", safe=""),
    }
    for path in (output_dir / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for source, replacement in replacements.items():
            text = text.replace(source, replacement)
        path.write_text(text, encoding="utf-8")


def _model(feature_dim: int, *, seed: int) -> FactorReconstructionModel:
    torch.manual_seed(seed)
    codec = ContinuousLatentCodec(
        ContinuousLatentConfig(
            num_states=1,
            latent_dim=8,
            hidden_dim=32,
            feature_dim=feature_dim,
            mode="semantic_trace",
        )
    )
    return FactorReconstructionModel(codec, feature_dim)


def run(*, steps: int, seed: int, agentv_dir: Path) -> dict[str, Any]:
    """Compare matched reconstruction-only and combined geometry arms locally."""
    corpus = load_admitted_binding_contrasts(ROOT / "src/slm_training/resources/data/eval/openui_hard_valid_v1/pairs.jsonl")
    control_config = GeometryTrainingConfig(steps=steps, contrast_weight=0.0, noise_weight=0.0)
    combined_config = GeometryTrainingConfig(steps=steps)
    control = _model(corpus.positive.shape[-1], seed=seed)
    combined = _model(corpus.positive.shape[-1], seed=seed)
    train_geometry_arm(control, corpus, control_config)
    train_geometry_arm(combined, corpus, combined_config)
    control_measurement = measure_geometry_arm(control, corpus, control_config)
    combined_measurement = measure_geometry_arm(combined, corpus, combined_config)
    stamp = build_version_stamp("harness.experiments", COMPONENT)
    gate_unavailable = (
        combined_measurement.ap030_oracle_semantic_gate
        == "unavailable_no_verified_latent_to_program_decoder"
    )
    report: dict[str, Any] = {
        "schema": "slm332_latent_geometry/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recipe": {
            "device": "cpu",
            "backend": "local",
            "seed": seed,
            "steps": steps,
            "pairs": corpus.positive.shape[0],
            "pair_transforms": sorted(set(corpus.transform_ids)),
            "matched_initialization": True,
        },
        "control": asdict(control_measurement),
        "combined": asdict(combined_measurement),
        "comparison": {
            "contrast_distance_delta": combined_measurement.contrast_distance - control_measurement.contrast_distance,
            "reconstruction_mse_delta": combined_measurement.reconstruction_mse - control_measurement.reconstruction_mse,
            "within_noise_mse_delta": combined_measurement.within_noise_mse - control_measurement.within_noise_mse,
        },
        "strict_semantics": {
            "status": "unavailable_fail_closed",
            "reason": "no_verified_latent_to_program_decoder",
            "ap030_factor_proxy_is_not_substituted": True,
            "promotion_eligible": False,
        },
        "interpolation": {
            "status": "unavailable_fail_closed",
            "reason": "no_verified_latent_to_program_decoder",
            "required_when_available": ["decoded_program", "verifier_verdict", "factor_fingerprint", "boundary_crossing"],
        },
        "version_stamp": stamp,
    }
    published = publish_agentv_evaluation(
            agentv_dir,
            name="slm332-latent-geometry",
            claim="local_codec_only_geometry_diagnostic_not_ship",
            cases=[{
                "id": "semantic_gate_remains_fail_closed",
                "criteria": "The local factor-only diagnostic must not claim strict semantic validation or promotion.",
                "assertions": [{
                    "id": "non_promotable_without_decoder",
                    "actual": gate_unavailable and not combined_measurement.promotion_eligible,
                    "operator": "eq",
                    "expected": True,
                }],
                "result": report["strict_semantics"],
            }],
            version_stamp=stamp,
    )
    _rewrite_agentv_paths(agentv_dir)
    report["agentv"] = _portable(published, agentv_dir)
    return report


def _markdown(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    recipe = report["recipe"]
    return f"""# SLM-332 local latent-geometry diagnostic\n\nThis bounded local run compares a reconstruction-only control with the combined\nreconstruction, hard-valid binding contrast, and local-noise objective. It is\n**not a ship result**.\n\n| arm | contrast distance | reconstruction MSE | promotion |\n| --- | ---: | ---: | --- |\n| control | {report['control']['contrast_distance']:.6f} | {report['control']['reconstruction_mse']:.6f} | no |\n| combined | {report['combined']['contrast_distance']:.6f} | {report['combined']['reconstruction_mse']:.6f} | no |\n\n- Local CPU; {recipe['pairs']} admitted AP-009 binding/reference pairs; seed {recipe['seed']}; {recipe['steps']} steps.\n- Combined minus control contrast distance: {comparison['contrast_distance_delta']:.6f}.\n- AP-030's factor proxy is not a semantic oracle. There is no verified\n  latent-to-program decoder, so no interpolation traversal or strict semantic\n  claim was produced and the result is fail-closed/non-promotable.\n- AgentEvals/AgentV records the non-promotion assertion alongside the JSON.\n"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=332)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--agentv-dir", type=Path, default=DEFAULT_AGENTV)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    report = run(steps=args.steps, seed=args.seed, agentv_dir=args.agentv_dir)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.json_out), "promotion_eligible": False}, sort_keys=True))


if __name__ == "__main__":
    main()
