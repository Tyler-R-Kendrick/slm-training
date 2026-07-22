"""SLM-108 external 1-7B constrained-decoding semantic ceiling matrix.

This module defines the experiment arms, manifest schema, fixture wiring, and
report rendering for comparing an off-the-shelf HuggingFace causal/instruct
model against the tiny SLM under matched compiler-owned constraints.

The fixture path is torch-free and uses a deterministic fake scorer. Frontier
(model-backed) rows are fully specified but left ``not_run`` unless a GPU host
and pinned checkpoints are available. No ship claim is made by the fixture.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.data.contract import GenerationRequest
from slm_training.evals.score_policy import CandidatePath
from slm_training.models.external_scorer import (
    ExternalLegalActionScorer,
    ExternalScorerConfig,
    ExternalScorePolicy,
    FakeExternalScorer,
    build_external_scorer,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "ExternalCeilingArm",
    "ExternalCeilingManifest",
    "ExternalCeilingReport",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "build_external_ceiling_manifest",
    "render_markdown",
    "run_fixture_matrix",
    "validate_external_ceiling_manifest",
]

MATRIX_VERSION = "efs1-01-v1"
MATRIX_SET = "external-ceiling"


@dataclass(frozen=True)
class CostProfile:
    parameters: int | None = None
    loaded_bytes: int | None = None
    quantization: str | None = None
    peak_memory_bytes: int | None = None
    latency_p50_ms: float | None = None
    candidates_scored: int = 0
    verifier_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class ExternalCeilingArm:
    """One matched arm of the external semantic-ceiling experiment."""

    arm_id: str
    description: str
    model_id: str
    revision: str
    decode_mode: str  # constrained | unconstrained | complete_rerank
    scorer_kind: str = "transformers_causal_lm"
    device: str = "cpu"
    dtype: str = "float32"
    claim_class: str = "fixture"
    status: str = "not_run"  # not_run | fixture | frontier
    primary_metric: str = "binding_aware_meaningful_v2_rate_strict"
    cost: CostProfile = field(default_factory=CostProfile)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["cost"] = self.cost.to_dict()
        return data

    def scorer_config(self) -> ExternalScorerConfig:
        return ExternalScorerConfig(
            model_id=self.model_id,
            revision=self.revision,
            device=self.device,
            dtype=self.dtype,
            claim_class=self.claim_class,
        )


@dataclass(frozen=True)
class ExternalCeilingManifest:
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    hypothesis: str = (
        "A capable external 1-7B model constrained by the same compiler exceeds "
        "the tiny SLM on binding-aware meaningful-program rate."
    )
    primary_metric: str = "binding_aware_meaningful_v2_rate_strict"
    suites: tuple[str, ...] = ("smoke", "held_out", "adversarial")
    arms: list[ExternalCeilingArm] = field(default_factory=list)
    tiny_slm_run_id: str | None = None
    checkpoint_reference_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["arms"] = [arm.to_dict() for arm in self.arms]
        return data

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class ArmResult:
    arm_id: str
    status: str
    records_evaluated: int = 0
    parse_rate: float | None = None
    binding_aware_meaningful_v2_rate_strict: float | None = None
    agentv_score: float | None = None
    whole_contract_pass: float | None = None
    non_empty_rate: float | None = None
    fallback_rate: float | None = None
    oom_or_timeout: int = 0
    cost: CostProfile = field(default_factory=CostProfile)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["cost"] = self.cost.to_dict()
        return data


@dataclass(frozen=True)
class ExternalCeilingReport:
    matrix_set: str
    matrix_version: str
    run_id: str
    status: str  # fixture | frontier | partial
    manifest: ExternalCeilingManifest
    results: list[ArmResult]
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "run_id": self.run_id,
            "status": self.status,
            "manifest": self.manifest.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )


def build_external_ceiling_manifest(
    *,
    tiny_slm_run_id: str | None = None,
    checkpoint_reference_uri: str | None = None,
) -> ExternalCeilingManifest:
    """Return the default SLM-108 arm manifest.

    Arms:
      A: tiny SLM constrained baseline (reference run, not executed here).
      B: external 1-2B constrained.
      C: external 6-7B constrained (hardware permitting).
      D: same external model as B, unconstrained + postvalidation.
      E: external complete-candidate rerank diagnostic.
    """
    arms = [
        ExternalCeilingArm(
            arm_id="A",
            description="Tiny SLM canonical constrained path",
            model_id="slm-training/twotower",
            revision="latest",
            decode_mode="constrained",
            claim_class="frontier",
            status="not_run",
        ),
        ExternalCeilingArm(
            arm_id="B",
            description="External 1-2B constrained (HuggingFaceTB/SmolLM2-135M wiring fixture)",
            model_id="HuggingFaceTB/SmolLM2-135M",
            revision="main",
            decode_mode="constrained",
            claim_class="fixture",
            status="fixture",
        ),
        ExternalCeilingArm(
            arm_id="C",
            description="External 6-7B constrained (blocked pending GPU/host allocation)",
            model_id="Qwen/Qwen2.5-7B-Instruct",
            revision="main",
            decode_mode="constrained",
            claim_class="frontier",
            status="not_run",
        ),
        ExternalCeilingArm(
            arm_id="D",
            description="External 1-2B unconstrained + postvalidation",
            model_id="HuggingFaceTB/SmolLM2-135M",
            revision="main",
            decode_mode="unconstrained",
            claim_class="fixture",
            status="fixture",
        ),
        ExternalCeilingArm(
            arm_id="E",
            description="External complete-candidate rerank diagnostic",
            model_id="HuggingFaceTB/SmolLM2-135M",
            revision="main",
            decode_mode="complete_rerank",
            claim_class="fixture",
            status="fixture",
        ),
    ]
    return ExternalCeilingManifest(
        arms=arms,
        tiny_slm_run_id=tiny_slm_run_id,
        checkpoint_reference_uri=checkpoint_reference_uri,
    )


def validate_external_ceiling_manifest(manifest: ExternalCeilingManifest) -> list[str]:
    errors: list[str] = []
    ids = {arm.arm_id for arm in manifest.arms}
    if len(ids) != len(manifest.arms):
        errors.append("duplicate arm_id")
    required = {"A", "B"}
    missing = required - ids
    if missing:
        errors.append(f"missing required arms: {sorted(missing)}")
    for arm in manifest.arms:
        if arm.claim_class in {"frontier", "ship_candidate"} and not manifest.checkpoint_reference_uri:
            errors.append(
                f"arm {arm.arm_id}: frontier/ship_candidate rows require a checkpoint_reference_uri"
            )
    return errors


def _build_scorer(arm: ExternalCeilingArm) -> ExternalLegalActionScorer:
    if arm.status == "fixture":
        return FakeExternalScorer(arm.scorer_config())
    return build_external_scorer(arm.scorer_config(), kind=arm.scorer_kind)


def run_fixture_matrix(
    manifest: ExternalCeilingManifest,
    requests: list[GenerationRequest] | None = None,
    *,
    run_id: str = "slm108_fixture",
    output_dir: Path | None = None,
) -> ExternalCeilingReport:
    """Run a CPU/torch-free fixture over a small synthetic request set."""
    if requests is None:
        requests = _fixture_requests()
    results: list[ArmResult] = []
    t0 = time.perf_counter()
    for arm in manifest.arms:
        if arm.status != "fixture":
            results.append(
                ArmResult(
                    arm_id=arm.arm_id,
                    status=arm.status,
                    notes=["skipped: not a fixture arm"],
                )
            )
            continue
        scorer = _build_scorer(arm)
        policy = ExternalScorePolicy(
            scorer=scorer,
            request=requests[0],
            prefix_text="",
            name=f"external_policy_{arm.arm_id}",
        )
        # Score a small set of candidate paths to prove the wiring.
        candidates = [
            CandidatePath(
                candidate_id=f"c{i}",
                token_ids=tuple(range(10 + i, 15 + i)),
                log_probs=tuple(0.0 for _ in range(5)),
            )
            for i in range(min(4, len(requests)))
        ]
        ranked = sorted(
            candidates,
            key=lambda c: policy.score(c),
            reverse=True,
        )
        diag = scorer.diagnostics()
        results.append(
            ArmResult(
                arm_id=arm.arm_id,
                status="fixture",
                records_evaluated=len(requests),
                binding_aware_meaningful_v2_rate_strict=0.0,
                non_empty_rate=1.0 if ranked else 0.0,
                fallback_rate=0.0,
                cost=CostProfile(
                    candidates_scored=len(candidates),
                    latency_p50_ms=(time.perf_counter() - t0) * 1000.0,
                ),
                notes=[
                    f"ranked {len(ranked)} candidates",
                    f"scorer diagnostics: {diag}",
                    "fixture-only: no actual 1-7B model was loaded",
                ],
            )
        )
    report = ExternalCeilingReport(
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        results=results,
        version_stamp=build_version_stamp(
            "harness.experiments",
        ),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "external_ceiling_report.json")
    return report


def render_markdown(report: ExternalCeilingReport) -> str:
    lines = [
        f"# External constrained-decoding semantic ceiling ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`  ",
        f"Version: `{report.matrix_version}`  ",
        f"Status: **{report.status}**  ",
        "",
        "## Manifest",
        "",
        f"Hypothesis: {report.manifest.hypothesis}",
        "",
        "| Arm | Model | Decode | Status | Claim |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in report.manifest.arms:
        lines.append(
            f"| {arm.arm_id} | {arm.model_id} | {arm.decode_mode} | "
            f"{arm.status} | {arm.claim_class} |"
        )
    lines.extend(["", "## Results", ""])
    for result in report.results:
        lines.append(f"### Arm {result.arm_id} ({result.status})")
        lines.append(f"- records evaluated: {result.records_evaluated}")
        lines.append(
            f"- binding-aware meaningful v2 (strict): {result.binding_aware_meaningful_v2_rate_strict}"
        )
        lines.append(f"- non-empty rate: {result.non_empty_rate}")
        lines.append(f"- candidates scored: {result.cost.candidates_scored}")
        for note in result.notes:
            lines.append(f"- {note}")
        lines.append("")
    lines.extend(
        [
            "## Verdict",
            "",
            "Fixture wiring only. No 1-7B model was loaded and no ship claim is made. "
            "Frontier arms require GPU access and pinned durable checkpoints per SLM-103.",
            "",
        ]
    )
    return "\n".join(lines)


def _fixture_requests() -> list[GenerationRequest]:
    return [
        GenerationRequest(
            prompt="Create a screen with a title and a button that submits a form.",
            slot_contract=(":slot_0", ":slot_1"),
        ),
        GenerationRequest(
            prompt="Create a login form with username and password fields.",
            slot_contract=(":slot_0", ":slot_1"),
        ),
    ]
