"""SLM-108 external 1-7B constrained-decoding semantic ceiling matrix.

This module defines the experiment arms, manifest schema, fixture wiring, and
report rendering for comparing an off-the-shelf HuggingFace causal/instruct
model against the tiny SLM under matched compiler-owned constraints.

The fixture path is torch-free and uses a deterministic fake scorer. Frontier
(model-backed) rows are fully specified but left ``not_run`` unless a GPU host
and pinned checkpoints are available. No ship claim is made by the fixture.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
    "DEFAULT_FROZEN_REQUESTS_FILE",
    "DEFAULT_FROZEN_REQUESTS_SHA256",
    "FRONTIER_ARMS",
    "FRONTIER_SYSTEM_PROMPT",
    "ExternalCeilingArm",
    "ExternalCeilingManifest",
    "ExternalCeilingReport",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "build_external_ceiling_manifest",
    "frontier_prompt_template_sha256",
    "load_frozen_requests",
    "render_markdown",
    "run_fixture_matrix",
    "run_frontier_chunk",
    "run_score_pass",
    "validate_external_ceiling_manifest",
]

MATRIX_VERSION = "efs1-01-v1"
MATRIX_SET = "external-ceiling"

# SLM-294 frontier execution constants (preregistration-locked).
FRONTIER_SYSTEM_PROMPT = (
    "Return only one valid OpenUI program using placeholder content."
)
FRONTIER_ARMS = {
    "B": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "C": "Qwen/Qwen2.5-7B-Instruct",
}
DEFAULT_FROZEN_REQUESTS_FILE = Path(
    "src/slm_training/resources/evals/slm294_frozen_requests_v1.jsonl"
)
DEFAULT_FROZEN_REQUESTS_SHA256 = (
    "8466b402452c055a794057eb5a1853fb8e1548a398949cdb35ce51bca0e2d50e"
)


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
            slot_contract=(":title", ":submit_label"),
        ),
        GenerationRequest(
            prompt="Create a login form with username and password fields.",
            slot_contract=(":username", ":password"),
        ),
    ]


# ---------------------------------------------------------------------------
# SLM-294 frontier execution (chunked, resumable, deterministic per chunk)
# ---------------------------------------------------------------------------


def frontier_prompt_template_sha256() -> str:
    """Content hash of the published prompt template (arms B and C)."""
    template = {
        "system": FRONTIER_SYSTEM_PROMPT,
        "user": "{prompt}",
    }
    return hashlib.sha256(
        json.dumps(template, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_frozen_requests(
    path: Path = DEFAULT_FROZEN_REQUESTS_FILE,
    *,
    expected_sha256: str = DEFAULT_FROZEN_REQUESTS_SHA256,
) -> list[Any]:
    """Load frozen SLM-294 requests as ExampleRecords; fail closed on sha mismatch."""
    from slm_training.dsl.schema import ExampleRecord

    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"frozen requests sha256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual}"
        )
    records: list[Any] = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records.append(
            ExampleRecord(
                id=row["id"],
                prompt=row["prompt"],
                openui=row["openui"],
                placeholders=list(row.get("placeholders", [])),
                split=row.get("split", "held_out"),
                source=row.get("source", "slm294_frozen"),
            )
        )
    return records


def _append_jsonl_atomic(path: Path, record: dict[str, Any]) -> None:
    """Append one JSONL record via tmp-file + os.replace (never partial)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        existing + json.dumps(record, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_frontier_chunk(
    *,
    arm: str,
    output_dir: Path,
    requests_file: Path = DEFAULT_FROZEN_REQUESTS_FILE,
    expected_sha256: str = DEFAULT_FROZEN_REQUESTS_SHA256,
    chunk_size: int = 5,
    scorer: ExternalLegalActionScorer | None = None,
    scorer_kind: str = "transformers_causal_lm",
    max_new_tokens: int = 384,
) -> dict[str, Any]:
    """Generate raw outputs for the next ``chunk_size`` unfinished requests.

    Deterministic resume: requests already present in ``raw/<arm>.jsonl`` are
    skipped. Each record is appended atomically, so a timed-out or killed
    chunk never leaves partial evidence behind.
    """
    if arm not in FRONTIER_ARMS:
        raise ValueError(
            f"unknown frontier arm {arm!r}; choose from {sorted(FRONTIER_ARMS)}"
        )
    requests = load_frozen_requests(requests_file, expected_sha256=expected_sha256)
    raw_path = output_dir / "raw" / f"{arm}.jsonl"
    done_ids = {row["id"] for row in _read_jsonl(raw_path)}
    pending = [record for record in requests if record.id not in done_ids]
    chunk = pending[:chunk_size]

    if chunk and scorer is None:
        scorer = build_external_scorer(
            ExternalScorerConfig(
                model_id=FRONTIER_ARMS[arm],
                revision="main",
                claim_class="diagnostic",
            ),
            kind=scorer_kind,
        )

    processed: list[str] = []
    for record in chunk:
        started_at = _utc_now_iso()
        t0 = time.perf_counter()
        raw_output = scorer.generate(record.prompt, max_new_tokens=max_new_tokens)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        finished_at = _utc_now_iso()
        config = scorer.config
        row = {
            "id": record.id,
            "prompt": record.prompt,
            "arm": arm,
            "model_id": config.model_id,
            "revision": scorer.resolved_revision,
            "device": config.device,
            "dtype": config.dtype,
            "prompt_template_sha256": frontier_prompt_template_sha256(),
            "raw_output": raw_output,
            "raw_output_sha256": hashlib.sha256(
                raw_output.encode("utf-8")
            ).hexdigest(),
            "latency_ms": latency_ms,
            "started_at": started_at,
            "finished_at": finished_at,
        }
        _append_jsonl_atomic(raw_path, row)
        processed.append(record.id)

    summary = {
        "arm": arm,
        "processed": processed,
        "completed": len(done_ids) + len(processed),
        "total": len(requests),
        "remaining": len(pending) - len(processed),
        "raw_path": str(raw_path),
    }
    return summary


def run_score_pass(
    *,
    output_dir: Path,
    requests_file: Path = DEFAULT_FROZEN_REQUESTS_FILE,
    expected_sha256: str = DEFAULT_FROZEN_REQUESTS_SHA256,
    arms: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """(Re)score every arm raw JSONL; deterministic and idempotent.

    Writes ``<output_dir>/scoreboard.json`` with per-request verdicts and
    per-arm Wilson 95% intervals for the v2 strict rate (primary), the v1
    verdict rate, and the official syntax rate.
    """
    from slm_training.dsl.parser import ParseError, validate
    from slm_training.evals.meaningful_program import (
        aggregate_meaning_reports_v2,
        binding_aware_meaningful_v2,
    )
    from slm_training.evals.power_protocol import wilson_interval
    from slm_training.harnesses.model_build.eval_runner import meaningful_program_v1

    requests = load_frozen_requests(requests_file, expected_sha256=expected_sha256)
    gold_by_id = {record.id: record for record in requests}

    raw_dir = output_dir / "raw"
    raw_files = sorted(raw_dir.glob("*.jsonl")) if raw_dir.exists() else []
    if arms is not None:
        wanted = set(arms)
        raw_files = [p for p in raw_files if p.stem in wanted]

    def _rate(successes: int, n: int) -> dict[str, Any]:
        return {
            "successes": successes,
            "n": n,
            "rate": successes / n if n else 0.0,
            "wilson_95": wilson_interval(successes, n),
        }

    scoreboard: dict[str, Any] = {
        "schema": "slm294_external_ceiling_scoreboard/v1",
        "requests_file": str(requests_file),
        "requests_sha256": expected_sha256,
        "prompt_template_sha256": frontier_prompt_template_sha256(),
        "arms": {},
        "version_stamp": build_version_stamp("harness.experiments.external_ceiling"),
    }
    for raw_file in raw_files:
        arm = raw_file.stem
        per_request: list[dict[str, Any]] = []
        v2_reports = []
        syntax_ok = v1_ok = v2_ok = 0
        for row in _read_jsonl(raw_file):
            pred = row["raw_output"]
            gold = gold_by_id.get(row["id"])
            try:
                validate(pred)
                syntax_valid = True
            except ParseError:
                syntax_valid = False
            v1_verdict, v1_reason, _ = meaningful_program_v1(pred, gold=gold)
            v1_reason_codes = [v1_reason] if v1_reason else []
            v2_report = None
            v2_reason_codes: list[str] = []
            v2_verdict = False
            if gold is not None:
                v2_report = binding_aware_meaningful_v2(pred, record=gold)
                v2_reports.append(v2_report)
                v2_verdict = bool(v2_report.verdict)
                v2_reason_codes = list(v2_report.reason_codes)
            syntax_ok += int(syntax_valid)
            v1_ok += int(v1_verdict)
            v2_ok += int(v2_verdict)
            per_request.append(
                {
                    "id": row["id"],
                    "syntax_valid": syntax_valid,
                    "v1_verdict": bool(v1_verdict),
                    "v1_reason_codes": v1_reason_codes,
                    "v2_verdict": v2_verdict,
                    "v2_reason_codes": v2_reason_codes,
                }
            )
        n = len(per_request)
        scoreboard["arms"][arm] = {
            "n": n,
            "per_request": per_request,
            "v2_strict": _rate(v2_ok, n),
            "v1_verdict": _rate(v1_ok, n),
            "syntax": _rate(syntax_ok, n),
            "v2_aggregate": aggregate_meaning_reports_v2(v2_reports),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "scoreboard.json"
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(
        json.dumps(scoreboard, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, out_path)
    return scoreboard
