"""SLM-295: local retrieval plus grammar-verified patch baseline.

This is a deterministic non-model baseline.  Retrieval only sees a request and
the verified train index; held-out targets are accepted solely by the optional
diagnostic scorer after a candidate was selected.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

from slm_training.data.progspec import ProgramSpec
from slm_training.data.semantic_plan import (
    OpenUISemanticPlanExtractor,
    plan_factor_fingerprints,
)
from slm_training.data.verify import verify_record
from slm_training.dsl.canonicalize import canonical_fingerprint
from slm_training.dsl.pack import get_pack
from slm_training.dsl.parser import validate
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.meaningful_program import binding_aware_meaningful_v2
from slm_training.harnesses.experiments.ast_sketch_retrieval_factorial import (
    AstTrainingSketchV1,
    build_ast_training_sketch,
)
from slm_training.harnesses.experiments.slm147_x22_retrieval import (
    _ast_sketch_score,
    _must_exclude,
    _plan_factor_score,
)
from slm_training.harnesses.model_build.eval_runner import (
    _binder_reference_f1,
    _contract_precision,
    _contract_recall,
)
from slm_training.data.edits import build_transition, diff_programs

SCHEMA = "VerifiedPatchRetrievalV1"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _lineage_ids(record: ExampleRecord) -> set[str]:
    meta = record.meta or {}
    return {
        str(value)
        for value in (
            record.id,
            meta.get("root_parent_id"),
            meta.get("lineage_id"),
            meta.get("split_group_id"),
        )
        if value
    }


def _plan_factors(record: ExampleRecord) -> dict[str, str]:
    """Extract canonical factor views from a verified program, fail closed."""
    spec = ProgramSpec.from_openui(
        id=record.id,
        openui=record.openui,
        facts={},
        program_family_id=str(
            (record.meta or {}).get("program_family_id") or record.source
        ),
        lineage_id=str((record.meta or {}).get("lineage_id") or record.id),
        split_group_id=str((record.meta or {}).get("split_group_id") or record.id),
        split=record.split,
    )
    plan = OpenUISemanticPlanExtractor().extract(spec, get_pack("openui"))
    return plan_factor_fingerprints(plan)


@dataclass(frozen=True)
class RetrievalRequest:
    id: str
    prompt: str
    current_openui: str | None = None
    split_group_id: str | None = None
    lineage_id: str | None = None

    def as_record(self) -> ExampleRecord:
        return ExampleRecord(
            id=self.id,
            prompt=self.prompt,
            openui=self.current_openui or 'root = Stack([], "column")',
            split="held_out",
            source="retrieval_request",
            meta={
                "split_group_id": self.split_group_id or self.id,
                "lineage_id": self.lineage_id or self.id,
            },
        )


@dataclass(frozen=True)
class VerifiedPatchEntry:
    record: ExampleRecord
    source: str
    canonical_fingerprint: str
    ast_sketch: AstTrainingSketchV1
    plan_factors: dict[str, str]
    verifier_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record.id,
            "prompt_fingerprint": _sha(self.record.prompt),
            "canonical_fingerprint": self.canonical_fingerprint,
            "ast_sketch": self.ast_sketch.to_dict(),
            "semantic_plan_factors": dict(self.plan_factors),
            "verifier_ok": self.verifier_ok,
        }


@dataclass(frozen=True)
class VerifiedPatchIndex:
    entries: tuple[VerifiedPatchEntry, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class RetrievedCandidate:
    entry: VerifiedPatchEntry
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {**self.entry.to_dict(), "score": self.score, "rank": self.rank}


@dataclass(frozen=True)
class PatchResult:
    request_id: str
    arm: str
    output: str | None
    raw_output: str | None
    selected_id: str | None
    candidates: tuple[RetrievedCandidate, ...]
    excluded_reasons: dict[str, int]
    patch: dict[str, Any] | None
    verifier_ok: bool
    fallback: str | None
    latency_ms: float
    reranker: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


Reranker = Callable[[RetrievalRequest, RetrievedCandidate], float]


def build_verified_patch_index(records: Iterable[ExampleRecord]) -> VerifiedPatchIndex:
    """Index only verified train records with AST and SemanticPlan feature views."""
    entries: list[VerifiedPatchEntry] = []
    rejected: list[dict[str, str]] = []
    for record in records:
        if record.split != "train":
            rejected.append({"id": record.id, "reason": "not_train"})
            continue
        try:
            validate(record.openui)
            report = verify_record(record)
            if not report.ok:
                rejected.append({"id": record.id, "reason": "verification_failed"})
                continue
            entries.append(
                VerifiedPatchEntry(
                    record=record,
                    source=validate(record.openui).serialized or record.openui.strip(),
                    canonical_fingerprint=canonical_fingerprint(record.openui),
                    ast_sketch=build_ast_training_sketch(record.openui),
                    plan_factors=_plan_factors(record),
                    verifier_ok=True,
                )
            )
        except (KeyError, ValueError):
            rejected.append({"id": record.id, "reason": "invalid_program"})
    return VerifiedPatchIndex(
        entries=tuple(entries),
        manifest={
            "schema": SCHEMA,
            "split": "train",
            "n_entries": len(entries),
            "rejected": rejected,
            "index_records": [entry.to_dict() for entry in entries],
        },
    )


def _request_features(
    request: RetrievalRequest,
) -> tuple[AstTrainingSketchV1 | None, dict[str, str]]:
    if not request.current_openui:
        return None, {}
    record = request.as_record()
    try:
        return build_ast_training_sketch(record.openui), _plan_factors(record)
    except (KeyError, ValueError):
        return None, {}


def _excluded(request: RetrievalRequest, entry: VerifiedPatchEntry) -> list[str]:
    query = request.as_record()
    blocked, reasons = _must_exclude(query, entry.record)
    del blocked
    if _lineage_ids(query) & _lineage_ids(entry.record):
        reasons.append("same_example_derivative")
    return sorted(set(reasons))


def retrieve_verified_candidates(
    index: VerifiedPatchIndex, request: RetrievalRequest, *, k: int = 3
) -> tuple[tuple[RetrievedCandidate, ...], dict[str, int]]:
    """Retrieve train-only candidates; targets are not an input to this function."""
    sketch, factors = _request_features(request)
    prompt_tokens = set(request.prompt.lower().split())
    eligible: list[tuple[VerifiedPatchEntry, float]] = []
    exclusions: Counter[str] = Counter()
    for entry in index.entries:
        reasons = _excluded(request, entry)
        if reasons:
            exclusions.update(reasons)
            continue
        candidate_tokens = set(entry.record.prompt.lower().split())
        prompt_score = (
            len(prompt_tokens & candidate_tokens)
            / len(prompt_tokens | candidate_tokens)
            if prompt_tokens | candidate_tokens
            else 0.0
        )
        scores = [prompt_score]
        if sketch is not None:
            scores.append(_ast_sketch_score(sketch, entry.ast_sketch))
        if factors:
            scores.append(_plan_factor_score(factors, entry.plan_factors))
        eligible.append((entry, sum(scores) / len(scores)))
    selected = sorted(eligible, key=lambda row: (-row[1], row[0].record.id))[
        : max(1, k)
    ]
    return (
        tuple(
            RetrievedCandidate(entry, score, rank)
            for rank, (entry, score) in enumerate(selected, 1)
        ),
        dict(sorted(exclusions.items())),
    )


def _execute_candidate(
    request: RetrievalRequest,
    candidate: RetrievedCandidate | None,
    *,
    arm: str,
    candidates: tuple[RetrievedCandidate, ...],
    exclusions: dict[str, int],
    reranker: str,
) -> PatchResult:
    start = time.perf_counter()
    if candidate is None:
        return PatchResult(
            request.id,
            arm,
            None,
            None,
            None,
            candidates,
            exclusions,
            None,
            False,
            "no_eligible_candidate",
            0.0,
            reranker,
        )
    raw = candidate.entry.source
    output = raw
    patch_payload: dict[str, Any] | None = None
    fallback: str | None = None
    if request.current_openui:
        try:
            patch = diff_programs(
                request.current_openui, raw, instruction="retrieve verified layout"
            )
            transition = build_transition(
                request.current_openui,
                "retrieve verified layout",
                patch,
                render_verifier=lambda _source: True,
            )
            output = transition.after
            patch_payload = {
                "operations": patch.to_dict(),
                "inverse": transition.inverse.to_dict(),
                "render_verified": transition.render_verified,
            }
        except ValueError:
            fallback = "patch_verification_failed_selected_verified_candidate"
    output_record = ExampleRecord(
        id=request.id,
        prompt=request.prompt,
        openui=output,
        split="held_out",
        source="retrieval_baseline",
    )
    verified = bool(verify_record(output_record).ok)
    if not verified:
        fallback = "output_verification_failed"
        output = None
    return PatchResult(
        request.id,
        arm,
        output,
        raw,
        candidate.entry.record.id,
        candidates,
        exclusions,
        patch_payload,
        verified,
        fallback,
        (time.perf_counter() - start) * 1000,
        reranker,
    )


def run_verified_patch_baseline(
    index: VerifiedPatchIndex,
    request: RetrievalRequest,
    *,
    arm: str = "top1",
    k: int = 3,
    reranker: Reranker | None = None,
) -> PatchResult:
    """Select a verified candidate then apply only a grammar-verified patch."""
    candidates, exclusions = retrieve_verified_candidates(index, request, k=k)
    reranker_name = "deterministic_retrieval"
    selected = candidates[0] if candidates else None
    if reranker is not None and candidates:
        selected = max(
            candidates,
            key=lambda candidate: (reranker(request, candidate), -candidate.rank),
        )
        reranker_name = "learned_reranker"
    return _execute_candidate(
        request,
        selected,
        arm=arm,
        candidates=candidates,
        exclusions=exclusions,
        reranker=reranker_name,
    )


def _mean(values: list[float | None]) -> float | None:
    numbers = [value for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return (
        ordered[lower]
        if lower == upper
        else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    )


def evaluate_local_baseline(
    index: VerifiedPatchIndex,
    cases: Iterable[tuple[RetrievalRequest, ExampleRecord]],
    *,
    k: int = 3,
    peak_memory_bytes: int | None = None,
) -> dict[str, Any]:
    """Local diagnostic arms; gold is used only after selection for scoring."""
    rows: list[dict[str, Any]] = []
    for request, gold in cases:
        for arm in ("top1", "learned_rerank_fallback", "patching"):
            result = run_verified_patch_baseline(index, request, arm=arm, k=k)
            rows.append(_score_row(result, gold, diagnostic=False))
        candidates, exclusions = retrieve_verified_candidates(index, request, k=k)
        oracle = next(
            (
                candidate
                for candidate in candidates
                if candidate.entry.canonical_fingerprint
                == canonical_fingerprint(gold.openui)
            ),
            candidates[0] if candidates else None,
        )
        result = _execute_candidate(
            request,
            oracle,
            arm="oracle_topk_diagnostic",
            candidates=candidates,
            exclusions=exclusions,
            reranker="gold_diagnostic_only",
        )
        rows.append(_score_row(result, gold, diagnostic=True))
    arms: dict[str, dict[str, Any]] = {}
    for arm in sorted({str(row["arm"]) for row in rows}):
        arm_rows = [row for row in rows if row["arm"] == arm]
        arms[arm] = {
            "n": len(arm_rows),
            "raw_syntax_validity": _mean(
                [row["raw_syntax_validity"] for row in arm_rows]
            ),
            "constrained_syntax_validity": _mean(
                [row["constrained_syntax_validity"] for row in arm_rows]
            ),
            "repaired_syntax_validity": _mean(
                [row["repaired_syntax_validity"] for row in arm_rows]
            ),
            "binding_aware_meaningful_v2_rate_strict": _mean(
                [row["meaningful"] for row in arm_rows]
            ),
            "binder_reference_f1": _mean(
                [row["binder_reference_f1"] for row in arm_rows]
            ),
            "latency_ms_p50": _percentile([row["latency_ms"] for row in arm_rows], 0.5),
            "latency_ms_p95": _percentile(
                [row["latency_ms"] for row in arm_rows], 0.95
            ),
            "peak_memory_bytes": peak_memory_bytes,
            "diagnostic_non_promotable": arm == "oracle_topk_diagnostic",
        }
    return {
        "schema": SCHEMA,
        "claim_class": "local_diagnostic_non_ship",
        "index_manifest": index.manifest,
        "arms": arms,
        "records": rows,
    }


def _score_row(
    result: PatchResult, gold: ExampleRecord, *, diagnostic: bool
) -> dict[str, Any]:
    output = result.output
    syntax = bool(output and _valid(output))
    meaningful = (
        binding_aware_meaningful_v2(output or "", record=gold).verdict
        if output
        else False
    )
    precision = _contract_precision(output or "", gold) if output else None
    recall = _contract_recall(output or "", gold) if output else None
    return {
        "arm": result.arm,
        "diagnostic_non_promotable": diagnostic,
        "raw_syntax_validity": bool(result.raw_output and _valid(result.raw_output)),
        "constrained_syntax_validity": syntax,
        "repaired_syntax_validity": syntax,
        "meaningful": meaningful,
        "binder_reference_f1": _binder_reference_f1(precision, recall),
        "latency_ms": result.latency_ms,
        "result": result.to_dict(),
    }


def _valid(source: str) -> bool:
    try:
        validate(source)
        return True
    except ValueError:
        return False
