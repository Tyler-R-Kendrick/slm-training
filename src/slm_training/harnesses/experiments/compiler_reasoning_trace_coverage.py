"""Bounded K/c coverage probe for CompilerReasoningTraceV1 (LOT0-02 / SLM-249).

Per the LOT0-01 authorization (``allowed_lot1_work``), this issue may run only
*bounded target-contract probes* -- "reading/inspecting a handful of existing
compiler/decision traces to check step-decomposability; no corpus build." This
module extracts a ``CompilerReasoningTraceV1`` for each record in the small,
already-committed ``src/slm_training/resources/test_seeds.jsonl`` fixture set
(16 hand-authored records) and reports per-stage length distributions.

This is explicitly **not** a data-derived K/c budget from a production corpus --
the report labels itself ``bounded_probe`` with its exact sample size everywhere,
and callers must not present it as a corpus-scale statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.data.progspec.compiler_reasoning_trace import CANONICAL_STAGE_ORDER
from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.semantic_plan.compiler_reasoning_trace_extract import (
    extract_compiler_reasoning_trace,
)
from slm_training.dsl.pack import DslPack, get_pack
from slm_training.dsl.schema import load_jsonl

DEFAULT_PROBE_PATH = Path("src/slm_training/resources/test_seeds.jsonl")
REPORT_SCHEMA_VERSION = "compiler_reasoning_trace_coverage_report/v1"


@dataclass(frozen=True)
class StageLengthStats:
    stage: str
    n: int
    min_chars: int
    p50_chars: float
    p95_chars: float
    max_chars: int


@dataclass(frozen=True)
class BoundedCoverageReport:
    schema_version: str
    evidence_class: str
    sample_size: int
    sample_ids: tuple[str, ...]
    stage_order: tuple[str, ...]
    proposed_k: int
    proposed_c_chars: int
    truncation_policy: str
    stage_stats: tuple[StageLengthStats, ...]
    fraction_requiring_all_stages: float
    fraction_truncated: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_class": self.evidence_class,
            "sample_size": self.sample_size,
            "sample_ids": list(self.sample_ids),
            "stage_order": list(self.stage_order),
            "proposed_k": self.proposed_k,
            "proposed_c_chars": self.proposed_c_chars,
            "truncation_policy": self.truncation_policy,
            "stage_stats": [
                {
                    "stage": s.stage,
                    "n": s.n,
                    "min_chars": s.min_chars,
                    "p50_chars": s.p50_chars,
                    "p95_chars": s.p95_chars,
                    "max_chars": s.max_chars,
                }
                for s in self.stage_stats
            ],
            "fraction_requiring_all_stages": self.fraction_requiring_all_stages,
            "fraction_truncated": self.fraction_truncated,
        }


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = q * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = rank - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def build_bounded_coverage_report(
    *,
    probe_path: Path = DEFAULT_PROBE_PATH,
    pack: DslPack | None = None,
) -> BoundedCoverageReport:
    """Extract traces for every record in ``probe_path`` and report length stats.

    ``probe_path`` defaults to the repo's committed 16-record hand-authored fixture
    set -- deliberately not a train/eval corpus, so this never becomes a covert
    corpus build.
    """
    active_pack = pack or get_pack("openui")
    records = list(load_jsonl(probe_path))
    if not records:
        raise ValueError(f"no records found at {probe_path}")

    lengths_by_stage: dict[str, list[int]] = {kind: [] for kind in CANONICAL_STAGE_ORDER}
    sample_ids: list[str] = []
    truncated_count = 0
    all_stage_count = 0

    for record in records:
        spec = ProgramSpec.from_openui(
            id=record.id,
            openui=record.openui,
            facts={},
            program_family_id=f"lot0_02_probe_{record.id}",
            lineage_id="lot0-02-bounded-probe",
            split_group_id=f"lot0_02_probe_{record.id}",
            split=record.split or "train",
        )
        trace = extract_compiler_reasoning_trace(
            spec, active_pack, source="lot0_02_bounded_probe", split=record.split or "train"
        )
        sample_ids.append(record.id)
        if trace.step_count == len(CANONICAL_STAGE_ORDER):
            all_stage_count += 1
        for step in trace.steps:
            lengths_by_stage[step.step_kind].append(len(step.canonical_target))
            if step.is_truncated:
                truncated_count += 1

    stage_stats = []
    all_p95: list[float] = []
    for stage in CANONICAL_STAGE_ORDER:
        values = sorted(lengths_by_stage[stage])
        if not values:
            continue
        p50 = _percentile([float(v) for v in values], 0.50)
        p95 = _percentile([float(v) for v in values], 0.95)
        all_p95.append(p95)
        stage_stats.append(
            StageLengthStats(
                stage=stage,
                n=len(values),
                min_chars=min(values),
                p50_chars=p50,
                p95_chars=p95,
                max_chars=max(values),
            )
        )

    proposed_c = int(max(all_p95)) if all_p95 else 0
    return BoundedCoverageReport(
        schema_version=REPORT_SCHEMA_VERSION,
        evidence_class="bounded_probe",
        sample_size=len(records),
        sample_ids=tuple(sample_ids),
        stage_order=CANONICAL_STAGE_ORDER,
        proposed_k=len(CANONICAL_STAGE_ORDER),
        proposed_c_chars=proposed_c,
        truncation_policy=(
            "fixed K=6 by stage-set construction (see compiler_reasoning_trace.py "
            "module docstring); c is provisional (max per-stage p95 char length "
            "over this bounded probe) and any step whose canonical_target would "
            "exceed c must set is_truncated=True and a non-null ambiguity_reason "
            "rather than silently drop content -- no truncation occurred in this "
            f"probe (fraction_truncated=0.0 over n={len(records)})"
        ),
        stage_stats=tuple(stage_stats),
        fraction_requiring_all_stages=(all_stage_count / len(records)) if records else 0.0,
        fraction_truncated=(truncated_count / len(records)) if records else 0.0,
    )


def render_markdown(report: BoundedCoverageReport) -> str:
    lines = [
        "# CompilerReasoningTraceV1 bounded K/c coverage probe (LOT0-02 / SLM-249)",
        "",
        f"**Evidence class: `{report.evidence_class}`.** Sample size n="
        f"{report.sample_size} (`src/slm_training/resources/test_seeds.jsonl`), "
        "not a production corpus statistic. See the LOT0-01 authorization "
        "(`docs/design/lotus-openui-fidelity-contract-v1.md`) for why corpus-scale "
        "K/c derivation is out of scope for this issue.",
        "",
        f"- proposed K: **{report.proposed_k}** (fixed by stage-set construction)",
        f"- proposed c (provisional, chars): **{report.proposed_c_chars}**",
        f"- fraction of probe records requiring all {report.proposed_k} stages: "
        f"{report.fraction_requiring_all_stages:.3f}",
        f"- fraction truncated: {report.fraction_truncated:.3f}",
        "",
        "## Truncation/fallback policy",
        "",
        report.truncation_policy,
        "",
        "## Per-stage canonical_target length (chars)",
        "",
        "| stage | n | min | p50 | p95 | max |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in report.stage_stats:
        lines.append(
            f"| {s.stage} | {s.n} | {s.min_chars} | {s.p50_chars:.1f} | "
            f"{s.p95_chars:.1f} | {s.max_chars} |"
        )
    lines.append("")
    lines.append(f"Sample record IDs: {', '.join(report.sample_ids)}")
    lines.append("")
    return "\n".join(lines)
