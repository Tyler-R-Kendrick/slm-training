"""SLM-231 (SPV0-05): RoleSlot cardinality dead-field consumption probe.

SLM-145's authorization gate for a learned `cardinality_head` predictor closed
with `blocked_pending_spv0_02_ceiling_evidence`, and one of the reasons it gave
was that ``RoleSlot.min_cardinality`` / ``max_cardinality`` extraction is
incomplete. SLM-230 (SPV0-04) directly confirmed the *producer* side of that
gap is still open: ``OpenUISemanticPlanExtractor`` never populates either
field on the SLM-144 fixture corpus (``cardinality_populated=False``).

SLM-230 did not check the *consumer* side. A dead field can be dead for two
independent reasons -- nobody writes it, or nobody reads it -- and closing only
the producer side later (teaching the extractor to fill in cardinality) would
be wasted work if nothing downstream ever looks at it. This harness answers
the consumer-side question directly, at fixture scale, without touching
`OpenUISemanticPlanExtractor`, `PlanOracleSubstitutor`, `PlanSeedBuilder`, or
`OpenUISemanticPlanCompiler`.

It builds a harness-local *candidate* cardinality derivation
(`_derive_sibling_cardinality`) -- for each role slot with a parent (per
`topology.parent_relation_candidates`), the observed count of
same-`component_family` siblings under that same parent; the root role (no
parent edge) gets `(1, 1)`. This is deliberately not proposed as the real
extraction policy -- it is only a vehicle to get *some* non-None cardinality
values through the real, unmodified pipeline. It then runs the standard
`roles+topology` oracle-substitution arm (the one SLM-230's C4 showed reaches
full seed validity) once with the gold-extracted role slots as-is (cardinality
None, matching production today) and once with the derived cardinality
populated on every slot, and diffs the two arms record-by-record: same seed
text, same `ok`, same `reason`?

No learned head, checkpoint, GPU run, or ship-gate claim is made. This harness
answers only: is `RoleSlot.min_cardinality` / `max_cardinality` currently
*consumed* anywhere in the real `PlanSeedBuilder` / `OpenUISemanticPlanCompiler`
mechanism, once it is non-None?
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.progspec.semantic_plan import PlanIdentity, RoleSlot, SemanticPlanV1
from slm_training.data.semantic_plan import OpenUISemanticPlanCompiler
from slm_training.data.semantic_plan.corpus import build_fixture_plan_corpus
from slm_training.data.semantic_plan.oracle import PlanOracleSubstitutor
from slm_training.dsl.grammar.backends.ast_utils import component_multiset
from slm_training.dsl.pack import DslPack, get_pack
from slm_training.dsl.parser import validate
from slm_training.harnesses.experiments.slm146_semantic_plan_compiler import (
    _render_fixture_ast,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "Slm231Mismatch",
    "Slm231Record",
    "Slm231Report",
    "Slm231Row",
    "render_markdown",
    "run_fixture_matrix",
]

MATRIX_VERSION = "spv0-05-v1"
MATRIX_SET = "slm231_role_cardinality_dead_field"
EXPERIMENT_ID = "slm231-spv0-05-role-cardinality-dead-field"

_HYPOTHESIS = (
    "Populating RoleSlot.min_cardinality/max_cardinality with a deterministic "
    "harness-local candidate derivation (observed same-component_family "
    "sibling count per parent), then oracle-substituting the `roles`+"
    "`topology` factors as in SLM-230's C4 arm, produces byte-identical "
    "PlanSeedBuilder/OpenUISemanticPlanCompiler output (seed text, ok flag, "
    "reason) to the same arm with cardinality left at None (today's "
    "production extractor output), on every record of the SLM-144 fixture "
    "corpus -- showing the cardinality fields are not just unpopulated "
    "(SLM-230) but also fully unconsumed by the current PlanSeedBuilder "
    "mechanism: a field with neither a producer nor a consumer."
)

_FALSIFIER = (
    "Any fixture-corpus record produces a different seed text, `ok` flag, or "
    "`reason` between the cardinality-populated arm and the cardinality-None "
    "arm, holding every other factor identical -- showing something in the "
    "current PlanSeedBuilder/OpenUISemanticPlanCompiler mechanism already "
    "reads RoleSlot cardinality."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: the deterministic SLM-144 fixture corpus "
    "(`build_fixture_plan_corpus`), not a real or held-out completion "
    "corpus. No checkpoint, learned predictor, GPU run, or ship-gate claim "
    "is made or implied.",
    "The cardinality values injected here are a harness-local *candidate* "
    "derivation (`_derive_sibling_cardinality`), not a proposal for "
    "`OpenUISemanticPlanExtractor`'s real extraction policy. It exists only "
    "to get non-None values through the real pipeline for this consumption "
    "probe; no accuracy, calibration, or policy claim is made about it.",
    "This harness does not modify `OpenUISemanticPlanExtractor`, "
    "`PlanOracleSubstitutor`, `PlanSeedBuilder`, or "
    "`OpenUISemanticPlanCompiler` -- it only exercises them, matching the "
    "SLM-230 (SPV0-04) convention.",
    "A negative (unconsumed) result here does not mean cardinality is "
    "worthless -- it means the *current* PlanSeedBuilder mechanism (which "
    "renders one child per topology edge with no repetition/expansion logic) "
    "has no code path that would act on a cardinality bound even if one were "
    "supplied. A future PlanSeedBuilder that synthesizes repeated children "
    "from a cardinality count would need new code, not just populated data.",
    "Both arms here use `honesty_mode=oracle_diagnostic` and "
    "`plan_source=gold`; neither arm is promotable per the SemanticPlanV1 "
    "contract, matching SLM-230.",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(_canonical_json(value))


def _derive_sibling_cardinality(plan: SemanticPlanV1) -> SemanticPlanV1:
    """Return a copy of *plan* with RoleSlot.min/max_cardinality populated.

    Harness-local *candidate* derivation only -- never wired into
    `OpenUISemanticPlanExtractor` and never claimed as the real extraction
    policy. For each role slot with a parent edge (per
    `topology.parent_relation_candidates`), cardinality is set to the
    observed count of same-`component_family` siblings under that same
    parent (`min == max == count`, an "observed now" cardinality rather than
    an inferred min/max policy range). The root role (no parent edge) gets
    `(1, 1)`.
    """
    family_by_role = {slot.role_id: slot.component_family for slot in plan.role_slots}
    children_by_parent: dict[str, list[str]] = {}
    child_ids: set[str] = set()
    for edge in plan.topology.parent_relation_candidates or ():
        parent = str(edge.get("parent_role_id") or "")
        child = str(edge.get("child_role_id") or "")
        if parent and child:
            children_by_parent.setdefault(parent, []).append(child)
            child_ids.add(child)

    cardinality_by_role: dict[str, tuple[int, int]] = {}
    for children in children_by_parent.values():
        family_counts: dict[str | None, int] = {}
        for child in children:
            fam = family_by_role.get(child)
            family_counts[fam] = family_counts.get(fam, 0) + 1
        for child in children:
            count = family_counts[family_by_role.get(child)]
            cardinality_by_role[child] = (count, count)

    new_slots: list[RoleSlot] = []
    for slot in plan.role_slots:
        if slot.role_id in cardinality_by_role:
            lo, hi = cardinality_by_role[slot.role_id]
        elif slot.role_id not in child_ids:
            lo, hi = 1, 1
        else:  # pragma: no cover - defensive: orphaned child edge
            lo, hi = None, None
        new_slots.append(
            RoleSlot(
                role_id=slot.role_id,
                component_family=slot.component_family,
                candidate_distribution=slot.candidate_distribution,
                min_cardinality=lo,
                max_cardinality=hi,
                required=slot.required,
                evidence_spans=slot.evidence_spans,
            )
        )
    return plan.model_copy(update={"role_slots": tuple(new_slots)})


@dataclass(frozen=True)
class Slm231Record:
    """Per fixture-corpus-record diagnostics for one arm."""

    record_id: str
    arm_id: str
    seed: str | None
    seed_ok: bool
    seed_present: bool
    component_coverage: float | None
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Slm231Row:
    """Aggregated row for one arm across the fixture corpus."""

    arm_id: str
    cardinality_populated: bool
    n_records: int
    seed_ok_count: int
    seed_valid_count: int
    seed_valid_rate: float
    mean_component_coverage: float | None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Slm231Mismatch:
    """One record where the two arms disagree."""

    record_id: str
    no_cardinality_seed: str | None
    with_cardinality_seed: str | None
    no_cardinality_ok: bool
    with_cardinality_ok: bool
    no_cardinality_reason: str | None
    with_cardinality_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


def _component_coverage(seed_source: str, gold_source: str) -> float | None:
    try:
        seed_program = validate(seed_source)
        gold_program = validate(gold_source)
    except Exception:  # noqa: BLE001
        return None
    seed_components = component_multiset(seed_program.root)
    gold_components = component_multiset(gold_program.root)
    keys = set(seed_components) | set(gold_components)
    if not keys:
        return 1.0
    total = sum(max(seed_components.get(k, 0), gold_components.get(k, 0)) for k in keys)
    if not total:
        return 1.0
    overlap = sum(min(seed_components.get(k, 0), gold_components.get(k, 0)) for k in keys)
    return overlap / total


def _evaluate_record(
    spec: ProgramSpec,
    oracle_plan: SemanticPlanV1,
    *,
    arm_id: str,
    pack: DslPack,
    compiler: OpenUISemanticPlanCompiler,
) -> Slm231Record:
    baseline = SemanticPlanV1(
        identity=PlanIdentity(
            pack_id=pack.pack_id,
            contract_hash=spec.contract_id,
            provenance="predicted",
        )
    )
    substitutor = PlanOracleSubstitutor(
        plan_source="gold",
        oracle_factors=("roles", "topology"),
        use_mode="seed",
        honesty_mode="oracle_diagnostic",
    )
    plan = substitutor.apply(baseline, oracle_plan)
    result = compiler.build_valid_seed(None, plan, pack)

    coverage: float | None = None
    if result.ok and result.seed:
        gold_source = _render_fixture_ast(spec.ast)
        coverage = _component_coverage(result.seed, gold_source)

    return Slm231Record(
        record_id=spec.id,
        arm_id=arm_id,
        seed=result.seed,
        seed_ok=result.ok,
        seed_present=bool(result.ok and result.seed),
        component_coverage=coverage,
        reason=result.reason,
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(arm_id: str, cardinality_populated: bool, records: list[Slm231Record]) -> Slm231Row:
    n = len(records)
    seed_ok = sum(1 for r in records if r.seed_ok)
    valid = [r for r in records if r.seed_present]
    return Slm231Row(
        arm_id=arm_id,
        cardinality_populated=cardinality_populated,
        n_records=n,
        seed_ok_count=seed_ok,
        seed_valid_count=len(valid),
        seed_valid_rate=(len(valid) / n) if n else 0.0,
        mean_component_coverage=_mean(
            [r.component_coverage for r in valid if r.component_coverage is not None]
        ),
    )


def _find_mismatches(
    no_card: list[Slm231Record], with_card: list[Slm231Record]
) -> list[Slm231Mismatch]:
    with_by_id = {r.record_id: r for r in with_card}
    mismatches: list[Slm231Mismatch] = []
    for a in no_card:
        b = with_by_id.get(a.record_id)
        if b is None:
            continue
        if a.seed != b.seed or a.seed_ok != b.seed_ok or a.reason != b.reason:
            mismatches.append(
                Slm231Mismatch(
                    record_id=a.record_id,
                    no_cardinality_seed=a.seed,
                    with_cardinality_seed=b.seed,
                    no_cardinality_ok=a.seed_ok,
                    with_cardinality_ok=b.seed_ok,
                    no_cardinality_reason=a.reason,
                    with_cardinality_reason=b.reason,
                )
            )
    return mismatches


def _resolve_disposition(
    rows: dict[str, Slm231Row], mismatches: list[Slm231Mismatch], derivation_populated: bool
) -> tuple[str, str]:
    if not derivation_populated:
        return (
            "derivation_failed",
            "The harness-local cardinality derivation produced no non-None "
            "values on this corpus (e.g. no multi-role records); the "
            "consumption probe did not get a meaningful non-None input and "
            "is inconclusive.",
        )
    if mismatches:
        ids = ", ".join(m.record_id for m in mismatches[:5])
        more = f" (+{len(mismatches) - 5} more)" if len(mismatches) > 5 else ""
        return (
            "cardinality_consumption_detected",
            f"{len(mismatches)} of {rows['F0_no_cardinality'].n_records} records "
            f"produced different seed text/ok/reason once RoleSlot cardinality "
            f"was populated on the roles+topology oracle arm (e.g. {ids}{more}) "
            "-- something in the current PlanSeedBuilder/OpenUISemanticPlanCompiler "
            "mechanism already reads cardinality; the 'fully dead field' "
            "hypothesis is falsified.",
        )
    return (
        "cardinality_confirmed_unconsumed",
        "Every fixture-corpus record produced byte-identical seed text, ok "
        "flag, and reason whether or not RoleSlot.min_cardinality/"
        "max_cardinality were populated on the roles+topology oracle arm. "
        "Combined with SLM-230's producer-side finding (extractor never "
        "populates these fields), the cardinality fields are confirmed to "
        "have neither a producer nor a consumer in the current pipeline: a "
        "fully dead field, not merely an unpopulated one.",
    )


@dataclass(frozen=True)
class Slm231Report:
    """Full fixture report for SLM-231."""

    schema: str = "Slm231RoleCardinalityDeadFieldReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = EXPERIMENT_ID
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    corpus_size: int = 0
    corpus_seed: int = 0
    derivation_populated_cardinality: bool = False
    rows: tuple[Slm231Row, ...] = field(default_factory=tuple)
    mismatches: tuple[Slm231Mismatch, ...] = field(default_factory=tuple)
    gate_hash: str = ""
    disposition: str = "inconclusive"
    disposition_rationale: str = ""
    honest_caveats: tuple[str, ...] = _HONEST_CAVEATS
    version_stamp: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "corpus_size": self.corpus_size,
            "corpus_seed": self.corpus_seed,
            "derivation_populated_cardinality": self.derivation_populated_cardinality,
            "rows": [r.to_dict() for r in self.rows],
            "mismatches": [m.to_dict() for m in self.mismatches],
            "gate_hash": self.gate_hash,
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "honest_caveats": list(self.honest_caveats),
            "version_stamp": dict(self.version_stamp),
            "timestamp": self.timestamp,
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Slm231Report":
        rows = tuple(
            Slm231Row(
                arm_id=str(r["arm_id"]),
                cardinality_populated=bool(r.get("cardinality_populated", False)),
                n_records=int(r["n_records"]),
                seed_ok_count=int(r["seed_ok_count"]),
                seed_valid_count=int(r["seed_valid_count"]),
                seed_valid_rate=float(r["seed_valid_rate"]),
                mean_component_coverage=r.get("mean_component_coverage"),
            )
            for r in data.get("rows", ())
        )
        mismatches = tuple(
            Slm231Mismatch(
                record_id=str(m["record_id"]),
                no_cardinality_seed=m.get("no_cardinality_seed"),
                with_cardinality_seed=m.get("with_cardinality_seed"),
                no_cardinality_ok=bool(m.get("no_cardinality_ok", False)),
                with_cardinality_ok=bool(m.get("with_cardinality_ok", False)),
                no_cardinality_reason=m.get("no_cardinality_reason"),
                with_cardinality_reason=m.get("with_cardinality_reason"),
            )
            for m in data.get("mismatches", ())
        )
        return cls(
            schema=str(data.get("schema", "Slm231RoleCardinalityDeadFieldReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", EXPERIMENT_ID)),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            corpus_size=int(data.get("corpus_size", 0)),
            corpus_seed=int(data.get("corpus_seed", 0)),
            derivation_populated_cardinality=bool(data.get("derivation_populated_cardinality", False)),
            rows=rows,
            mismatches=mismatches,
            gate_hash=str(data.get("gate_hash", "")),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def _cardinality_populated(plans: list[SemanticPlanV1]) -> bool:
    return any(
        slot.min_cardinality is not None or slot.max_cardinality is not None
        for plan in plans
        for slot in plan.role_slots
    )


def run_fixture_matrix(
    *,
    corpus_size: int = 24,
    corpus_seed: int = 0,
    run_id: str | None = None,
) -> Slm231Report:
    """Run the roles+topology oracle arm twice (with/without derived
    cardinality) through the real PlanOracleSubstitutor/PlanSeedBuilder/
    OpenUISemanticPlanCompiler pipeline against the deterministic SLM-144
    fixture corpus, and diff the two arms record-by-record."""
    pack = get_pack("openui")
    compiler = OpenUISemanticPlanCompiler(honesty_mode="oracle_diagnostic")
    corpus = build_fixture_plan_corpus(
        count=corpus_size,
        seed=corpus_seed,
        root_containers=["Stack", "Card"],
        leaf_components=["TextContent", "Button"],
    )
    train = corpus["train"]

    with_card_plans = [_derive_sibling_cardinality(plan) for _, plan in train]
    derivation_populated = _cardinality_populated(with_card_plans)

    no_card_records = [
        _evaluate_record(spec, plan, arm_id="F0_no_cardinality", pack=pack, compiler=compiler)
        for (spec, plan), _ in zip(train, with_card_plans)
    ]
    with_card_records = [
        _evaluate_record(spec, plan, arm_id="F1_with_cardinality", pack=pack, compiler=compiler)
        for (spec, _), plan in zip(train, with_card_plans)
    ]

    rows = {
        "F0_no_cardinality": _aggregate("F0_no_cardinality", False, no_card_records),
        "F1_with_cardinality": _aggregate("F1_with_cardinality", derivation_populated, with_card_records),
    }
    mismatches = _find_mismatches(no_card_records, with_card_records)
    disposition, rationale = _resolve_disposition(rows, mismatches, derivation_populated)

    payload = {
        "row_digests": sorted(_digest(row.to_dict()) for row in rows.values()),
        "mismatch_digests": sorted(_digest(m.to_dict()) for m in mismatches),
    }
    gate_hash = _sha256(_canonical_json(payload))

    return Slm231Report(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        corpus_size=len(train),
        corpus_seed=corpus_seed,
        derivation_populated_cardinality=derivation_populated,
        rows=(rows["F0_no_cardinality"], rows["F1_with_cardinality"]),
        mismatches=tuple(mismatches),
        gate_hash=gate_hash,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm231_role_cardinality_dead_field",
        ),
    )


def render_markdown(report: Slm231Report) -> str:
    lines = [
        f"# SLM-231 (SPV0-05): RoleSlot cardinality dead-field consumption probe ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        f"**Corpus:** SLM-144 fixture, n={report.corpus_size}, seed={report.corpus_seed}",
        f"**Harness-local derivation populated cardinality:** {report.derivation_populated_cardinality}",
        f"**Gate hash:** `{report.gate_hash[:16]}...`",
        f"**Disposition:** {report.disposition} — {report.disposition_rationale}",
        "",
        "## Hypothesis",
        "",
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Honest caveats",
        "",
        *(f"- {c}" for c in report.honest_caveats),
        "",
        "## Per-arm results",
        "",
        "| arm | cardinality populated | n | seed valid rate | mean component coverage |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report.rows:
        def _fmt(v: float | None) -> str:
            return "—" if v is None else f"{v:.3f}"

        lines.append(
            f"| {row.arm_id} | {row.cardinality_populated} | {row.n_records} | "
            f"{row.seed_valid_rate:.3f} | {_fmt(row.mean_component_coverage)} |"
        )
    lines += ["", "## Mismatches", ""]
    if report.mismatches:
        for m in report.mismatches:
            lines.append(
                f"- **{m.record_id}**: ok {m.no_cardinality_ok}→{m.with_cardinality_ok}, "
                f"reason {m.no_cardinality_reason!r}→{m.with_cardinality_reason!r}"
            )
    else:
        lines.append("- (none — every record matched exactly)")
    lines += [
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence only. It does not change "
        "`OpenUISemanticPlanExtractor`, `PlanOracleSubstitutor`, "
        "`PlanSeedBuilder`, or `OpenUISemanticPlanCompiler`, does not train "
        "or evaluate a learned head, and does not reopen SLM-145. It "
        "supplies consumer-side evidence complementing SLM-230's "
        "producer-side finding, for a human maintainer to weigh together "
        "when deciding whether cardinality extraction is worth building.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m scripts.run_slm231_role_cardinality_dead_field --mode plan-only",
        "python -m scripts.run_slm231_role_cardinality_dead_field --mode fixture",
        "```",
        "",
    ]
    return "\n".join(lines)
