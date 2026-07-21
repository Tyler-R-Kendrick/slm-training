"""SLM-230 (SPV0-04): plan-factor oracle-substitution ceiling matrix.

SLM-142 (SPV0-02) wired `PlanOracleSubstitutor`, `PlanSeedBuilder`, and
`OpenUISemanticPlanCompiler`, but its own honesty caveats say plainly:
"Factor-wise oracle substitution is implemented only for the four factors
listed above... A future issue can add explicit symbol/coverage oracle arms
if diagnostics require them." SLM-145's authorization gate for learned
`topology_head` / `cardinality_head` / `live_symbol_pointer_head` predictors
then found "no oracle-substitution experiment found" for `topology` and
`bindings_pointers`, and that `cardinality` extraction is incomplete, closing
with `blocked_pending_spv0_02_ceiling_evidence` and the explicit recommended
next step: "Run a factor-wise oracle-substitution matrix on a real or fixture
completion corpus." The program-wide SLM-160 (SPV4-02) disposition audit
independently reached the same conclusion, naming this the single "next
high-leverage step" for the whole SPV program.

This harness is that step, at fixture scale. It never modifies
`PlanOracleSubstitutor`, `PlanSeedBuilder`, or `OpenUISemanticPlanCompiler` --
it only *exercises* them (for the first time end-to-end) across a battery of
factor-subset oracle arms on the deterministic SLM-144 fixture corpus, and
measures three downstream ceiling proxies per arm:

- ``seed_valid_rate`` -- fraction of records where the compiler produces a
  non-empty, grammar-valid seed from the given factor subset alone (baseline
  factors empty, only the named factors set from the gold-extracted plan).
- ``mean_component_coverage`` -- multiset overlap of component families
  between the built seed and the record's true program, among valid seeds.
- ``mean_placeholder_attachment_ratio`` -- fraction of the gold record's
  content-placeholder positions that the built seed also fills with *some*
  placeholder argument, among valid seeds. This isolates the `bindings`
  factor's marginal contribution from the `seed_to_gold_ratio` token-overlap
  metric reused by the sibling SLM-146/147/148 harnesses, which turns out (by
  direct measurement below) to be dominated by `PlanSeedBuilder`'s own
  statement-naming/ordering convention (``node_N`` root-last) diverging from
  the fixture gold renderer's convention (``nN`` root-first) regardless of
  binding content -- see the honest caveats on ``Slm230Report``.

No learned head, checkpoint, GPU run, or ship-gate claim is made. This
harness answers only the narrow authorization-gate question SLM-145 and
SLM-160 left open: does giving the compiler the *true* value of a factor
(and only that factor) move a measurable downstream proxy off the no-plan
baseline, and which factors are jointly rather than independently sufficient
to do so in the current `PlanSeedBuilder` mechanism?
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.progspec.semantic_plan import PlanIdentity, SemanticPlanV1
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
    "Slm230Arm",
    "Slm230Record",
    "Slm230Report",
    "Slm230Row",
    "build_default_arms",
    "render_markdown",
    "run_fixture_matrix",
]

MATRIX_VERSION = "spv0-04-v1"
MATRIX_SET = "slm230_plan_factor_ceiling_matrix"
EXPERIMENT_ID = "slm230-spv0-04-plan-factor-ceiling-matrix"

_HYPOTHESIS = (
    "Oracle-substituting the `roles` and `topology` SemanticPlanV1 factors "
    "together (holding bindings/archetype at the empty no-plan baseline) "
    "raises PlanSeedBuilder/OpenUISemanticPlanCompiler seed validity from 0% "
    "(no-plan baseline) to a high rate on the SLM-144 fixture corpus, and "
    "additionally oracle-substituting `bindings` raises the measured "
    "content-placeholder attachment ratio from 0.0 to a high value -- "
    "supplying the factor-wise downstream-ceiling evidence SLM-145's "
    "authorization gate and the SLM-160 program disposition both found "
    "missing. Individually oracle-substituting only `roles`, only "
    "`topology`, or only `bindings` (holding the other factors at baseline) "
    "does not produce a valid seed / does not change the compiled output on "
    "any record, showing the current PlanSeedBuilder mechanism needs these "
    "factors jointly rather than accepting them as independently sufficient."
)

_FALSIFIER = (
    "Any single-factor arm (roles-only, topology-only, or bindings-only) "
    "produces a valid, non-trivial seed on a meaningful fraction of "
    "records; or the combined roles+topology arm fails to raise seed "
    "validity above the no-plan baseline; or adding the bindings factor on "
    "top of roles+topology fails to raise the placeholder-attachment ratio "
    "above the roles+topology-only arm."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: the deterministic SLM-144 fixture corpus "
    "(`build_fixture_plan_corpus`), not a real or held-out completion "
    "corpus. No checkpoint, learned predictor, GPU run, or ship-gate claim "
    "is made or implied.",
    "Every oracle arm here (all but `C0_no_plan`) injects the true "
    "gold-extracted plan factor(s) directly -- this measures an upper-bound "
    "ceiling given perfect factor knowledge, not the accuracy of any "
    "predictor (none exists). `honesty_mode=oracle_diagnostic` is used "
    "throughout; none of these arms are promotable per the SemanticPlanV1 "
    "contract.",
    "RoleSlot.min_cardinality / max_cardinality are never populated by "
    "OpenUISemanticPlanExtractor (verified directly against the fixture "
    "corpus by this harness's `cardinality_populated` field) -- SLM-145's "
    "documented cardinality-extraction gap is confirmed still open, so no "
    "cardinality-specific arm distinct from `roles` can be measured yet.",
    "The `mean_seed_to_gold_ratio` token-overlap metric (reused from the "
    "sibling SLM-146/147/148 harnesses) is measured here to be dominated by "
    "PlanSeedBuilder's own statement-naming/ordering convention "
    "(`node_N`, root-last) diverging from the fixture gold renderer's "
    "convention (`nN`, root-first), independent of binding content -- it "
    "does not detect the bindings factor's marginal contribution. It is "
    "reported for continuity with those harnesses but "
    "`mean_placeholder_attachment_ratio` is the primary bindings-ceiling "
    "signal in this report.",
    "`mean_component_coverage` saturates at 1.0 once `roles`+`topology` are "
    "supplied, because oracle substitution injects the exact true role/"
    "component set by construction; this is expected for an oracle ceiling "
    "and is not evidence a learned predictor would reach the same coverage.",
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


@dataclass(frozen=True)
class Slm230Arm:
    """One oracle factor-subset arm."""

    arm_id: str
    oracle_factors: tuple[str, ...]
    description: str
    promotable: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["oracle_factors"] = list(self.oracle_factors)
        return data


def build_default_arms() -> list[Slm230Arm]:
    """Seven arms: no-plan baseline, three isolated single-factor arms, two
    cumulative arms, and the full-oracle upper bound."""
    return [
        Slm230Arm(
            arm_id="C0_no_plan",
            oracle_factors=(),
            description=(
                "No-plan baseline: empty SemanticPlanV1, no oracle "
                "substitution. The floor every other arm is measured against."
            ),
            promotable=True,
        ),
        Slm230Arm(
            arm_id="C1_roles_only",
            oracle_factors=("roles",),
            description=(
                "Only the true role_slots factor is substituted; topology "
                "and bindings stay at baseline (empty)."
            ),
        ),
        Slm230Arm(
            arm_id="C2_topology_only",
            oracle_factors=("topology",),
            description=(
                "Only the true topology (parent/child edges) factor is "
                "substituted; roles and bindings stay at baseline (empty)."
            ),
        ),
        Slm230Arm(
            arm_id="C3_bindings_only",
            oracle_factors=("bindings",),
            description=(
                "Only the true bindings (content-symbol) factor is "
                "substituted; roles and topology stay at baseline (empty)."
            ),
        ),
        Slm230Arm(
            arm_id="C4_roles_topology",
            oracle_factors=("roles", "topology"),
            description=(
                "The true roles and topology factors are substituted "
                "together (structural skeleton, no content); bindings stay "
                "at baseline."
            ),
        ),
        Slm230Arm(
            arm_id="C5_roles_topology_bindings",
            oracle_factors=("roles", "topology", "bindings"),
            description=(
                "The true roles, topology, and bindings factors are "
                "substituted together (structural skeleton plus content); "
                "archetype stays at baseline."
            ),
        ),
        Slm230Arm(
            arm_id="C6_full_gold_oracle",
            oracle_factors=("archetype", "roles", "topology", "bindings"),
            description=(
                "Full oracle upper bound: every SPV0-02 oracle factor "
                "substituted from the gold plan."
            ),
        ),
    ]


@dataclass(frozen=True)
class Slm230Record:
    """Per fixture-corpus-record diagnostics for one arm."""

    record_id: str
    arm_id: str
    seed_ok: bool
    seed_present: bool
    component_coverage: float | None
    seed_to_gold_ratio: float | None
    placeholder_attachment_ratio: float | None
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Slm230Row:
    """Aggregated row for one arm across the fixture corpus."""

    arm_id: str
    oracle_factors: tuple[str, ...]
    promotable: bool
    n_records: int
    seed_ok_count: int
    seed_valid_count: int
    seed_valid_rate: float
    mean_component_coverage: float | None
    mean_seed_to_gold_ratio: float | None
    mean_placeholder_attachment_ratio: float | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["oracle_factors"] = list(self.oracle_factors)
        return data


def _canonical_source(source: str | None) -> str | None:
    if source is None:
        return None
    try:
        program = validate(source)
        return program.serialized or source.strip()
    except Exception:  # noqa: BLE001
        return None


def _token_ratio(a: str | None, b: str | None) -> float | None:
    if a is None or b is None:
        return None
    return SequenceMatcher(None, a.split(), b.split()).ratio()


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


def _placeholder_attachment_ratio(seed: str, gold_source: str) -> float:
    """Fraction of the gold record's placeholder-argument positions the
    built seed also fills with some string-literal placeholder argument.

    Counts occurrences of the ``(":`` pattern -- a component call whose
    first positional argument is a placeholder string -- which is robust to
    PlanSeedBuilder's own statement-naming/ordering convention, unlike a
    whole-source token-overlap ratio.
    """
    gold_count = gold_source.count('(":')
    if gold_count == 0:
        return 1.0
    seed_count = seed.count('(":')
    return min(seed_count / gold_count, 1.0)


def _evaluate_record(
    spec: ProgramSpec,
    gold_plan: SemanticPlanV1,
    arm: Slm230Arm,
    pack: DslPack,
    compiler: OpenUISemanticPlanCompiler,
) -> Slm230Record:
    baseline = SemanticPlanV1(
        identity=PlanIdentity(
            pack_id=pack.pack_id,
            contract_hash=spec.contract_id,
            provenance="predicted",
        )
    )
    substitutor = PlanOracleSubstitutor(
        plan_source="gold" if arm.oracle_factors else "none",
        oracle_factors=arm.oracle_factors,
        use_mode="seed",
        honesty_mode="oracle_diagnostic",
    )
    plan = substitutor.apply(baseline, gold_plan)
    result = compiler.build_valid_seed(None, plan, pack)

    gold_source = _render_fixture_ast(spec.ast)
    coverage: float | None = None
    ratio: float | None = None
    attachment: float | None = None
    if result.ok and result.seed:
        coverage = _component_coverage(result.seed, gold_source)
        ratio = _token_ratio(_canonical_source(result.seed), _canonical_source(gold_source))
        attachment = _placeholder_attachment_ratio(result.seed, gold_source)

    return Slm230Record(
        record_id=spec.id,
        arm_id=arm.arm_id,
        seed_ok=result.ok,
        seed_present=bool(result.ok and result.seed),
        component_coverage=coverage,
        seed_to_gold_ratio=ratio,
        placeholder_attachment_ratio=attachment,
        reason=result.reason,
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(arm: Slm230Arm, records: list[Slm230Record]) -> Slm230Row:
    n = len(records)
    seed_ok = sum(1 for r in records if r.seed_ok)
    valid = [r for r in records if r.seed_present]
    notes: list[str] = []
    if not valid and arm.oracle_factors:
        reasons = sorted({r.reason for r in records if r.reason})
        notes.append(f"no valid seed on any record; reasons: {', '.join(reasons) or 'unknown'}")
    return Slm230Row(
        arm_id=arm.arm_id,
        oracle_factors=arm.oracle_factors,
        promotable=arm.promotable,
        n_records=n,
        seed_ok_count=seed_ok,
        seed_valid_count=len(valid),
        seed_valid_rate=(len(valid) / n) if n else 0.0,
        mean_component_coverage=_mean([r.component_coverage for r in valid if r.component_coverage is not None]),
        mean_seed_to_gold_ratio=_mean([r.seed_to_gold_ratio for r in valid if r.seed_to_gold_ratio is not None]),
        mean_placeholder_attachment_ratio=_mean(
            [r.placeholder_attachment_ratio for r in valid if r.placeholder_attachment_ratio is not None]
        ),
        notes=notes,
    )


def _cardinality_populated(gold_plans: list[SemanticPlanV1]) -> bool:
    return any(
        slot.min_cardinality is not None or slot.max_cardinality is not None
        for plan in gold_plans
        for slot in plan.role_slots
    )


def _resolve_disposition(rows: dict[str, Slm230Row]) -> tuple[str, str]:
    baseline = rows["C0_no_plan"]
    isolated = [rows["C1_roles_only"], rows["C2_topology_only"], rows["C3_bindings_only"]]
    combined_structural = rows["C4_roles_topology"]
    combined_full = rows["C5_roles_topology_bindings"]
    full_oracle = rows["C6_full_gold_oracle"]

    unexpected = [r for r in isolated if r.seed_valid_rate > baseline.seed_valid_rate]
    if unexpected:
        names = ", ".join(r.arm_id for r in unexpected)
        return (
            "unexpected_isolated_factor_success",
            f"Single-factor arm(s) {names} produced valid seeds above the "
            "no-plan baseline rate, contradicting the joint-requirement "
            "hypothesis; the ceiling is measurable in isolation for at "
            "least one factor.",
        )

    if combined_structural.seed_valid_rate <= baseline.seed_valid_rate:
        return (
            "no_gap_evidence",
            "The combined roles+topology arm did not raise seed validity "
            "above the no-plan baseline; no downstream ceiling gain is "
            "measurable from these factors in the current mechanism.",
        )

    structural_attach = combined_structural.mean_placeholder_attachment_ratio or 0.0
    full_attach = combined_full.mean_placeholder_attachment_ratio or 0.0
    if full_attach <= structural_attach:
        return (
            "bindings_ceiling_not_confirmed",
            "roles+topology raised seed validity as expected, but adding "
            "the bindings factor did not raise the placeholder-attachment "
            "ratio above the roles+topology-only arm; the bindings factor's "
            "marginal downstream ceiling is not confirmed by this proxy.",
        )

    return (
        "ceiling_confirmed_joint_requirement",
        "roles-only, topology-only, and bindings-only arms each stayed at "
        f"or below the no-plan baseline seed-valid rate ({baseline.seed_valid_rate:.2f}), "
        f"while roles+topology raised seed-valid rate to {combined_structural.seed_valid_rate:.2f} "
        f"and adding bindings raised mean placeholder-attachment ratio from "
        f"{structural_attach:.2f} to {full_attach:.2f} (full-oracle arm: "
        f"{full_oracle.seed_valid_rate:.2f} valid rate, "
        f"{full_oracle.mean_placeholder_attachment_ratio or 0.0:.2f} attachment). "
        "This supplies the factor-wise downstream-ceiling evidence SLM-145 "
        "and SLM-160 found missing: roles and topology are jointly (not "
        "individually) sufficient to reach the structural ceiling, and "
        "bindings is jointly sufficient to reach the content ceiling on top "
        "of that -- still wiring/fixture evidence, not a learned-head "
        "promotion.",
    )


@dataclass(frozen=True)
class Slm230Report:
    """Full fixture report for SLM-230."""

    schema: str = "Slm230PlanFactorCeilingMatrixReportV1"
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
    cardinality_populated: bool = False
    rows: tuple[Slm230Row, ...] = field(default_factory=tuple)
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
            "cardinality_populated": self.cardinality_populated,
            "rows": [r.to_dict() for r in self.rows],
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
    def from_dict(cls, data: dict[str, Any]) -> "Slm230Report":
        rows = tuple(
            Slm230Row(
                arm_id=str(r["arm_id"]),
                oracle_factors=tuple(r.get("oracle_factors", ())),
                promotable=bool(r.get("promotable", False)),
                n_records=int(r["n_records"]),
                seed_ok_count=int(r["seed_ok_count"]),
                seed_valid_count=int(r["seed_valid_count"]),
                seed_valid_rate=float(r["seed_valid_rate"]),
                mean_component_coverage=r.get("mean_component_coverage"),
                mean_seed_to_gold_ratio=r.get("mean_seed_to_gold_ratio"),
                mean_placeholder_attachment_ratio=r.get("mean_placeholder_attachment_ratio"),
                notes=list(r.get("notes", ())),
            )
            for r in data.get("rows", ())
        )
        return cls(
            schema=str(data.get("schema", "Slm230PlanFactorCeilingMatrixReportV1")),
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
            cardinality_populated=bool(data.get("cardinality_populated", False)),
            rows=rows,
            gate_hash=str(data.get("gate_hash", "")),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def run_fixture_matrix(
    *,
    corpus_size: int = 24,
    corpus_seed: int = 0,
    arms: list[Slm230Arm] | None = None,
    run_id: str | None = None,
) -> Slm230Report:
    """Run every arm through the real PlanOracleSubstitutor /
    OpenUISemanticPlanCompiler pipeline against the deterministic SLM-144
    fixture corpus."""
    arms = arms if arms is not None else build_default_arms()
    pack = get_pack("openui")
    compiler = OpenUISemanticPlanCompiler(honesty_mode="oracle_diagnostic")
    corpus = build_fixture_plan_corpus(
        count=corpus_size,
        seed=corpus_seed,
        root_containers=["Stack", "Card"],
        leaf_components=["TextContent", "Button"],
    )
    train = corpus["train"]

    rows: dict[str, Slm230Row] = {}
    for arm in arms:
        records = [_evaluate_record(spec, plan, arm, pack, compiler) for spec, plan in train]
        rows[arm.arm_id] = _aggregate(arm, records)

    disposition, rationale = _resolve_disposition(rows)

    payload = {"row_digests": sorted(_digest(row.to_dict()) for row in rows.values())}
    gate_hash = _sha256(_canonical_json(payload))

    return Slm230Report(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        corpus_size=len(train),
        corpus_seed=corpus_seed,
        cardinality_populated=_cardinality_populated([plan for _, plan in train]),
        rows=tuple(rows[arm.arm_id] for arm in arms),
        gate_hash=gate_hash,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm230_plan_factor_ceiling_matrix",
        ),
    )


def render_markdown(report: Slm230Report) -> str:
    lines = [
        f"# SLM-230 (SPV0-04): plan-factor oracle-substitution ceiling matrix ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        f"**Corpus:** SLM-144 fixture, n={report.corpus_size}, seed={report.corpus_seed}",
        f"**Cardinality fields populated by extractor:** {report.cardinality_populated}",
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
        "| arm | factors | n | seed valid rate | mean component coverage | mean seed-to-gold ratio | mean placeholder attachment | promotable |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.rows:
        factors = ", ".join(row.oracle_factors) or "(none)"

        def _fmt(v: float | None) -> str:
            return "—" if v is None else f"{v:.3f}"

        lines.append(
            f"| {row.arm_id} | {factors} | {row.n_records} | {row.seed_valid_rate:.3f} | "
            f"{_fmt(row.mean_component_coverage)} | {_fmt(row.mean_seed_to_gold_ratio)} | "
            f"{_fmt(row.mean_placeholder_attachment_ratio)} | {row.promotable} |"
        )
    lines += ["", "## Arm notes", ""]
    any_notes = False
    for row in report.rows:
        for note in row.notes:
            any_notes = True
            lines.append(f"- **{row.arm_id}**: {note}")
    if not any_notes:
        lines.append("- (none)")
    lines += [
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence only. It does not change "
        "`PlanOracleSubstitutor`, `PlanSeedBuilder`, or "
        "`OpenUISemanticPlanCompiler`, does not train or evaluate a learned "
        "head, and does not flip the SLM-160 `gold_oracle_factor_heads` "
        "disposition (`retain_diagnostic`, default off). It supplies the "
        "factor-wise ceiling evidence input SLM-145's authorization gate and "
        "SLM-160's program disposition both named as the missing next step, "
        "for a human maintainer to weigh when deciding whether to reopen "
        "SLM-145.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m scripts.run_slm230_plan_factor_ceiling_matrix --mode plan-only",
        "python -m scripts.run_slm230_plan_factor_ceiling_matrix --mode fixture",
        "```",
        "",
    ]
    return "\n".join(lines)
