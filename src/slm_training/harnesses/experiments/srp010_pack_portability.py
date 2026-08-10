"""SLM-474 (SRP-010): symbolic-regression pack portability fixture.

Fixture/wiring-only harness. Exercises the SRP-001 pack through the same
durable pack / VerifiedSynthesisProblemV1 / evidence contracts as OpenUI,
confirms OpenUI remains the default, and inventorizes honest forks /
unsupported hooks. Related catalogue identity: ``exp-sr-12`` (RSP-008) —
this harness does **not** lock or execute ``ExperimentCampaignV1`` for
promotion.

No Julia/PySR/SymPy. Does not register SR into ``dsl/packs/`` or change
``_ALIASES`` / OpenUI defaults.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.data.progspec.synthesis_capability_report import (
    build_sygus_capability_report,
)
from slm_training.data.progspec.synthesis_problem import (
    ObjectiveTermV1,
    PackIdentityV1,
    RuntimeSymbolDeclarationV1,
    VerificationRequirementV1,
    VerifiedSynthesisProblemV1,
)
from slm_training.dsl.pack import DslPack, PackSlotUnavailable, get_pack, list_packs
from slm_training.dsl.symbolic_expr_canonicalize import canonicalize_expr
from slm_training.dsl.symbolic_expr_ir import parse_problem_expr, serialize_expr
from slm_training.dsl.symbolic_expr_report import build_evidence_record
from slm_training.dsl.symbolic_regression_pack import (
    SYMBOLIC_REGRESSION_PACK_ID,
    SymbolicRegressionProblemV1,
)
from slm_training.harnesses.experiments.slm159_cross_dsl_replication import (
    PackReadinessReport,
    assess_pack_readiness,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "PORTABILITY_CAMPAIGN_ID",
    "RELATED_CATALOGUE_ID",
    "CLAIM_CLASS",
    "VERSION_STAMP_COMPONENTS",
    "SR_PACK_MODULES",
    "OPTIONAL_HOOK_MODULES",
    "MISSING_DSL_PACK_SLOTS",
    "UnsupportedHook",
    "ForkRecord",
    "DefaultIsolationCheck",
    "ImportAuditFinding",
    "PackPortabilityReport",
    "require_pack_id",
    "require_pack_slot",
    "audit_modules_for_openui_imports",
    "build_tiny_sr_problem",
    "build_vsp_envelope",
    "inventory_forks_and_hooks",
    "exercise_optional_hook_modules",
    "assess_symbolic_pack_portability",
    "run_fixture_campaign",
    "render_markdown",
]

MATRIX_SET = "slm474_srp010_pack_portability"
MATRIX_VERSION = "srp010-v2"
PORTABILITY_CAMPAIGN_ID = "slm474-srp010-pack-portability"
RELATED_CATALOGUE_ID = "exp-sr-12"
CLAIM_CLASS = "fixture"
VERSION_STAMP_COMPONENTS = (
    "harness.experiments",
    "harness.experiments.slm474_srp010_portability",
)

# Pack-owned modules for the OpenUI-import AST audit (parallel to mini_pack).
_DSL_DIR = Path(__file__).resolve().parents[2] / "dsl"
SR_PACK_MODULES: tuple[Path, ...] = tuple(
    sorted(
        p
        for p in (
            _DSL_DIR / "symbolic_regression_pack.py",
            _DSL_DIR / "symbolic_expr_ir.py",
            _DSL_DIR / "symbolic_expr_canonicalize.py",
            _DSL_DIR / "symbolic_expr_evaluator.py",
            _DSL_DIR / "symbolic_expr_fit.py",
            _DSL_DIR / "symbolic_expr_report.py",
            _DSL_DIR / "grammar" / "backends" / "symbolic_regression.py",
        )
        if p.is_file()
    )
)

OPTIONAL_HOOK_MODULES: tuple[str, ...] = (
    "slm_training.dsl.symbolic_expr_corpus",
    "slm_training.dsl.symbolic_expr_enumerate",
)

# Slots OpenUI fills that the SR reference pack intentionally omits today.
MISSING_DSL_PACK_SLOTS: tuple[str, ...] = (
    "corpus_generator",
    "completion_artifact",
    "scope_extractor",
    "prop_order",
    "incremental_engine",
)

_TINY_EXPR = "v0"
_TINY_PACK_SOURCE = "root = x\n"


@dataclass(frozen=True)
class UnsupportedHook:
    """Honest inventory of a missing or non-conforming seam."""

    hook_id: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnsupportedHook":
        return cls(
            hook_id=str(data.get("hook_id", "")),
            status=str(data.get("status", "")),
            detail=str(data.get("detail", "")),
        )


@dataclass(frozen=True)
class ForkRecord:
    """Named divergence from the OpenUI / shared-seam path."""

    fork_id: str
    surface: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForkRecord":
        return cls(
            fork_id=str(data.get("fork_id", "")),
            surface=str(data.get("surface", "")),
            detail=str(data.get("detail", "")),
        )


@dataclass(frozen=True)
class DefaultIsolationCheck:
    """Confirms OpenUI remains the unresolved / alias default."""

    unresolved_pack_id: str
    auto_pack_id: str
    default_alias_pack_id: str
    env_unset: bool
    openui_is_default: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DefaultIsolationCheck":
        return cls(
            unresolved_pack_id=str(data.get("unresolved_pack_id", "")),
            auto_pack_id=str(data.get("auto_pack_id", "")),
            default_alias_pack_id=str(data.get("default_alias_pack_id", "")),
            env_unset=bool(data.get("env_unset", False)),
            openui_is_default=bool(data.get("openui_is_default", False)),
        )


@dataclass(frozen=True)
class ImportAuditFinding:
    """One module's OpenUI-import AST audit result."""

    path: str
    offenders: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "offenders": list(self.offenders)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImportAuditFinding":
        return cls(
            path=str(data.get("path", "")),
            offenders=tuple(data.get("offenders", ())),
        )


@dataclass(frozen=True)
class PackPortabilityReport:
    """JSON-serializable SRP-010 portability fixture report."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    related_catalogue_id: str
    run_id: str
    status: str
    claim_class: str
    default_isolation: DefaultIsolationCheck
    pack_id: str
    pack_available: bool
    filled_slots: tuple[str, ...]
    missing_slots: tuple[str, ...]
    readiness: PackReadinessReport | None
    problem_fingerprint: str
    vsp_digest: str
    expression: str
    canonical_expression: str
    oracle_ok: bool
    evidence: dict[str, Any]
    sygus_capability: dict[str, Any]
    import_audit: tuple[ImportAuditFinding, ...]
    forks: tuple[ForkRecord, ...]
    unsupported_hooks: tuple[UnsupportedHook, ...]
    enumeration_exercise: dict[str, Any] = field(default_factory=dict)
    corpus_exercise: dict[str, Any] = field(default_factory=dict)
    version_stamp: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "related_catalogue_id": self.related_catalogue_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "default_isolation": self.default_isolation.to_dict(),
            "pack_id": self.pack_id,
            "pack_available": self.pack_available,
            "filled_slots": list(self.filled_slots),
            "missing_slots": list(self.missing_slots),
            "readiness": self.readiness.to_dict() if self.readiness else None,
            "problem_fingerprint": self.problem_fingerprint,
            "vsp_digest": self.vsp_digest,
            "expression": self.expression,
            "canonical_expression": self.canonical_expression,
            "oracle_ok": self.oracle_ok,
            "evidence": self.evidence,
            "sygus_capability": self.sygus_capability,
            "import_audit": [f.to_dict() for f in self.import_audit],
            "forks": [f.to_dict() for f in self.forks],
            "unsupported_hooks": [h.to_dict() for h in self.unsupported_hooks],
            "enumeration_exercise": dict(self.enumeration_exercise),
            "corpus_exercise": dict(self.corpus_exercise),
            "version_stamp": self.version_stamp,
            "notes": list(self.notes),
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PackPortabilityReport":
        readiness_raw = data.get("readiness")
        return cls(
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", PORTABILITY_CAMPAIGN_ID)),
            related_catalogue_id=str(
                data.get("related_catalogue_id", RELATED_CATALOGUE_ID)
            ),
            run_id=str(data.get("run_id", "srp010_fixture")),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", CLAIM_CLASS)),
            default_isolation=DefaultIsolationCheck.from_dict(
                data.get("default_isolation", {})
            ),
            pack_id=str(data.get("pack_id", "")),
            pack_available=bool(data.get("pack_available", False)),
            filled_slots=tuple(data.get("filled_slots", ())),
            missing_slots=tuple(data.get("missing_slots", ())),
            readiness=(
                PackReadinessReport.from_dict(readiness_raw)
                if isinstance(readiness_raw, dict)
                else None
            ),
            problem_fingerprint=str(data.get("problem_fingerprint", "")),
            vsp_digest=str(data.get("vsp_digest", "")),
            expression=str(data.get("expression", "")),
            canonical_expression=str(data.get("canonical_expression", "")),
            oracle_ok=bool(data.get("oracle_ok", False)),
            evidence=dict(data.get("evidence", {})),
            sygus_capability=dict(data.get("sygus_capability", {})),
            import_audit=tuple(
                ImportAuditFinding.from_dict(f)
                for f in data.get("import_audit", [])
            ),
            forks=tuple(ForkRecord.from_dict(f) for f in data.get("forks", [])),
            unsupported_hooks=tuple(
                UnsupportedHook.from_dict(h)
                for h in data.get("unsupported_hooks", [])
            ),
            enumeration_exercise=dict(data.get("enumeration_exercise", {})),
            corpus_exercise=dict(data.get("corpus_exercise", {})),
            version_stamp=dict(data.get("version_stamp", {})),
            notes=tuple(data.get("notes", ())),
        )


def require_pack_id(problem: SymbolicRegressionProblemV1, expected: str) -> None:
    """Fail closed when a problem's ``pack_id`` does not match *expected*."""
    if problem.pack_id != expected:
        raise ValueError(
            f"pack/contract mismatch: problem.pack_id={problem.pack_id!r} "
            f"expected {expected!r}"
        )


def require_pack_slot(pack: DslPack, slot: str) -> Any:
    """Fail closed via :meth:`DslPack.require` (raises ``PackSlotUnavailable``)."""
    return pack.require(slot)


def audit_modules_for_openui_imports(
    modules: tuple[Path, ...] | None = None,
) -> tuple[ImportAuditFinding, ...]:
    """AST-audit pack modules for ``openui`` imports (mini_pack pattern)."""
    findings: list[ImportAuditFinding] = []
    for path in modules or SR_PACK_MODULES:
        if not path.is_file():
            findings.append(
                ImportAuditFinding(path=str(path), offenders=("missing_file",))
            )
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders.extend(
                    alias.name for alias in node.names if "openui" in alias.name
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "openui" in node.module
            ):
                offenders.append(node.module)
        findings.append(
            ImportAuditFinding(path=str(path), offenders=tuple(offenders))
        )
    return tuple(findings)


def build_tiny_sr_problem() -> SymbolicRegressionProblemV1:
    """Minimal identity-fit problem for the fixture path."""
    return SymbolicRegressionProblemV1(
        variables=("x",),
        table=((0.0,), (1.0,), (2.0,)),
        targets=(0.0, 1.0, 2.0),
        allowed_operators=frozenset({"+", "*"}),
        coefficient_hole_policy="fixed_constants_only",
        complexity_budget=8,
        numeric_domain_policy="reject_nonfinite",
        train_indices=(0, 1),
        validation_indices=(2,),
        seed=0,
        provenance={"fixture": PORTABILITY_CAMPAIGN_ID},
    )


def build_vsp_envelope(
    problem: SymbolicRegressionProblemV1,
) -> VerifiedSynthesisProblemV1:
    """Matching :class:`VerifiedSynthesisProblemV1` envelope for *problem*."""
    require_pack_id(problem, SYMBOLIC_REGRESSION_PACK_ID)
    return VerifiedSynthesisProblemV1(
        problem_id=f"srp010-{problem.fingerprint[:12]}",
        pack_identity=PackIdentityV1(pack_id=problem.pack_id),
        runtime_symbols=tuple(
            RuntimeSymbolDeclarationV1(surface=name, role="input_variable")
            for name in problem.variables
        ),
        verification_requirements=(
            VerificationRequirementV1(
                requirement_id="r_eval",
                kind="evaluator",
                evaluator_version="v1",
            ),
        ),
        objective=(
            ObjectiveTermV1(name="fit_error", weight=1.0, direction="minimize"),
            ObjectiveTermV1(name="complexity", weight=0.1, direction="minimize"),
        ),
        provenance={
            "symbolic_regression_problem_fingerprint": problem.fingerprint,
            "fixture": PORTABILITY_CAMPAIGN_ID,
        },
    )


def _check_default_isolation() -> DefaultIsolationCheck:
    """Confirm unresolved / ``auto`` / ``default`` still resolve to OpenUI."""
    env_unset = not bool(os.environ.get("SLM_GRAMMAR_DSL"))
    unresolved = get_pack(None).pack_id
    auto = get_pack("auto").pack_id
    default_alias = get_pack("default").pack_id
    return DefaultIsolationCheck(
        unresolved_pack_id=unresolved,
        auto_pack_id=auto,
        default_alias_pack_id=default_alias,
        env_unset=env_unset,
        openui_is_default=(
            env_unset
            and unresolved == "openui"
            and auto == "openui"
            and default_alias == "openui"
        ),
    )


# Tiny fixture budgets — exercise only, not ship-scale enumeration/corpus.
_ENUM_MAX_DEPTH = 2
_ENUM_MAX_NODES = 3
_ENUM_MAX_STATES = 100
_CORPUS_SEED = 474
_CORPUS_NUM_PROBLEMS = 4


def _unsupported_import_hook(module_name: str, exc: BaseException) -> UnsupportedHook:
    short = module_name.rsplit(".", 1)[-1]
    if isinstance(exc, ModuleNotFoundError):
        detail = (
            f"{module_name} not on main ({exc.name or module_name}); "
            "SRP-008/SRP-009 corpus/baseline remain optional"
        )
    else:
        detail = f"{module_name} import failed: {exc}"
    return UnsupportedHook(hook_id=short, status="unsupported", detail=detail)


def _exercise_enumerate_module(
    problem: SymbolicRegressionProblemV1,
) -> tuple[UnsupportedHook, dict[str, Any]]:
    """Call the public enumerative baseline on a tiny budget when importable."""
    module_name = "slm_training.dsl.symbolic_expr_enumerate"
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - inventory only
        return _unsupported_import_hook(module_name, exc), {}

    result = mod.run_enumerative_baseline(
        problem,
        max_depth=_ENUM_MAX_DEPTH,
        max_nodes=_ENUM_MAX_NODES,
        max_states=_ENUM_MAX_STATES,
    )
    counters = result.counters.to_dict()
    summary = {
        "status": "available",
        "exercised": True,
        "schema_version": result.schema_version,
        "max_depth": result.max_depth_budget,
        "max_nodes": result.max_nodes_budget,
        "optimality_claim": result.optimality_claim,
        "bound_exhausted": bool(counters.get("bound_exhausted")),
        "states_generated": int(counters.get("states_generated", 0)),
        "unique_canonical_states": int(counters.get("unique_canonical_states", 0)),
        "duplicate_eliminations": int(counters.get("duplicate_eliminations", 0)),
        "evaluation_count": int(counters.get("evaluation_count", 0)),
        "pack_slot_corpus_generator": None,
    }
    hook = UnsupportedHook(
        hook_id="symbolic_expr_enumerate",
        status="available",
        detail=(
            f"{module_name} exercised "
            f"(optimality_claim={result.optimality_claim!r}, "
            f"states_generated={summary['states_generated']}, "
            f"unique={summary['unique_canonical_states']}); "
            "DslPack.corpus_generator remains unset"
        ),
    )
    return hook, summary


def _exercise_corpus_module() -> tuple[UnsupportedHook, dict[str, Any]]:
    """Generate a tiny deterministic corpus + leakage audit when importable."""
    module_name = "slm_training.dsl.symbolic_expr_corpus"
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - inventory only
        return _unsupported_import_hook(module_name, exc), {}

    from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

    config = mod.CorpusGenerationConfigV1(
        schema_version=mod.CORPUS_SCHEMA_VERSION,
        seed=_CORPUS_SEED,
        num_problems=_CORPUS_NUM_PROBLEMS,
        num_variables_range=(1, 1),
        allowed_operators=frozenset({"+", "*"}),
        max_basis_depth=2,
        max_coefficient_terms=1,
        coefficient_probability=0.0,
        numeric_domain_policy="reject_nonfinite",
        complexity_budget=16,
        train_rows=4,
        validation_rows=2,
        extrapolation_rows=2,
        train_domain=(-2.0, 2.0),
        extrapolation_domain=(3.0, 5.0),
        coefficient_range=(-2.0, 2.0),
        split_policy=RootFamilySplitPolicyV1(
            modulus=4, validation_buckets=(2,), test_buckets=(3,)
        ),
    )
    corpus = mod.generate_corpus(config)
    audit = mod.audit_corpus_leakage(corpus)
    summary = {
        "status": "available",
        "exercised": True,
        "schema_version": corpus.schema_version,
        "seed": corpus.seed,
        "num_problems": len(corpus.records),
        "rejection_count": len(corpus.rejections),
        "corpus_fingerprint": corpus.corpus_fingerprint,
        "leakage_is_clean": bool(audit.is_clean),
        "cross_split_family_violations": len(audit.cross_split_family_violations),
        "duplicate_family_ids": len(audit.duplicate_family_ids),
        "pack_slot_corpus_generator": None,
    }
    hook = UnsupportedHook(
        hook_id="symbolic_expr_corpus",
        status="available",
        detail=(
            f"{module_name} exercised "
            f"(n={summary['num_problems']}, seed={_CORPUS_SEED}, "
            f"rejections={summary['rejection_count']}, "
            f"leakage_clean={summary['leakage_is_clean']}); "
            "DslPack.corpus_generator remains unset"
        ),
    )
    return hook, summary


def exercise_optional_hook_modules(
    problem: SymbolicRegressionProblemV1,
) -> tuple[tuple[UnsupportedHook, ...], dict[str, Any], dict[str, Any]]:
    """Import + exercise SRP-008/009 modules when present; never wire pack slots."""
    enum_hook, enum_summary = _exercise_enumerate_module(problem)
    corpus_hook, corpus_summary = _exercise_corpus_module()
    return (enum_hook, corpus_hook), enum_summary, corpus_summary


def inventory_forks_and_hooks(
    pack: DslPack,
    *,
    sygus_conformance: str,
    problem: SymbolicRegressionProblemV1 | None = None,
) -> tuple[
    tuple[ForkRecord, ...],
    tuple[UnsupportedHook, ...],
    dict[str, Any],
    dict[str, Any],
]:
    """Honest forks + optional-hook inventory/exercise (never silent)."""
    forks: list[ForkRecord] = [
        ForkRecord(
            fork_id="shadow_dsl_packs_registry",
            surface="dsl/packs/",
            detail=(
                "Canonical registration is dsl/pack.py "
                f"(list_packs includes {SYMBOLIC_REGRESSION_PACK_ID!r}); the "
                "legacy dsl/packs/ shadow registry still only ships "
                "openui/toy-layout/arith_sketch and does not list SR."
            ),
        ),
        ForkRecord(
            fork_id="openui_pinned_language_contract",
            surface="dsl/language_contract.py",
            detail=(
                "LanguageContract / OUTPUT_CONTRACT_VERSION remain OpenUI-pinned "
                "(LANG_SPEC=openui-lang-0.2.x); SR uses its own problem schema "
                "rather than that contract identity."
            ),
        ),
        ForkRecord(
            fork_id="sygus_inspired_not_conformance",
            surface="data/progspec/synthesis_capability_report.py",
            detail=(
                f"SyGuS capability report conformance={sygus_conformance!r} "
                "(inspired interoperability only, not SyGuS-IF standards "
                "conformance)."
            ),
        ),
    ]
    exercise_problem = problem if problem is not None else build_tiny_sr_problem()
    optional_hooks, enumeration_exercise, corpus_exercise = (
        exercise_optional_hook_modules(exercise_problem)
    )
    hooks: list[UnsupportedHook] = list(optional_hooks)
    for slot in MISSING_DSL_PACK_SLOTS:
        try:
            pack.require(slot)
            hooks.append(
                UnsupportedHook(
                    hook_id=f"slot.{slot}",
                    status="available",
                    detail=f"pack provides {slot}",
                )
            )
        except PackSlotUnavailable:
            hooks.append(
                UnsupportedHook(
                    hook_id=f"slot.{slot}",
                    status="unsupported",
                    detail=(
                        f"DslPack slot {slot!r} is None on "
                        f"{SYMBOLIC_REGRESSION_PACK_ID} (honest partial pack)"
                    ),
                )
            )
    return tuple(forks), tuple(hooks), enumeration_exercise, corpus_exercise


def assess_symbolic_pack_portability(
    *,
    run_id: str = "srp010_fixture",
) -> PackPortabilityReport:
    """Run the SRP-010 end-to-end symbolic-pack portability fixture."""
    notes: list[str] = [
        "claim_class=fixture; no ExperimentCampaignV1 lock/execute for promotion",
        f"related catalogue identity: {RELATED_CATALOGUE_ID}",
    ]

    default_isolation = _check_default_isolation()
    if not default_isolation.openui_is_default:
        notes.append(
            "default isolation failed: unresolved/auto/default did not stay openui "
            f"with SLM_GRAMMAR_DSL unset ({default_isolation.to_dict()})"
        )

    pack_available = SYMBOLIC_REGRESSION_PACK_ID in set(list_packs())
    if not pack_available:
        raise RuntimeError(
            f"pack {SYMBOLIC_REGRESSION_PACK_ID!r} missing from canonical registry"
        )
    pack = get_pack(SYMBOLIC_REGRESSION_PACK_ID)

    readiness = assess_pack_readiness(SYMBOLIC_REGRESSION_PACK_ID)
    filled = pack.filled_slots()
    missing = tuple(s for s in MISSING_DSL_PACK_SLOTS if getattr(pack, s) is None)

    problem = build_tiny_sr_problem()
    require_pack_id(problem, SYMBOLIC_REGRESSION_PACK_ID)
    envelope = build_vsp_envelope(problem)

    node = parse_problem_expr(problem, _TINY_EXPR)
    canonical = serialize_expr(canonicalize_expr(node))
    oracle = require_pack_slot(pack, "oracle")
    canonicalize = require_pack_slot(pack, "canonicalize")
    oracle(_TINY_PACK_SOURCE)
    canonicalize(_TINY_PACK_SOURCE)
    evidence = build_evidence_record(node, problem).to_dict()
    sygus = build_sygus_capability_report(envelope).to_dict()
    import_audit = audit_modules_for_openui_imports()
    openui_offenders = [f for f in import_audit if f.offenders]
    if openui_offenders:
        notes.append(
            "import audit found openui imports: "
            + ", ".join(f"{f.path}:{list(f.offenders)}" for f in openui_offenders)
        )

    forks, unsupported_hooks, enumeration_exercise, corpus_exercise = (
        inventory_forks_and_hooks(
            pack,
            sygus_conformance=str(sygus.get("conformance", "")),
            problem=problem,
        )
    )

    status = "fixture"
    if not default_isolation.openui_is_default or openui_offenders:
        status = "fixture_failed"

    return PackPortabilityReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=PORTABILITY_CAMPAIGN_ID,
        related_catalogue_id=RELATED_CATALOGUE_ID,
        run_id=run_id,
        status=status,
        claim_class=CLAIM_CLASS,
        default_isolation=default_isolation,
        pack_id=pack.pack_id,
        pack_available=True,
        filled_slots=filled,
        missing_slots=missing,
        readiness=readiness,
        problem_fingerprint=problem.fingerprint,
        vsp_digest=envelope.digest,
        expression=_TINY_EXPR,
        canonical_expression=canonical,
        oracle_ok=True,
        evidence=evidence,
        sygus_capability=sygus,
        import_audit=import_audit,
        forks=forks,
        unsupported_hooks=unsupported_hooks,
        enumeration_exercise=enumeration_exercise,
        corpus_exercise=corpus_exercise,
        version_stamp=build_version_stamp(*VERSION_STAMP_COMPONENTS),
        notes=tuple(notes),
    )


def run_fixture_campaign(
    *,
    run_id: str = "srp010_fixture",
    output_dir: Path | None = None,
) -> PackPortabilityReport:
    """Alias for :func:`assess_symbolic_pack_portability` with optional write."""
    report = assess_symbolic_pack_portability(run_id=run_id)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "srp010_pack_portability_report.json")
    return report


def render_markdown(report: PackPortabilityReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-474 (SRP-010): Symbolic-pack portability fixture ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`",
        "",
        f"Version: `{report.matrix_version}`",
        "",
        f"Status: **{report.status}**",
        "",
        f"**Claim class:** `{report.claim_class}` only. Related catalogue "
        f"identity `{report.related_catalogue_id}` is cited, not locked or "
        "executed as ExperimentCampaignV1 for promotion. No Julia/PySR/SymPy.",
        "",
        "## Default isolation",
        "",
        f"- `SLM_GRAMMAR_DSL` unset: `{report.default_isolation.env_unset}`",
        f"- `get_pack()` → `{report.default_isolation.unresolved_pack_id}`",
        f"- `get_pack('auto')` → `{report.default_isolation.auto_pack_id}`",
        f"- `get_pack('default')` → `{report.default_isolation.default_alias_pack_id}`",
        f"- OpenUI is default: `{report.default_isolation.openui_is_default}`",
        "",
        "## Pack path",
        "",
        f"- Pack: `{report.pack_id}` available=`{report.pack_available}`",
        f"- Filled slots: {', '.join(report.filled_slots) or '(none)'}",
        f"- Missing reference slots: {', '.join(report.missing_slots) or '(none)'}",
        f"- Problem fingerprint: `{report.problem_fingerprint[:16]}…`",
        f"- VSP digest: `{report.vsp_digest[:16]}…`",
        f"- Expression `{report.expression}` → canonical "
        f"`{report.canonical_expression}`; oracle_ok=`{report.oracle_ok}`",
        f"- SyGuS conformance: "
        f"`{report.sygus_capability.get('conformance', '')}`",
        "",
        "## Forks",
        "",
        "| Fork | Surface | Detail |",
        "| --- | --- | --- |",
    ]
    for fork in report.forks:
        lines.append(f"| {fork.fork_id} | {fork.surface} | {fork.detail} |")
    lines.extend(
        [
            "",
            "## Unsupported hooks",
            "",
            "| Hook | Status | Detail |",
            "| --- | --- | --- |",
        ]
    )
    for hook in report.unsupported_hooks:
        lines.append(f"| {hook.hook_id} | {hook.status} | {hook.detail} |")
    lines.extend(
        [
            "",
            "## Optional module exercise (SRP-008/009)",
            "",
            (
                f"- Enumerate: exercised=`{report.enumeration_exercise.get('exercised', False)}` "
                f"optimality_claim=`{report.enumeration_exercise.get('optimality_claim', '')}` "
                f"states=`{report.enumeration_exercise.get('states_generated', '')}` "
                f"unique=`{report.enumeration_exercise.get('unique_canonical_states', '')}`"
            ),
            (
                f"- Corpus: exercised=`{report.corpus_exercise.get('exercised', False)}` "
                f"n=`{report.corpus_exercise.get('num_problems', '')}` "
                f"rejections=`{report.corpus_exercise.get('rejection_count', '')}` "
                f"leakage_clean=`{report.corpus_exercise.get('leakage_is_clean', '')}`"
            ),
            "- Pack slots `corpus_generator` / `completion_artifact` stay intentionally empty.",
            "",
            "## Import audit (no OpenUI imports)",
            "",
            "| Module | Offenders |",
            "| --- | --- |",
        ]
    )
    for finding in report.import_audit:
        offenders = ", ".join(finding.offenders) if finding.offenders else "—"
        lines.append(f"| `{Path(finding.path).name}` | {offenders} |")
    if report.notes:
        lines.extend(["", "## Notes", ""])
        for note in report.notes:
            lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "Fixture wiring only. The symbolic-regression pack exercises "
            "parse/canonicalize/oracle/evaluate-fit/evidence, optional "
            "enumerate/corpus modules when importable, and the VSP/SyGuS "
            "capability export without changing OpenUI defaults. Real "
            f"{RELATED_CATALOGUE_ID} certification remains a separate campaign.",
            "",
        ]
    )
    return "\n".join(lines)
