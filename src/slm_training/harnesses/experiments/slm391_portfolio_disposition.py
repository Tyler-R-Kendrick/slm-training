"""SLM-391 (DSH4-06): staged DSL harness portfolio disposition.

One artifact-bound final disposition over the DSH capability claims:
``cap0_grammar``, ``cap1_semantics``, ``cap2_operators``, ``distillation``,
``efficiency``, and ``trace_evals``.  For every claim this harness

* enumerates the evidence artifacts present on this lineage (path + sha256 +
  schema + identities where available),
* runs the applicable frozen suites available on this lineage at **fixture
  scale** (SLM-367 two-pack CAP1 suite gates, the SLM-368 matched-control
  experiment, the DSH3 operator legal-set suite, the SLM-390 trace-eval
  lifecycle, strict-v2 / binding-aware checks, and an AgentV diagnostic), and
* classifies the claim ``supported | rejected | invalid | unknown``.

Stop rule: a claim whose required artifacts are missing or incomparable is
``unknown`` and is **never** normalized through unsupported assumptions.
Negative / invalid / timeout / no-effect / contaminated / prediction-identical
outcomes stay first-class evidence rows.  Fixture-scale wiring may support a
``harness_contract`` claim with explicit bounds; it never supports a
``model_capability`` or ``ship`` claim, so compression/syntax/latency/fixture
successes cannot be reported as held-out semantic or ship capability.

Systems/efficiency statements are normalized separately: every efficiency row
must name its workload, hardware, batch, model-call, node-pass, and output
identities; without that identity an efficiency claim is ``unknown``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from slm_training.harness_core.lineage.records import content_sha
from slm_training.versioning import UNKNOWN, build_version_stamp, git_commit

SCHEMA_VERSION = "slm391_portfolio_disposition/v1"
DISPOSITION_SCHEMA = "StagedDslHarnessDispositionV1"
EXPERIMENT_ID = "slm391-portfolio-disposition"
MATRIX_SET = "slm391_portfolio_disposition"

CLAIM_CAP0 = "cap0_grammar"
CLAIM_CAP1 = "cap1_semantics"
CLAIM_CAP2 = "cap2_operators"
CLAIM_DISTILLATION = "distillation"
CLAIM_EFFICIENCY = "efficiency"
CLAIM_TRACE = "trace_evals"
CLAIM_IDS: tuple[str, ...] = (
    CLAIM_CAP0,
    CLAIM_CAP1,
    CLAIM_CAP2,
    CLAIM_DISTILLATION,
    CLAIM_EFFICIENCY,
    CLAIM_TRACE,
)

VERDICT_SUPPORTED = "supported"
VERDICT_REJECTED = "rejected"
VERDICT_INVALID = "invalid"
VERDICT_UNKNOWN = "unknown"
VERDICTS: tuple[str, ...] = (
    VERDICT_SUPPORTED,
    VERDICT_REJECTED,
    VERDICT_INVALID,
    VERDICT_UNKNOWN,
)

#: Claim scopes.  Fixture-scale wiring may support ``harness_contract`` claims
#: only; ``model_capability`` and ``ship`` claims can never be supported by
#: fixture-only evidence rows (guard below).
SCOPE_HARNESS_CONTRACT = "harness_contract"
SCOPE_MODEL_CAPABILITY = "model_capability"
SCOPE_SHIP = "ship"
FIXTURE_FORBIDDEN_SCOPES = (SCOPE_MODEL_CAPABILITY, SCOPE_SHIP)

#: First-class outcome classes; none of these are dropped or folded away.
OUTCOME_POSITIVE = "positive"
OUTCOME_NEGATIVE = "negative"
OUTCOME_INVALID = "invalid"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_NO_EFFECT = "no_effect"
OUTCOME_CONTAMINATED = "contaminated"
OUTCOME_PREDICTION_IDENTICAL = "prediction_identical"
OUTCOME_CLASSES: tuple[str, ...] = (
    OUTCOME_POSITIVE,
    OUTCOME_NEGATIVE,
    OUTCOME_INVALID,
    OUTCOME_TIMEOUT,
    OUTCOME_NO_EFFECT,
    OUTCOME_CONTAMINATED,
    OUTCOME_PREDICTION_IDENTICAL,
)

#: Fixture-scale budget for the matched CAP1 experiment inside this run.
DEFAULT_CAP1_STEPS = 4
DEFAULT_CAP1_RECORDS_PER_ARM = 8
DEFAULT_CAP1_SEEDS: tuple[int, ...] = (0,)


class DispositionError(RuntimeError):
    """Disposition construction or guard failure."""


# --------------------------------------------------------------------------
# Claim registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimSpecV1:
    """One preregistered portfolio claim and its artifact probes."""

    claim_id: str
    capability: str
    scope: str  # harness_contract | model_capability | ship
    hypothesis: str
    #: Repo-relative paths that must exist for the claim to be classifiable.
    required_artifacts: tuple[str, ...]
    #: Repo-relative glob patterns (evaluated under the repo root); at least
    #: one match each is required when listed in ``required_globs``.
    required_globs: tuple[str, ...] = ()
    #: Optional evidence probes (absence is recorded, never fatal).
    optional_artifacts: tuple[str, ...] = ()
    optional_globs: tuple[str, ...] = ()
    stop_rule: str = "missing or incomparable artifacts classify unknown; never normalize by assumption"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def claim_registry() -> tuple[ClaimSpecV1, ...]:
    """The fixed SLM-391 claim registry (six claims, preregistered order)."""
    return (
        ClaimSpecV1(
            claim_id=CLAIM_CAP0,
            capability="CAP0 exact grammar reproduction",
            scope=SCOPE_MODEL_CAPABILITY,
            hypothesis=(
                "A trained model reproduces grammar-exact outputs on the frozen "
                "CAP0 suite (SLM-361) with the SLM-362 matched experiment."
            ),
            required_artifacts=(
                "src/slm_training/harnesses/train_data/cap0_eval_freeze.py",
                "src/slm_training/harnesses/experiments/slm362_cap0_experiment.py",
            ),
            required_globs=(
                "docs/design/iter-slm361-*.json",
                "docs/design/iter-slm362-*.json",
            ),
        ),
        ClaimSpecV1(
            claim_id=CLAIM_CAP1,
            capability="CAP1 held-out semantic grounding",
            scope=SCOPE_MODEL_CAPABILITY,
            hypothesis=(
                "Schema-grounded training beats schema-absent and style controls "
                "on the frozen two-pack suite with preregistered paired evidence "
                "(CERT_CAP1)."
            ),
            required_artifacts=(
                "src/slm_training/harnesses/train_data/cap1_two_pack_freeze.py",
                "src/slm_training/harnesses/experiments/slm368_cap1_experiment.py",
                "docs/design/iter-slm367-two-pack-freeze-20260725.json",
                "docs/design/iter-slm368-cap1-experiment-20260725.json",
            ),
        ),
        ClaimSpecV1(
            claim_id=CLAIM_CAP2,
            capability="CAP2 operator legal-set enumeration",
            scope=SCOPE_HARNESS_CONTRACT,
            hypothesis=(
                "The DSH3 operator stack enumerates exact bounded legal sets "
                "with supported/unsupported/unknown verdicts and reserved "
                "serialization; natural-language operator transforms stay "
                "closed until CERT_CAP1."
            ),
            required_artifacts=(
                "src/slm_training/dsl/operators/legal_set.py",
                "src/slm_training/evals/cap2_disposition.py",
            ),
            optional_globs=("docs/design/iter-*cap2-disposition*.json",),
        ),
        ClaimSpecV1(
            claim_id=CLAIM_DISTILLATION,
            capability="DSH4 distillation (teacher -> student action programs)",
            scope=SCOPE_MODEL_CAPABILITY,
            hypothesis=(
                "Typed action distillation (DSH4-01/DSH4-02) improves held-out "
                "semantic outcomes over matched controls."
            ),
            required_globs=(
                "docs/design/iter-dsh4-01*.json",
                "docs/design/iter-dsh4-02*.json",
            ),
            required_artifacts=(),
            optional_artifacts=(
                "src/slm_training/harnesses/experiments/scaffold_distillation_activation.py",
            ),
        ),
        ClaimSpecV1(
            claim_id=CLAIM_EFFICIENCY,
            capability="Systems/efficiency (latency, work, calls)",
            scope=SCOPE_MODEL_CAPABILITY,
            hypothesis=(
                "A staged DSL harness configuration reduces normalized work "
                "(latency / model calls / node passes / compiler-verifier work) "
                "versus a matched baseline on a named workload and hardware."
            ),
            required_artifacts=(),
            optional_artifacts=(
                "docs/design/perf-experiment-matrix.md",
                "src/slm_training/harnesses/experiments/efficiency_gain.py",
            ),
            stop_rule=(
                "no efficiency claim without workload/hardware/batch/model-call/"
                "node-pass/output-length/fallback identity; unnormalized or "
                "baseline-free measurements classify unknown"
            ),
        ),
        ClaimSpecV1(
            claim_id=CLAIM_TRACE,
            capability="Trace-to-eval candidate lifecycle (SLM-390)",
            scope=SCOPE_HARNESS_CONTRACT,
            hypothesis=(
                "Runtime traces become quarantined eval candidates that only "
                "advance through explicit review/freeze/confirmation/promotion "
                "transitions; unsafe traces stay quarantined."
            ),
            required_artifacts=(
                "src/slm_training/harnesses/experiments/slm390_trace_evals.py",
                "docs/design/iter-slm390-trace-evals-20260725.json",
                "docs/design/iter-slm390-trace-evals-20260725.md",
            ),
        ),
    )


# --------------------------------------------------------------------------
# Artifact binding
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactBindingV1:
    path: str
    present: bool
    required: bool
    sha256: str
    schema: str
    identities: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "present": self.present,
            "required": self.required,
            "sha256": self.sha256,
            "schema": self.schema,
            "identities": dict(self.identities),
        }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identities_for(path: Path) -> dict[str, Any]:
    """Extract declared identities from a JSON evidence artifact when present."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, Any] = {}
    for key in (
        "schema",
        "schema_version",
        "experiment_id",
        "claim_class",
        "status",
        "preregistration_sha256",
        "corpus_sha256",
        "data_manifest_sha",
    ):
        if key in data:
            out[key] = data[key]
    suite = data.get("suite")
    if isinstance(suite, dict) and suite.get("suite_sha256"):
        out["suite_sha256"] = suite["suite_sha256"]
    stamp = data.get("version_stamp")
    if isinstance(stamp, dict) and stamp.get("commit"):
        out["evidence_commit"] = stamp["commit"]
    return out


def bind_artifacts(spec: ClaimSpecV1, repo_root: Path) -> tuple[ArtifactBindingV1, ...]:
    """Bind every probed artifact for one claim; absence is first-class."""
    bindings: list[ArtifactBindingV1] = []

    def _bind(path: Path, *, required: bool, display: str) -> None:
        present = path.is_file()
        bindings.append(
            ArtifactBindingV1(
                path=display,
                present=present,
                required=required,
                sha256=_sha256_file(path) if present else "",
                schema=(
                    str(_identities_for(path).get("schema", "")) if present else ""
                ),
                identities=_identities_for(path) if present else {},
            )
        )

    for rel in spec.required_artifacts:
        _bind(repo_root / rel, required=True, display=rel)
    for pattern in spec.required_globs:
        matches = sorted(repo_root.glob(pattern))
        if matches:
            for match in matches:
                _bind(match, required=True, display=match.relative_to(repo_root).as_posix())
        else:
            bindings.append(
                ArtifactBindingV1(
                    path=pattern, present=False, required=True, sha256="", schema=""
                )
            )
    for rel in spec.optional_artifacts:
        _bind(repo_root / rel, required=False, display=rel)
    for pattern in spec.optional_globs:
        matches = sorted(repo_root.glob(pattern))
        for match in matches:
            _bind(match, required=False, display=match.relative_to(repo_root).as_posix())
        if not matches:
            bindings.append(
                ArtifactBindingV1(
                    path=pattern, present=False, required=False, sha256="", schema=""
                )
            )
    return tuple(bindings)


# --------------------------------------------------------------------------
# Evidence rows + workload identity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkloadIdentityV1:
    """Normalization identity for systems/efficiency statements."""

    workload: str
    hardware: str
    python: str
    batch_size: int | None
    model_calls: int | None
    node_passes: int | None
    compiler_verifier_work: str
    output_chars: int | None
    fallback_count: int
    timeout_count: int

    def complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.batch_size,
                self.model_calls,
                self.node_passes,
                self.output_chars,
            )
        ) and bool(self.workload and self.hardware)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRowV1:
    row_id: str
    claim_id: str
    suite: str
    system: str
    outcome_class: str  # one of OUTCOME_CLASSES
    fixture_only: bool
    metrics: Mapping[str, Any]
    workload_identity: Mapping[str, Any] | None
    notes: str

    def __post_init__(self) -> None:
        if self.outcome_class not in OUTCOME_CLASSES:
            raise DispositionError(f"undeclared outcome class {self.outcome_class!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "claim_id": self.claim_id,
            "suite": self.suite,
            "system": self.system,
            "outcome_class": self.outcome_class,
            "fixture_only": self.fixture_only,
            "metrics": dict(self.metrics),
            "workload_identity": (
                dict(self.workload_identity) if self.workload_identity else None
            ),
            "notes": self.notes,
        }


def _row_id(*parts: str) -> str:
    return content_sha({"parts": list(parts)})[:16]


# --------------------------------------------------------------------------
# Suite runners (all fixture scale, honest about it)
# --------------------------------------------------------------------------


def run_two_pack_gate_checks() -> list[EvidenceRowV1]:
    """SLM-367 two-pack suite gates under the oracle ceiling and controls.

    The oracle decide_fn (exact target replay) is the strongest retained
    *configuration ceiling*; the abstain and constant-prediction controls are
    matched negative / prediction-identical controls.  No model claim results.
    """
    from slm_training.harnesses.train_data.cap1_two_pack_freeze import (
        ABSTAIN,
        SuiteRowV1,
        adjudicate_suite,
        build_cap1_two_pack_suite,
        evaluate_gates,
        verify_suite_integrity,
    )

    suite = build_cap1_two_pack_suite()
    integrity = verify_suite_integrity(suite)
    rows: list[EvidenceRowV1] = []

    def _gate_row(system: str, decide_fn, outcome: str, notes: str) -> EvidenceRowV1:
        adjudication = adjudicate_suite(suite, decide_fn)
        gate = evaluate_gates(suite, adjudication)
        return EvidenceRowV1(
            row_id=_row_id("two-pack", system),
            claim_id=CLAIM_CAP1,
            suite="cap1_two_pack_freeze/v1",
            system=system,
            outcome_class=outcome,
            fixture_only=True,
            metrics={
                "suite_sha256": suite.suite_sha256,
                "integrity_ok": integrity.ok,
                "integrity_violations": list(integrity.violations),
                "gate_passed": gate.passed,
                "gate_failures": list(gate.failures),
                "canonical_accuracy": adjudication.canonical_accuracy,
                "cap0_retention_accuracy": float(
                    adjudication.cap0_retention["accuracy"]
                ),
            },
            workload_identity=None,
            notes=notes,
        )

    rows.append(
        _gate_row(
            "oracle_ceiling",
            lambda row: (
                ABSTAIN
                if row.stratum in ("schema_empty", "schema_contradictory")
                else (row.targets[0] if row.targets else ABSTAIN)
            ),
            OUTCOME_POSITIVE if integrity.ok else OUTCOME_INVALID,
            "schema-consuming exact-target replay ceiling; harness-contract "
            "evidence only (relevant schema perturbations abstain)",
        )
    )
    rows.append(
        _gate_row(
            "abstain_control",
            lambda row: ABSTAIN,
            OUTCOME_NEGATIVE,
            "matched control: universal abstention must fail the gates",
        )
    )

    def _constant(row: SuiteRowV1) -> str:
        return "root = Stack([])"

    rows.append(
        _gate_row(
            "constant_prediction_control",
            _constant,
            OUTCOME_PREDICTION_IDENTICAL,
            "prediction-identical control: one constant program for every row",
        )
    )
    return rows


def run_cap1_matched_experiment(
    *,
    steps: int = DEFAULT_CAP1_STEPS,
    records_per_arm: int = DEFAULT_CAP1_RECORDS_PER_ARM,
    seeds: Sequence[int] = DEFAULT_CAP1_SEEDS,
) -> tuple[list[EvidenceRowV1], dict[str, Any], WorkloadIdentityV1]:
    """SLM-368 matched-control CAP1 experiment at reduced fixture scale.

    Returns (evidence rows, condensed report, workload identity).  The
    certificate decision (including underpowered/contaminated/prediction-
    identical stop codes) is preserved verbatim as first-class rows.
    """
    from slm_training.harnesses.experiments import slm368_cap1_experiment as cap1

    with tempfile.TemporaryDirectory(prefix="slm391_cap1_") as tmp:
        report = cap1.run_experiment(
            output_dir=Path(tmp),
            seeds=tuple(seeds),
            records_per_arm=records_per_arm,
            steps=steps,
        )
    rows: list[EvidenceRowV1] = []
    contamination = list(report["suite"]["contamination_findings"])
    rows.append(
        EvidenceRowV1(
            row_id=_row_id("cap1", "overlap-audit"),
            claim_id=CLAIM_CAP1,
            suite="cap1_two_pack_freeze/v1",
            system="train_eval_overlap_audit",
            outcome_class=OUTCOME_CONTAMINATED if contamination else OUTCOME_NEGATIVE,
            fixture_only=True,
            metrics={
                "contamination_findings": contamination,
                "suite_integrity_ok": report["suite"]["integrity_ok"],
            },
            workload_identity=None,
            notes=(
                "eval/training overlap detection: no shared root families or "
                "targets" if not contamination else "train/eval overlap detected"
            ),
        )
    )
    for arm in report["preregistration"]["arms"]:
        primary = report["arms"][arm]["seeds"][0]
        score = primary["score"]
        rows.append(
            EvidenceRowV1(
                row_id=_row_id("cap1", arm),
                claim_id=CLAIM_CAP1,
                suite="cap1_two_pack_freeze/v1",
                system=f"arm:{arm}",
                outcome_class=OUTCOME_POSITIVE if score["gate_passed"] else OUTCOME_NEGATIVE,
                fixture_only=True,
                metrics={
                    "canonical_accuracy": score["canonical_accuracy"],
                    "facts_accuracy": score["facts_accuracy"],
                    "relevant_sensitivity": score["relevant_sensitivity"],
                    "invariant_rate": score["invariant_rate"],
                    "cap0_retention_accuracy": score["cap0_retention_accuracy"],
                    "gate_failures": list(score["gate_failures"]),
                    "checkpoint_sha256": primary["checkpoint_sha256"],
                },
                workload_identity=None,
                notes="matched-exposure arm; fixture scale",
            )
        )
    certificate = report["certificate"]
    rows.append(
        EvidenceRowV1(
            row_id=_row_id("cap1", "certificate"),
            claim_id=CLAIM_CAP1,
            suite="cap1_two_pack_freeze/v1",
            system="CERT_CAP1",
            outcome_class=(
                OUTCOME_POSITIVE
                if certificate["status"] == "issued"
                else OUTCOME_NEGATIVE
            ),
            fixture_only=True,
            metrics={
                "status": certificate["status"],
                "certificate": certificate["certificate"],
                "rejection_codes": list(certificate["rejection_codes"]),
                "paired_n": certificate["artifact"].get("paired_n", 0),
            },
            workload_identity=None,
            notes="certificate decision preserved verbatim; fixture n is honest",
        )
    )
    condensed = {
        "preregistration_sha256": report["preregistration_sha256"],
        "suite_sha256": report["suite"]["suite_sha256"],
        "corpus_sha256": report["corpus_sha256"],
        "certificate": certificate,
        "paired_evidence": report["paired_evidence"],
        "exposure": report["exposure"],
        "symbol_only": report["symbol_only"],
        "contamination_findings": contamination,
    }
    suite_rows = int(report["suite"]["rows_total"])
    n_arms = len(report["preregistration"]["arms"])
    output_chars = sum(
        arm_row["target_chars_total"]
        for arm_row in report["exposure"]["per_arm"].values()
    )
    workload = WorkloadIdentityV1(
        workload=(
            f"slm368 matched CAP1 fixture: {n_arms} arms x {len(seeds)} seed(s) x "
            f"{records_per_arm} records/arm x {steps} steps on the frozen "
            f"two-pack suite ({suite_rows} rows)"
        ),
        hardware=f"{platform.machine()} CPU, {platform.processor() or 'generic'}",
        python=platform.python_version(),
        batch_size=int(report["preregistration"]["batch_size"]),
        model_calls=suite_rows * n_arms * len(seeds),
        node_passes=steps * n_arms * len(seeds),
        compiler_verifier_work="adjudicate_suite + evaluate_gates per arm per seed",
        output_chars=output_chars,
        fallback_count=0,
        timeout_count=0,
    )
    return rows, condensed, workload


def run_operator_legal_set_check() -> list[EvidenceRowV1]:
    """DSH3 operator legal-set enumeration on a bounded exact fixture."""
    from dataclasses import replace

    from slm_training.dsl.operators import (
        ActionEffectV1,
        ApplicationProvenanceV1,
        AstOperatorV1,
        BindingPhase,
        CompilerCoverage,
        OperatorArgumentSlotV1,
        OperatorLibraryV1,
        OperatorMutationV1,
        OperatorRejectedError,
        OperatorStateV1,
        OperatorSupportVerdict,
        RefKind,
        ReferenceDescriptorV1,
        RegisteredOperatorV1,
        build_reference_table,
        enumerate_operator_legal_set,
        serialize_operator_action,
    )
    from slm_training.dsl.pack import get_pack

    base_pack = get_pack("openui")
    state = OperatorStateV1.from_source(
        base_pack, 'root = TextContent(":hero.title")'
    )
    descriptors = tuple(
        ReferenceDescriptorV1(
            ref_kind=RefKind.VALUE,
            semantic_fingerprint=content_sha(f"slm391-value-{i}"),
            value_type="openui.string",
        )
        for i in range(4)
    )
    table = build_reference_table(
        request_id="slm391-request",
        state_digest=state.state_digest,
        branch_digest=content_sha("slm391-branch"),
        descriptors=descriptors,
        seed=3,
    )
    allowed = {descriptors[1].semantic_fingerprint, descriptors[3].semantic_fingerprint}
    semantic_by_ref = {
        entry.ref: entry.descriptor.semantic_fingerprint for entry in table.entries
    }

    def execute(operator_state: Any, arguments: Any) -> OperatorMutationV1:
        if semantic_by_ref[arguments[0].value] not in allowed:
            raise OperatorRejectedError("fixture.not_legal")
        return OperatorMutationV1(
            source=operator_state.source,
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    declaration = AstOperatorV1(
        operator_id="slm391.replace_marker",
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1(
                "value", RefKind.VALUE, BindingPhase.APPLICATION
            ),
        ),
        preconditions=(),
        effect_signature=(),
        locality="node",
        cost=1.0,
    )
    library = OperatorLibraryV1((RegisteredOperatorV1(declaration, execute),))
    pack = replace(base_pack, operator_library=library)
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.fixture",
        compiler_version="v1",
        source_artifact_digest=state.state_digest,
        request_id="slm391-request",
    )
    legal_set = enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=state,
        reference_table=table,
        provenance=provenance,
        max_combinations_per_operator=16,
    )
    entry = legal_set.entries[0]
    serializable = all(
        serialize_operator_action(action.operator_id, action.arguments)
        == action.serialized
        for action in entry.legal_actions
    )
    return [
        EvidenceRowV1(
            row_id=_row_id("cap2", "legal-set"),
            claim_id=CLAIM_CAP2,
            suite="dsl.operators.legal_set",
            system="bounded_exact_fixture(n=4,allowed=2)",
            outcome_class=OUTCOME_POSITIVE,
            fixture_only=True,
            metrics={
                "verdict": entry.verdict.value,
                "coverage": entry.coverage.value,
                "legal_actions": len(entry.legal_actions),
                "expected_legal_actions": 2,
                "evaluated_combinations": entry.evaluated_combinations,
                "total_combinations": entry.total_combinations,
                "all_actions_serializable": serializable,
                "operator_fingerprint": entry.operator_fingerprint,
                "pack_id": pack.pack_id,
            },
            workload_identity=None,
            notes="exact enumeration matches the independent expectation (2 legal of 4)",
        ),
        EvidenceRowV1(
            row_id=_row_id("cap2", "unknown-budget"),
            claim_id=CLAIM_CAP2,
            suite="dsl.operators.legal_set",
            system="truncated_budget_fixture",
            outcome_class=OUTCOME_NEGATIVE,
            fixture_only=True,
            metrics={
                "budget_truncation_stays_unknown": (
                    OperatorSupportVerdict.UNKNOWN.value == "unknown"
                ),
            },
            workload_identity=None,
            notes="budget truncation must classify UNKNOWN, never unsupported",
        ),
    ]


def run_trace_eval_lifecycle(tmp_dir: Path) -> list[EvidenceRowV1]:
    """SLM-390 lifecycle: one clean trace end to end, one stop-rule trace."""
    from slm_training.harnesses.experiments.slm390_trace_evals import (
        ConfirmationLedger,
        LifecycleError,
        STATUS_QUARANTINED,
        freeze_candidate,
        ingest_trace,
        mark_reviewed,
        promote_to_training,
        record_confirmation_use,
        verify_frozen_eval,
    )

    clean_trace = {
        "trace_id": "slm391-trace-1",
        "operation": "openui.compile",
        "error_type": "marker_resolution",
        "message": "marker :hero.title unresolved at /old/path on host example.com",
        "task_prompt": "render the hero section",
        "expected_output": "ok",
        "observed_output": "marker_error",
        "attributes": {"tool.name": "openui-compiler", "tool.version": "v1"},
    }
    # The proposed eval root family must hash into the eval split; rotate the
    # operation (which feeds the cluster fingerprint) until it does.
    from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

    policy = RootFamilySplitPolicyV1()
    candidate = None
    for nonce in range(1000):
        trial = ingest_trace(
            {**clean_trace, "operation": f"openui.compile_{nonce}"},
            now="2026-07-25T00:00:00Z",
        )
        family = trial.proposed["task"]["root_family_id"]
        if policy.assign(str(family)) != "train":
            candidate = trial
            break
    if candidate is None:  # pragma: no cover - deterministic loop
        raise LifecycleError("no eval-split trace family available")
    rows: list[EvidenceRowV1] = []
    quarantined = candidate.status == STATUS_QUARANTINED and not candidate.rejection_reasons
    reviewed = mark_reviewed(
        candidate,
        {
            "reviewer_id": "slm391-fixture-reviewer",
            "checklist": {
                "environment_pinned": True,
                "verifier_independent": True,
                "no_exposed_answer": True,
                "no_shortcut": True,
                "privacy_ok": True,
                "agent_trajectory_reviewed": True,
                "verifier_trajectory_reviewed": True,
            },
        },
        now="2026-07-25T00:00:00Z",
    )
    frozen_candidate, frozen = freeze_candidate(
        reviewed,
        training_root_family_ids=(),
        output_dir=tmp_dir,
        now="2026-07-25T00:00:00Z",
    )
    integrity_ok = verify_frozen_eval(Path(frozen.frozen_path))
    ledger = ConfirmationLedger()
    confirmed = record_confirmation_use(frozen_candidate, frozen, ledger)
    one_touch_held = False
    try:
        record_confirmation_use(confirmed, frozen, ledger)
    except LifecycleError:
        one_touch_held = True
    promoted, activity = promote_to_training(
        confirmed, frozen, existing_training_family_ids=(), now="2026-07-25T00:00:00Z"
    )
    rows.append(
        EvidenceRowV1(
            row_id=_row_id("trace", "clean-lifecycle"),
            claim_id=CLAIM_TRACE,
            suite="slm390_trace_evals",
            system="clean_trace_lifecycle",
            outcome_class=OUTCOME_POSITIVE,
            fixture_only=True,
            metrics={
                "starts_quarantined": quarantined,
                "frozen_integrity_ok": integrity_ok,
                "manifest_sha256": frozen.manifest_sha256,
                "one_touch_confirmation_held": one_touch_held,
                "promotion_dataset_version": activity.dataset_version,
                "promotion_new_family": activity.training_root_family_id,
                "final_status": promoted.status,
            },
            workload_identity=None,
            notes="QUARANTINED->REVIEWED->FROZEN->CONFIRMATION_USED->PROMOTED explicit transitions only",
        )
    )
    unsafe_trace = {
        "trace_id": "slm391-trace-2",
        "operation": "openui.compile",
        "error_type": "io",
        "message": "read failed",
        "expected_output": "ok",
        "observed_output": "io_error",
        "attributes": {
            "tool.name": "openui-compiler",
            "tool.version": "v1",
        },
        # Secret-shaped content outside the redactable free-text fields: the
        # stop rule must keep this candidate quarantined (fail closed).
        "run_config": {"auth": "hf_abcdefghijklmno"},
    }
    unsafe = ingest_trace(unsafe_trace, now="2026-07-25T00:00:00Z")
    blocked = False
    try:
        mark_reviewed(unsafe, {"reviewer_id": "x", "checklist": {}})
    except LifecycleError:
        blocked = True
    rows.append(
        EvidenceRowV1(
            row_id=_row_id("trace", "stop-rule"),
            claim_id=CLAIM_TRACE,
            suite="slm390_trace_evals",
            system="undeidentifiable_trace",
            outcome_class=OUTCOME_INVALID,
            fixture_only=True,
            metrics={
                "status": unsafe.status,
                "rejection_reasons": list(unsafe.rejection_reasons),
                "review_blocked": blocked,
            },
            workload_identity=None,
            notes="secret outside redactable fields: stop rule keeps it quarantined",
        )
    )
    return rows


def run_strict_v2_check() -> list[EvidenceRowV1]:
    """Strict-v2 / binding-aware metric over the frozen adversarial corpus."""
    from slm_training.dsl.schema import ExampleRecord
    from slm_training.evals.meaningful_program import (
        aggregate_meaning_reports_v2,
        binding_aware_meaningful_v2,
    )

    corpus = (
        Path(__file__).resolve().parents[2]
        / "resources"
        / "evals"
        / "meaningful_v2_gaming.jsonl"
    )
    if not corpus.is_file():
        return [
            EvidenceRowV1(
                row_id=_row_id("cap1", "strict-v2-missing"),
                claim_id=CLAIM_CAP1,
                suite="meaningful_v2_gaming",
                system="binding_aware_meaningful_v2",
                outcome_class=OUTCOME_INVALID,
                fixture_only=True,
                metrics={"corpus": str(corpus)},
                workload_identity=None,
                notes="frozen adversarial corpus absent on this lineage",
            )
        ]
    reports = []
    expected = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        record = ExampleRecord(
            id=str(case["id"]),
            prompt=str(case["prompt"]),
            openui=str(case["prediction"]),
            split="adversarial",
            source="deterministic",
        )
        reports.append(binding_aware_meaningful_v2(case["prediction"], record=record))
        expected.append(bool(case["expected_verdict"]))
    aggregate = aggregate_meaning_reports_v2(reports)
    actual = [bool(report.verdict) for report in reports]
    agreement = sum(int(a == e) for a, e in zip(actual, expected)) / max(len(expected), 1)
    return [
        EvidenceRowV1(
            row_id=_row_id("cap1", "strict-v2"),
            claim_id=CLAIM_CAP1,
            suite="meaningful_v2_gaming",
            system="binding_aware_meaningful_v2",
            outcome_class=OUTCOME_POSITIVE if agreement == 1.0 else OUTCOME_NEGATIVE,
            fixture_only=True,
            metrics={
                "metric_name": aggregate["metric_name"],
                "metric_version": aggregate["metric_version"],
                "metric_implementation_hash": aggregate["metric_implementation_hash"],
                "n": aggregate["n"],
                "strict_rate": aggregate["strict_rate"],
                "expected_verdict_agreement": agreement,
            },
            workload_identity=None,
            notes="metric-only evidence; never a model capability claim",
        )
    ]


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimDispositionV1:
    claim_id: str
    capability: str
    scope: str
    verdict: str
    supported_bounds: str
    rationale: str
    identities: Mapping[str, Any]
    artifacts: tuple[ArtifactBindingV1, ...]
    evidence_rows: tuple[EvidenceRowV1, ...]

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise DispositionError(f"undeclared verdict {self.verdict!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "capability": self.capability,
            "scope": self.scope,
            "verdict": self.verdict,
            "supported_bounds": self.supported_bounds,
            "rationale": self.rationale,
            "identities": dict(self.identities),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "evidence_rows": [row.to_dict() for row in self.evidence_rows],
        }


def guard_supported_requires_identity(claim: ClaimDispositionV1) -> bool:
    """A supported claim must name code, suite, config, and hardware identities."""
    if claim.verdict != VERDICT_SUPPORTED:
        return True
    identities = claim.identities
    return all(identities.get(key) for key in ("code", "suite", "config", "hardware"))


def guard_no_fixture_only_capability_support(claim: ClaimDispositionV1) -> bool:
    """Fixture-only rows never support a model-capability or ship claim."""
    if claim.scope not in FIXTURE_FORBIDDEN_SCOPES:
        return True
    if claim.verdict != VERDICT_SUPPORTED:
        return True
    return any(
        row.outcome_class == OUTCOME_POSITIVE and not row.fixture_only
        for row in claim.evidence_rows
    )


def guard_unknown_not_normalized(claim: ClaimDispositionV1) -> bool:
    """Unknown claims carry no measured bounds normalized by assumption."""
    if claim.verdict != VERDICT_UNKNOWN:
        return True
    return not claim.supported_bounds


def guard_efficiency_requires_workload_identity(claim: ClaimDispositionV1) -> bool:
    """An efficiency claim is supported only with a complete workload identity."""
    if claim.claim_id != CLAIM_EFFICIENCY or claim.verdict != VERDICT_SUPPORTED:
        return True
    for row in claim.evidence_rows:
        if row.workload_identity:
            return True
    return False


GUARDS = (
    guard_supported_requires_identity,
    guard_no_fixture_only_capability_support,
    guard_unknown_not_normalized,
    guard_efficiency_requires_workload_identity,
)


def enforce_guards(claims: Sequence[ClaimDispositionV1]) -> None:
    for claim in claims:
        for guard in GUARDS:
            if not guard(claim):
                raise DispositionError(
                    f"guard {guard.__name__} failed for claim {claim.claim_id}"
                )


# --------------------------------------------------------------------------
# Disposition assembly
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StagedDslHarnessDispositionV1:
    schema: str
    schema_version: str
    experiment_id: str
    generated_at: str
    evidence_cutoff_commit: str
    claims: tuple[ClaimDispositionV1, ...]
    follow_ups: tuple[Mapping[str, str], ...]
    version_stamp: Mapping[str, Any]
    agentv_diagnostic: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "generated_at": self.generated_at,
            "evidence_cutoff_commit": self.evidence_cutoff_commit,
            "claims": [claim.to_dict() for claim in self.claims],
            "follow_ups": [dict(f) for f in self.follow_ups],
            "version_stamp": dict(self.version_stamp),
            "agentv_diagnostic": dict(self.agentv_diagnostic),
        }


def _build_version_stamp() -> dict[str, Any]:
    try:
        return build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm391_portfolio_disposition",
        )
    except KeyError:
        base = build_version_stamp("harness.experiments")
        base["components"]["harness.experiments.slm391_portfolio_disposition"] = UNKNOWN
        return base


def _hardware_identity() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "device": "cpu",
    }


def build_disposition(
    *,
    repo_root: Path,
    run_suite_checks: bool = True,
    cap1_steps: int = DEFAULT_CAP1_STEPS,
    cap1_records_per_arm: int = DEFAULT_CAP1_RECORDS_PER_ARM,
    cap1_seeds: Sequence[int] = DEFAULT_CAP1_SEEDS,
    now: str,
    trace_tmp_dir: Path | None = None,
) -> StagedDslHarnessDispositionV1:
    """Run the fixture-scale suites and assemble the final disposition."""
    registry = claim_registry()
    artifacts = {spec.claim_id: bind_artifacts(spec, repo_root) for spec in registry}

    rows_by_claim: dict[str, list[EvidenceRowV1]] = {cid: [] for cid in CLAIM_IDS}
    cap1_condensed: dict[str, Any] = {}
    cap1_workload: WorkloadIdentityV1 | None = None
    if run_suite_checks:
        rows_by_claim[CLAIM_CAP1].extend(run_two_pack_gate_checks())
        cap1_rows, cap1_condensed, cap1_workload = run_cap1_matched_experiment(
            steps=cap1_steps,
            records_per_arm=cap1_records_per_arm,
            seeds=cap1_seeds,
        )
        rows_by_claim[CLAIM_CAP1].extend(cap1_rows)
        rows_by_claim[CLAIM_CAP1].extend(run_strict_v2_check())
        rows_by_claim[CLAIM_CAP2].extend(run_operator_legal_set_check())
        if trace_tmp_dir is None:
            with tempfile.TemporaryDirectory(prefix="slm391_trace_") as tmp:
                rows_by_claim[CLAIM_TRACE].extend(run_trace_eval_lifecycle(Path(tmp)))
        else:
            rows_by_claim[CLAIM_TRACE].extend(run_trace_eval_lifecycle(trace_tmp_dir))

    code_commit = git_commit()
    claims: list[ClaimDispositionV1] = []
    for spec in registry:
        bound = artifacts[spec.claim_id]
        rows = tuple(rows_by_claim[spec.claim_id])
        missing_required = [b.path for b in bound if b.required and not b.present]
        identities: dict[str, Any] = {
            "code": code_commit,
            "suite": "",
            "config": "",
            "hardware": _hardware_identity(),
        }
        if spec.claim_id == CLAIM_CAP1 and cap1_condensed:
            identities["suite"] = str(cap1_condensed.get("suite_sha256", ""))
            identities["config"] = str(cap1_condensed.get("preregistration_sha256", ""))
        if spec.claim_id == CLAIM_TRACE:
            clean = next(
                (r for r in rows if r.system == "clean_trace_lifecycle"), None
            )
            if clean is not None:
                identities["suite"] = str(clean.metrics.get("manifest_sha256", ""))
                identities["config"] = "slm390_trace_evals lifecycle fixture"
        if spec.claim_id == CLAIM_CAP2:
            legal = next((r for r in rows if r.suite == "dsl.operators.legal_set"), None)
            if legal is not None:
                identities["suite"] = "dsl.operators.legal_set bounded fixture"
                identities["config"] = str(legal.metrics.get("operator_fingerprint", ""))

        verdict: str
        supported_bounds = ""
        if missing_required:
            # Stop rule: incomparable lineage — unknown, never normalized.
            verdict = VERDICT_UNKNOWN
            rationale = (
                f"required artifacts absent on this lineage: "
                f"{', '.join(missing_required)}; classified unknown per the stop "
                f"rule — never normalized through unsupported assumptions"
            )
        elif spec.claim_id == CLAIM_CAP1:
            certificate = cap1_condensed.get("certificate", {})
            if certificate.get("status") == "issued":
                # Still fixture scale: a fixture certificate is not a held-out
                # semantic or ship capability.
                verdict = VERDICT_SUPPORTED
                supported_bounds = ""
                rationale = "unexpected: fixture certificate issued"
            else:
                verdict = VERDICT_REJECTED
                rationale = (
                    "CERT_CAP1 rejected with stop codes "
                    f"{certificate.get('rejection_codes', [])}; the strongest "
                    "retained configuration does not clear the preregistered "
                    "paired-evidence bar at fixture scale, so held-out semantic "
                    "and ship capability stay closed"
                )
        elif spec.claim_id == CLAIM_CAP2:
            legal = next((r for r in rows if r.suite == "dsl.operators.legal_set"), None)
            ok = bool(
                legal
                and legal.metrics.get("verdict") == "supported"
                and legal.metrics.get("legal_actions")
                == legal.metrics.get("expected_legal_actions")
                and legal.metrics.get("all_actions_serializable")
            )
            verdict = VERDICT_SUPPORTED if ok else VERDICT_REJECTED
            if ok:
                supported_bounds = (
                    "fixture-scale harness-contract evidence only: exact bounded "
                    "legal-set enumeration over a 4-candidate fixture with "
                    "reserved serialization; not a model capability, NL "
                    "transform, or ship claim"
                )
            rationale = (
                "operator legal-set enumeration reproduces the independent "
                "expectation exactly at fixture scale"
                if ok
                else "operator legal-set fixture did not reproduce its expectation"
            )
        elif spec.claim_id == CLAIM_TRACE:
            clean = next((r for r in rows if r.system == "clean_trace_lifecycle"), None)
            stop = next((r for r in rows if r.system == "undeidentifiable_trace"), None)
            ok = bool(
                clean
                and clean.metrics.get("frozen_integrity_ok")
                and clean.metrics.get("one_touch_confirmation_held")
                and stop
                and stop.metrics.get("status") == "QUARANTINED"
                and stop.metrics.get("rejection_reasons")
                and stop.metrics.get("review_blocked")
            )
            verdict = VERDICT_SUPPORTED if ok else VERDICT_REJECTED
            if ok:
                supported_bounds = (
                    "fixture-scale harness-contract evidence only: the lifecycle "
                    "state machine, quarantine stop rules, freeze integrity, and "
                    "one-touch confirmation all verified on synthetic traces; no "
                    "runtime contamination claim"
                )
            rationale = (
                "trace-eval lifecycle verified end to end with the stop-rule "
                "trace held in quarantine"
                if ok
                else "trace-eval lifecycle fixture check failed"
            )
        elif spec.claim_id == CLAIM_EFFICIENCY:
            # No comparative baseline latency/work measurement ran on this
            # lineage; normalization identity is recorded, claim stays unknown.
            verdict = VERDICT_UNKNOWN
            if cap1_workload is not None:
                rows = rows + (
                    EvidenceRowV1(
                        row_id=_row_id("efficiency", "workload-identity"),
                        claim_id=CLAIM_EFFICIENCY,
                        suite="cap1_two_pack_freeze/v1",
                        system="slm368_fixture_workload",
                        outcome_class=OUTCOME_NO_EFFECT,
                        fixture_only=True,
                        metrics={"comparative_baseline": None},
                        workload_identity=cap1_workload.to_dict(),
                        notes=(
                            "workload/hardware/batch/call identity recorded; no "
                            "matched-baseline measurement exists, so no efficiency "
                            "effect is claimed"
                        ),
                    ),
                )
            rationale = (
                "no normalized comparative measurement (matched baseline, "
                "hardware, batch, model calls, node passes, compiler/verifier "
                "work, output length, fallback/timeout) exists on this lineage; "
                "classified unknown per the stop rule"
            )
        else:
            verdict = VERDICT_UNKNOWN
            rationale = "unclassifiable"

        claims.append(
            ClaimDispositionV1(
                claim_id=spec.claim_id,
                capability=spec.capability,
                scope=spec.scope,
                verdict=verdict,
                supported_bounds=supported_bounds,
                rationale=rationale,
                identities=identities,
                artifacts=bound,
                evidence_rows=rows,
            )
        )

    enforce_guards(claims)
    follow_ups = recommend_follow_ups(claims)
    return StagedDslHarnessDispositionV1(
        schema=DISPOSITION_SCHEMA,
        schema_version=SCHEMA_VERSION,
        experiment_id=EXPERIMENT_ID,
        generated_at=now,
        evidence_cutoff_commit=code_commit,
        claims=tuple(claims),
        follow_ups=follow_ups,
        version_stamp=_build_version_stamp(),
        agentv_diagnostic={},
    )


def recommend_follow_ups(
    claims: Sequence[ClaimDispositionV1],
) -> tuple[Mapping[str, str], ...]:
    """Deduplicated follow-ups grounded in positive or unresolved evidence."""
    by_id = {claim.claim_id: claim for claim in claims}
    proposals: list[dict[str, str]] = []
    if by_id[CLAIM_CAP1].verdict in (VERDICT_REJECTED, VERDICT_UNKNOWN):
        proposals.append(
            {
                "id": "cap1-scale-paired-n",
                "title": "Scale CERT_CAP1 paired evidence beyond fixture n",
                "grounding": (
                    f"cap1_semantics is {by_id[CLAIM_CAP1].verdict}: the frozen "
                    "suite harness is intact and the matched experiment runs; "
                    "only the preregistered paired-n bar is unmet at fixture scale"
                ),
            }
        )
    if by_id[CLAIM_CAP0].verdict == VERDICT_UNKNOWN:
        proposals.append(
            {
                "id": "cap0-port-frozen-suite",
                "title": "Port the SLM-361/SLM-362 frozen CAP0 suite onto this lineage",
                "grounding": (
                    "cap0_grammar is unknown: the CAP0 freeze and matched "
                    "experiment harness are absent on this lineage (parallel "
                    "unmerged branches); classification cannot proceed without them"
                ),
            }
        )
    if by_id[CLAIM_DISTILLATION].verdict == VERDICT_UNKNOWN:
        proposals.append(
            {
                "id": "dsh4-distillation-evidence",
                "title": "Land DSH4-01/DSH4-02 distillation artifacts, then reclassify",
                "grounding": (
                    "distillation is unknown: the DSH4-01/02 evidence artifacts "
                    "(in review as PRs #903/#905) are not present on this lineage"
                ),
            }
        )
    if by_id[CLAIM_EFFICIENCY].verdict == VERDICT_UNKNOWN:
        proposals.append(
            {
                "id": "efficiency-normalized-baseline",
                "title": "Run a normalized matched-baseline efficiency measurement",
                "grounding": (
                    "efficiency is unknown: workload identity is recorded but no "
                    "matched-baseline latency/work comparison exists on this lineage"
                ),
            }
        )
    deduped: dict[str, dict[str, str]] = {}
    for proposal in proposals:
        deduped.setdefault(proposal["id"], proposal)
    return tuple(deduped.values())


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def render_markdown(disposition: StagedDslHarnessDispositionV1, *, command: str) -> str:
    lines = [
        "# SLM-391 (DSH4-06): staged DSL harness portfolio disposition",
        "",
        "**Claim class:** fixture (disposition audit at fixture scale; no ship claim)",
        "",
        f"**Schema:** `{disposition.schema}`",
        "",
        f"**Evidence cutoff commit:** `{disposition.evidence_cutoff_commit}`",
        "",
        f"**Generated at:** {disposition.generated_at}",
        "",
        "Every claim names its compatible code/suite/config/hardware identities "
        "or is `unknown`. Missing or incomparable artifacts are classified "
        "`unknown` per the stop rule and are never normalized through "
        "unsupported assumptions. Negative, invalid, timeout, no-effect, "
        "contaminated, and prediction-identical outcomes remain first-class "
        "rows. Efficiency and semantic claims are separate; no fixture-only "
        "row supports a capability or ship claim.",
        "",
        "## Claim dispositions",
        "",
        "| Claim | Scope | Verdict | Supported bounds |",
        "| --- | --- | --- | --- |",
    ]
    for claim in disposition.claims:
        bounds = claim.supported_bounds.replace("|", "\\|") or "—"
        lines.append(
            f"| {claim.claim_id} | {claim.scope} | **{claim.verdict}** | {bounds} |"
        )
    for claim in disposition.claims:
        lines += [
            "",
            f"### {claim.claim_id} — {claim.capability}",
            "",
            f"- Verdict: **{claim.verdict}** ({claim.scope})",
            f"- Rationale: {claim.rationale}",
            f"- Identities: code `{claim.identities.get('code', '')}`; suite "
            f"`{claim.identities.get('suite', '') or 'n/a'}`; config "
            f"`{claim.identities.get('config', '') or 'n/a'}`; hardware "
            f"`{claim.identities.get('hardware', {})}`",
            "",
            "| Suite | System | Outcome | Notes |",
            "| --- | --- | --- | --- |",
        ]
        if claim.evidence_rows:
            for row in claim.evidence_rows:
                notes = row.notes.replace("|", "\\|")
                lines.append(
                    f"| {row.suite} | {row.system} | {row.outcome_class} | {notes} |"
                )
        else:
            lines.append("| — | — | — | no suite rows ran on this lineage |")
        missing = [a.path for a in claim.artifacts if a.required and not a.present]
        if missing:
            lines += ["", "Missing required artifacts (stop rule):"]
            lines += [f"- `{path}`" for path in missing]
    lines += [
        "",
        "## Follow-up recommendations (deduplicated, evidence-grounded)",
        "",
    ]
    if disposition.follow_ups:
        for follow in disposition.follow_ups:
            lines.append(f"- **{follow['title']}** (`{follow['id']}`): {follow['grounding']}")
    else:
        lines.append("- _None._")
    lines += [
        "",
        "## Exact command",
        "",
        f"```bash\n{command}\n```",
        "",
        "## Limitations",
        "",
        "- Fixture scale only; a fixture certificate or gate pass is wiring "
        "evidence, never a held-out semantic or ship capability.",
        "- CAP0 (SLM-361/362) and DSH4-01/02 distillation artifacts are absent "
        "on this lineage (parallel unmerged branches / in-review PRs); those "
        "claims are `unknown`, not negative.",
        "- No promotion, checkpoint, model-card, or default change results.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "CLAIM_CAP0",
    "CLAIM_CAP1",
    "CLAIM_CAP2",
    "CLAIM_DISTILLATION",
    "CLAIM_EFFICIENCY",
    "CLAIM_IDS",
    "CLAIM_TRACE",
    "DISPOSITION_SCHEMA",
    "DispositionError",
    "EXPERIMENT_ID",
    "FIXTURE_FORBIDDEN_SCOPES",
    "GUARDS",
    "MATRIX_SET",
    "OUTCOME_CLASSES",
    "OUTCOME_CONTAMINATED",
    "OUTCOME_INVALID",
    "OUTCOME_NEGATIVE",
    "OUTCOME_NO_EFFECT",
    "OUTCOME_POSITIVE",
    "OUTCOME_PREDICTION_IDENTICAL",
    "OUTCOME_TIMEOUT",
    "SCHEMA_VERSION",
    "SCOPE_HARNESS_CONTRACT",
    "SCOPE_MODEL_CAPABILITY",
    "SCOPE_SHIP",
    "VERDICT_INVALID",
    "VERDICT_REJECTED",
    "VERDICT_SUPPORTED",
    "VERDICT_UNKNOWN",
    "VERDICTS",
    "ArtifactBindingV1",
    "ClaimDispositionV1",
    "ClaimSpecV1",
    "EvidenceRowV1",
    "StagedDslHarnessDispositionV1",
    "WorkloadIdentityV1",
    "bind_artifacts",
    "build_disposition",
    "claim_registry",
    "enforce_guards",
    "guard_efficiency_requires_workload_identity",
    "guard_no_fixture_only_capability_support",
    "guard_supported_requires_identity",
    "guard_unknown_not_normalized",
    "recommend_follow_ups",
    "render_markdown",
    "run_cap1_matched_experiment",
    "run_operator_legal_set_check",
    "run_strict_v2_check",
    "run_trace_eval_lifecycle",
    "run_two_pack_gate_checks",
]
