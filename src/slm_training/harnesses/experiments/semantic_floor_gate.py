"""SLM-213: authoritative semantic-floor claim authorization.

The gate aggregates existing SDE5 evidence. It does not run an evaluation,
change a metric, or promote a checkpoint.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from slm_training.evals.meaningful_program import METRIC_NAME, METRIC_VERSION
from slm_training.harness_core.lineage.records import content_sha
from slm_training.versioning import build_version_stamp, git_commit

SCHEMA_VERSION = "semantic_floor_gate/v1"
GATE_COMPONENT = "harness.experiments.semantic_floor_gate"
DEFAULT_GATE_PATH = "docs/design/semantic-floor-gate-v1.json"

Verdict = Literal["floor_escaped", "proxy_only", "rejected", "inconclusive"]

SEMANTIC_CLAIM_CLASSES = frozenset(
    {"semantic_prediction", "semantic_causal", "learned_latent"}
)
PROXY_CLAIM_CLASSES = frozenset(
    {"diagnostic", "proxy", "constraint_debt", "structure", "wiring", "synthetic_wiring"}
)

EVIDENCE_PATHS: tuple[tuple[str, str, str], ...] = (
    (
        "SLM-208",
        "docs/design/iter-slm208-constraint-debt-20260720.json",
        "Slm208ConstraintDebtReportV1",
    ),
    (
        "SLM-209",
        "docs/design/iter-slm209-debt-targeted-curriculum-20260720.json",
        "DebtCurriculumManifestV1",
    ),
    (
        "SLM-210",
        "docs/design/sde5-floor-escape-matrix-results.json",
        "SDE5FloorEscapeMatrixV1",
    ),
    (
        "SLM-211",
        "docs/design/iter-slm211-untied-output-head-20260721.json",
        "UntiedOutputHeadReportV1",
    ),
    (
        "SLM-212",
        "docs/design/iter-slm212-debt-routing-20260721.json",
        "DebtRoutingMatrixManifest",
    ),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _literal_assignment(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise ValueError(f"{path}: string constant {name} is unresolved")


@dataclass(frozen=True)
class EvidenceReferenceV1:
    issue_id: str
    path: str
    sha256: str
    schema: str
    status: str
    claim_class: str
    disposition: str | None
    source_commit: str
    source_dirty: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceReferenceV1":
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown evidence-reference fields: {sorted(unknown)}")
        return cls(
            issue_id=str(data["issue_id"]),
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            schema=str(data["schema"]),
            status=str(data["status"]),
            claim_class=str(data["claim_class"]),
            disposition=None if data.get("disposition") is None else str(data["disposition"]),
            source_commit=str(data["source_commit"]),
            source_dirty=data.get("source_dirty"),
        )


@dataclass(frozen=True)
class SemanticFloorGateV1:
    source_commit: str
    generated_at: str
    evidence_cutoff: str
    metric_versions: Mapping[str, str]
    evaluator_versions: Mapping[str, str]
    checkpoint_references: tuple[Mapping[str, Any], ...]
    config_hashes: tuple[str, ...]
    train_manifests: tuple[Mapping[str, Any], ...]
    eval_manifests: tuple[Mapping[str, Any], ...]
    sample_sizes: Mapping[str, int]
    anti_gaming: Mapping[str, Any]
    seeds: tuple[int, ...]
    paired_statistics: Mapping[str, Any]
    strict_meaning_v2_by_suite: Mapping[str, Any]
    agentv_evaluation: Mapping[str, Any]
    legacy_meaning_v1_by_suite: Mapping[str, Any]
    constraint_debt: Mapping[str, Any]
    protected_objectives: Mapping[str, Any]
    semantic_checks: Mapping[str, Any]
    runtime_cost: Mapping[str, Any]
    verdict: Verdict
    allowed_claims: tuple[str, ...]
    blocked_claims: tuple[str, ...]
    resolving_evidence: tuple[str, ...]
    evidence: tuple[EvidenceReferenceV1, ...]
    version_stamp: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def _hash_payload(self) -> dict[str, Any]:
        payload = self.to_dict(include_hash=False)
        payload.pop("generated_at")
        payload.pop("source_commit")
        stamp = dict(payload.pop("version_stamp", {}))
        stamp.pop("stamped_at", None)
        stamp.pop("code_commit", None)
        stamp.pop("code_dirty", None)
        payload["component_versions"] = stamp.get("components", {})
        return payload

    @property
    def gate_hash(self) -> str:
        return content_sha(self._hash_payload())

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["metric_versions"] = dict(self.metric_versions)
        payload["evaluator_versions"] = dict(self.evaluator_versions)
        payload["checkpoint_references"] = [dict(row) for row in self.checkpoint_references]
        payload["train_manifests"] = [dict(row) for row in self.train_manifests]
        payload["eval_manifests"] = [dict(row) for row in self.eval_manifests]
        payload["evidence"] = [row.to_dict() for row in self.evidence]
        if include_hash:
            payload["gate_hash"] = self.gate_hash
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticFloorGateV1":
        unknown = set(data) - set(cls.__dataclass_fields__) - {"gate_hash"}
        if unknown:
            raise ValueError(f"unknown semantic-floor gate fields: {sorted(unknown)}")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported semantic-floor schema: {data.get('schema_version')}")
        verdict = str(data["verdict"])
        if verdict not in {"floor_escaped", "proxy_only", "rejected", "inconclusive"}:
            raise ValueError(f"unknown semantic-floor verdict: {verdict}")
        gate = cls(
            source_commit=str(data["source_commit"]),
            generated_at=str(data["generated_at"]),
            evidence_cutoff=str(data["evidence_cutoff"]),
            metric_versions=dict(data.get("metric_versions", {})),
            evaluator_versions=dict(data.get("evaluator_versions", {})),
            checkpoint_references=tuple(dict(row) for row in data.get("checkpoint_references", ())),
            config_hashes=tuple(str(row) for row in data.get("config_hashes", ())),
            train_manifests=tuple(dict(row) for row in data.get("train_manifests", ())),
            eval_manifests=tuple(dict(row) for row in data.get("eval_manifests", ())),
            sample_sizes={str(k): int(v) for k, v in data.get("sample_sizes", {}).items()},
            anti_gaming=dict(data.get("anti_gaming", {})),
            seeds=tuple(int(seed) for seed in data.get("seeds", ())),
            paired_statistics=dict(data.get("paired_statistics", {})),
            strict_meaning_v2_by_suite=dict(data.get("strict_meaning_v2_by_suite", {})),
            agentv_evaluation=dict(data.get("agentv_evaluation", {})),
            legacy_meaning_v1_by_suite=dict(data.get("legacy_meaning_v1_by_suite", {})),
            constraint_debt=dict(data.get("constraint_debt", {})),
            protected_objectives=dict(data.get("protected_objectives", {})),
            semantic_checks=dict(data.get("semantic_checks", {})),
            runtime_cost=dict(data.get("runtime_cost", {})),
            verdict=verdict,  # type: ignore[arg-type]
            allowed_claims=tuple(str(row) for row in data.get("allowed_claims", ())),
            blocked_claims=tuple(str(row) for row in data.get("blocked_claims", ())),
            resolving_evidence=tuple(str(row) for row in data.get("resolving_evidence", ())),
            evidence=tuple(EvidenceReferenceV1.from_dict(row) for row in data.get("evidence", ())),
            version_stamp=dict(data.get("version_stamp", {})),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        )
        expected = data.get("gate_hash")
        if expected is not None and expected != gate.gate_hash:
            raise ValueError("semantic-floor gate hash does not match canonical content")
        return gate


def decide_verdict(
    *,
    strict_meaning_v2: float | None,
    eval_n: int,
    paired_reproducible: bool,
    anti_gaming_passed: bool,
    identities_resolved: bool,
    agentv_contradiction: bool,
    proxy_moved: bool,
    stable_failure: bool = False,
    gaming_explains_gain: bool = False,
) -> Verdict:
    """Apply the SLM-213 decision rules without weakening missing evidence."""
    if not identities_resolved or strict_meaning_v2 is None or eval_n < 8:
        return "inconclusive"
    if gaming_explains_gain or stable_failure:
        return "rejected"
    if (
        strict_meaning_v2 > 0
        and paired_reproducible
        and anti_gaming_passed
        and not agentv_contradiction
    ):
        return "floor_escaped"
    if proxy_moved and strict_meaning_v2 <= 0:
        return "proxy_only"
    return "inconclusive"


def _load_evidence(repo_root: Path) -> tuple[tuple[EvidenceReferenceV1, ...], dict[str, Any]]:
    refs: list[EvidenceReferenceV1] = []
    payloads: dict[str, Any] = {}
    for issue_id, relative, expected_schema in EVIDENCE_PATHS:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"unsafe evidence path: {relative}")
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required semantic-floor evidence is missing: {relative}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{relative}: invalid JSON") from exc
        if payload.get("schema") != expected_schema:
            raise ValueError(
                f"{relative}: expected schema {expected_schema}, got {payload.get('schema')}"
            )
        stamp = payload.get("version_stamp", {})
        commit = str(stamp.get("code_commit", ""))
        if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
            raise ValueError(f"{relative}: unresolved source commit")
        if not payload.get("status") or not payload.get("claim_class"):
            raise ValueError(f"{relative}: status and claim_class are required")
        refs.append(
            EvidenceReferenceV1(
                issue_id=issue_id,
                path=relative,
                sha256=_sha256(path),
                schema=str(payload.get("schema", "")),
                status=str(payload.get("status", "")),
                claim_class=str(payload.get("claim_class", "")),
                disposition=payload.get("disposition"),
                source_commit=commit,
                source_dirty=stamp.get("code_dirty"),
            )
        )
        payloads[issue_id] = payload
    return tuple(refs), payloads


def build_semantic_floor_gate(
    *,
    repo_root: Path | None = None,
    source_commit: str | None = None,
    generated_at: str | None = None,
) -> SemanticFloorGateV1:
    """Load the registered SDE5 evidence bundle and derive one fail-closed gate."""
    root = repo_root or _repo_root()
    evidence, payloads = _load_evidence(root)
    floor = payloads["SLM-210"]
    versions = json.loads(
        (root / "src/slm_training/resources/versions.json").read_text(encoding="utf-8")
    )["components"]
    anti_gaming_path = root / "src/slm_training/evals/metric_gaming.py"
    package = json.loads((root / "package.json").read_text(encoding="utf-8"))
    package_lock = json.loads((root / "package-lock.json").read_text(encoding="utf-8"))
    agentv_version = str(package["devDependencies"]["@agentv/core"])
    locked_agentv_version = str(
        package_lock["packages"]["node_modules/@agentv/core"]["version"]
    )
    if locked_agentv_version != agentv_version:
        raise ValueError("AgentV package and lockfile versions disagree")
    anti_gaming_schema = _literal_assignment(anti_gaming_path, "SCHEMA_VERSION")

    seeds = tuple(sorted({int(cell["selection_cell"]["seed"]) for cell in floor["cells"]}))
    anti_gaming_scheduled = sum(bool(cell.get("runs_anti_gaming_suite")) for cell in floor["cells"])
    cutoff = max(
        str(payloads[issue].get("timestamp") or payloads[issue].get("generated_at") or "")
        for issue, _, _ in EVIDENCE_PATHS
    )
    identity_failures = [
        "no durable checkpoint reference for the SDE5 floor-escape family",
        "no hash-pinned train manifest",
        "no hash-pinned eval manifest or preregistered strict meaning-v2 sample count",
        "no executed anti-gaming result bundle (schedule flags are not outcomes)",
        "no SDE5 AgentV/independent-evaluation bundle",
        "no paired strict meaning-v2 statistics across the declared seeds",
    ]
    if any(ref.source_dirty is not False for ref in evidence):
        identity_failures.append("all SLM-208–212 producer stamps report code_dirty=true")

    verdict = decide_verdict(
        strict_meaning_v2=None,
        eval_n=0,
        paired_reproducible=False,
        anti_gaming_passed=False,
        identities_resolved=False,
        agentv_contradiction=False,
        proxy_moved=True,
    )
    return SemanticFloorGateV1(
        source_commit=source_commit or git_commit(),
        generated_at=generated_at or _now(),
        evidence_cutoff=cutoff,
        metric_versions={
            METRIC_NAME: METRIC_VERSION,
            "registry:evals.meaningful_program": str(
                versions["evals.meaningful_program"]["version"]
            ),
        },
        evaluator_versions={
            "evals.scoring": str(versions["evals.scoring"]["version"]),
            "@agentv/core": agentv_version,
            "agentv_owner_sha256": _sha256(root / "src/slm_training/evals/agentv.py"),
        },
        checkpoint_references=(),
        config_hashes=(),
        train_manifests=(),
        eval_manifests=(),
        sample_sizes={"strict_meaning_v2": 0, "agentv": 0},
        anti_gaming={
            "schema_version": anti_gaming_schema,
            "owner_sha256": _sha256(anti_gaming_path),
            "scheduled_cells": anti_gaming_scheduled,
            "executed": False,
            "passed": False,
            "status": "scheduled_not_executed",
        },
        seeds=seeds,
        paired_statistics={
            "available": False,
            "reason": "no paired strict meaning-v2 statistics across the declared seeds",
        },
        strict_meaning_v2_by_suite={},
        agentv_evaluation={
            "sdk": "@agentv/core",
            "sdk_version": agentv_version,
            "resolved": False,
            "n": 0,
            "status": "missing",
        },
        legacy_meaning_v1_by_suite={},
        constraint_debt={
            "slm208_summary": payloads["SLM-208"].get("summary", {}),
            "slm209_disposition": payloads["SLM-209"].get("disposition"),
            "slm212_disposition": payloads["SLM-212"].get("disposition"),
            "evidence_class": "synthetic_fixture",
        },
        protected_objectives={
            "measured": False,
            "reason": "no trained matched campaign or protected-objective outcome artifact",
        },
        semantic_checks={
            "inventory": "scheduled_not_executed",
            "binding": "not_measured",
            "repeated_subtree": "scheduled_not_executed",
            "hidden_gold": "not_measured",
            "minimal_valid": "scheduled_not_executed",
            "retry": "scheduled_not_executed",
        },
        runtime_cost={
            "evidence_class": "fixture_only",
            "model_runtime_measured": False,
            "real_decode_cost_measured": False,
        },
        verdict=verdict,
        allowed_claims=(
            "diagnostic",
            "proxy",
            "constraint_debt",
            "structure",
            "synthetic_wiring",
        ),
        blocked_claims=(
            "semantic_prediction",
            "semantic_causal",
            "learned_latent",
            "floor_escape",
            "promotion",
            "ship",
        ),
        resolving_evidence=tuple(identity_failures),
        evidence=evidence,
        version_stamp=build_version_stamp("harness.experiments", GATE_COMPONENT),
    )


def load_semantic_floor_gate(
    gate_or_path: SemanticFloorGateV1 | str | Path,
) -> SemanticFloorGateV1:
    if isinstance(gate_or_path, SemanticFloorGateV1):
        return gate_or_path
    payload = json.loads(Path(gate_or_path).read_text(encoding="utf-8"))
    return SemanticFloorGateV1.from_dict(payload)


def require_floor_gate(
    gate_or_path: SemanticFloorGateV1 | str | Path,
    claim_class: str,
) -> SemanticFloorGateV1:
    """Authorize explicit proxy work or require a real floor escape."""
    gate = load_semantic_floor_gate(gate_or_path)
    if claim_class in PROXY_CLAIM_CLASSES:
        return gate
    if claim_class not in SEMANTIC_CLAIM_CLASSES:
        raise ValueError(f"unknown semantic-floor claim class: {claim_class}")
    if gate.verdict != "floor_escaped":
        raise PermissionError(
            f"{claim_class} is blocked by SemanticFloorGateV1 "
            f"{gate.gate_hash}: verdict={gate.verdict}"
        )
    return gate


def render_markdown(gate: SemanticFloorGateV1) -> str:
    evidence_rows = "\n".join(
        f"| {row.issue_id} | `{row.path}` | `{row.sha256}` | {row.status}/{row.claim_class} |"
        for row in gate.evidence
    )
    allowed = "\n".join(f"- `{claim}`" for claim in gate.allowed_claims)
    blocked = "\n".join(f"- `{claim}`" for claim in gate.blocked_claims)
    failures = "\n".join(f"- {reason}" for reason in gate.resolving_evidence)
    return f"""# SemanticFloorGateV1 closeout (SLM-213)

**Schema:** `{gate.schema_version}`
**Verdict:** **{gate.verdict}**
**Gate hash:** `{gate.gate_hash}`
**Evidence cutoff:** `{gate.evidence_cutoff}`

## Decision

The SDE5 evidence is fixture/wiring only. Strict binding-aware meaning-v2 was
not measured on a preregistered real evaluation, and checkpoint, data,
anti-gaming, paired-statistics, and AgentV identities are unresolved. The
semantic floor therefore remains **inconclusive**, not escaped or rejected.

Constraint-debt, structural, spectral, and recurrent/latent diagnostics remain
usable only as explicitly scoped proxies.

## Allowed claims

{allowed}

## Blocked claims

{blocked}

## Mediators

- Constraint debt: synthetic instrumentation/selection/routing signals exist.
- Protected objectives: not measured in a trained matched campaign.
- Strict meaning-v2: unmeasured (`n={gate.sample_sizes.get("strict_meaning_v2", 0)}`).
- Legacy meaning-v1: no SDE5 result was supplied.
- Anti-gaming: {gate.anti_gaming.get("status")}; scheduled cells are not pass evidence.
- AgentV: {gate.agentv_evaluation.get("status")} (`n={gate.agentv_evaluation.get("n", 0)}`).

## Resolving evidence still required

{failures}

## Source artifacts

| issue | path | SHA-256 | status/claim |
| --- | --- | --- | --- |
{evidence_rows}

## Reproduction

```bash
python -m scripts.publish_semantic_floor_gate --check
pytest -q tests/test_harnesses/experiments/test_semantic_floor_gate.py tests/test_scripts/test_publish_semantic_floor_gate.py
```

No training, decoder experiment, evaluator change, gate-threshold change, or
checkpoint promotion was performed.
"""


def canonical_gate_json(gate: SemanticFloorGateV1) -> str:
    return json.dumps(gate.to_dict(), indent=2, sort_keys=True) + "\n"


def validate_gate_references(gate: SemanticFloorGateV1, *, repo_root: Path) -> list[str]:
    failures: list[str] = []
    for ref in gate.evidence:
        relative = Path(ref.path)
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"{ref.issue_id}: unsafe path {ref.path}")
            continue
        path = repo_root / relative
        if not path.is_file():
            failures.append(f"{ref.issue_id}: missing {ref.path}")
        elif _sha256(path) != ref.sha256:
            failures.append(f"{ref.issue_id}: SHA-256 mismatch for {ref.path}")
    return failures


__all__ = [
    "DEFAULT_GATE_PATH",
    "EVIDENCE_PATHS",
    "EvidenceReferenceV1",
    "PROXY_CLAIM_CLASSES",
    "SCHEMA_VERSION",
    "SEMANTIC_CLAIM_CLASSES",
    "SemanticFloorGateV1",
    "build_semantic_floor_gate",
    "canonical_gate_json",
    "decide_verdict",
    "load_semantic_floor_gate",
    "render_markdown",
    "require_floor_gate",
    "validate_gate_references",
]
