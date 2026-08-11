"""EVID-08 / SLM-535: checker capabilities, trust domains, acceptance policies.

Own capability schema + policies for FormalAuthorityV2 / mutation_suite /
structural checkers — not a third evidence stack. Distinct backend *names* are
not automatically independent trust roots.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

CheckerCapabilityName = Literal[
    "structural_consistency",
    "exact_proposition",
    "axiom_audit",
    "semantic_replay",
    "certificate_checking",
    "encoding_correctness",
    "runtime_refinement",
    "provenance_only",
    "toolchain_lock",
    "source_identity",
    "module_binding",
    "binding_identity",
    "trust_domain_diversity",
]

CheckerRole = Literal[
    "independent",
    "redundant_conformance",
    "provenance_only",
]

VerificationKind = Literal[
    "independent_verification",
    "redundant_conformance",
    "rejected",
    "incomplete",
]

CAPABILITY_VOCABULARY: frozenset[str] = frozenset(
    {
        "structural_consistency",
        "exact_proposition",
        "axiom_audit",
        "semantic_replay",
        "certificate_checking",
        "encoding_correctness",
        "runtime_refinement",
        "provenance_only",
        "toolchain_lock",
        "source_identity",
        "module_binding",
        "binding_identity",
        "trust_domain_diversity",
    }
)

# Implementation / trust domains (language/runtime/parser/kernel/encoding family).
TRUST_DOMAIN_PYTHON_STRUCTURAL_FAMILY = "python_cpython_formal_structural_family"
TRUST_DOMAIN_PYTHON_REPLAY = "python_enumerative_replay"
TRUST_DOMAIN_LEAN4_KERNEL = "lean4_kernel"
TRUST_DOMAIN_ENCODING_BRIDGE = "python_encoding_bridge"

CHECKER_CAPABILITY_SCHEMA = "checker_capability_registry/v1"
POLICY_REPORT_SCHEMA = "checker_acceptance_policy_report/v1"
REGISTRY_RELPATH = (
    "src/slm_training/resources/formal/checker_capability_registry.v1.json"
)

# Production defaults (also mirrored in the registry JSON).
CHECKER_TRUST_DOMAINS: Mapping[str, str] = {
    "python_structural": TRUST_DOMAIN_PYTHON_STRUCTURAL_FAMILY,
    "python_reference": TRUST_DOMAIN_PYTHON_STRUCTURAL_FAMILY,
    "python_replay": TRUST_DOMAIN_PYTHON_REPLAY,
    "lean_kernel": TRUST_DOMAIN_LEAN4_KERNEL,
    "python_encoding_ref": TRUST_DOMAIN_ENCODING_BRIDGE,
}


@dataclass(frozen=True)
class CheckerAdvertisementV1:
    """What a named checker actually claims to prove."""

    backend: str
    capabilities: frozenset[str]
    trust_domain: str
    role: CheckerRole
    required_evidence_inputs: tuple[str, ...]
    language_runtime: str
    notes: str = ""

    def advertises(self, capability: str) -> bool:
        return capability in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "capabilities": sorted(self.capabilities),
            "trust_domain": self.trust_domain,
            "role": self.role,
            "required_evidence_inputs": list(self.required_evidence_inputs),
            "language_runtime": self.language_runtime,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckerAdvertisementV1:
        caps = frozenset(str(item) for item in data["capabilities"])
        unknown = caps - CAPABILITY_VOCABULARY
        if unknown:
            raise ValueError(f"unknown checker capabilities: {sorted(unknown)}")
        role = str(data["role"])
        if role not in {"independent", "redundant_conformance", "provenance_only"}:
            raise ValueError(f"unknown checker role: {role!r}")
        return cls(
            backend=str(data["backend"]),
            capabilities=caps,
            trust_domain=str(data["trust_domain"]),
            role=role,  # type: ignore[arg-type]
            required_evidence_inputs=tuple(
                str(item) for item in data.get("required_evidence_inputs", ())
            ),
            language_runtime=str(data.get("language_runtime", "")),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class AcceptancePolicyV1:
    """Required capabilities + minimum distinct trust domains."""

    policy_id: str
    required_capabilities: frozenset[str]
    min_distinct_trust_domains: int
    require_independent_verification: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "required_capabilities": sorted(self.required_capabilities),
            "min_distinct_trust_domains": self.min_distinct_trust_domains,
            "require_independent_verification": self.require_independent_verification,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AcceptancePolicyV1:
        caps = frozenset(str(item) for item in data["required_capabilities"])
        unknown = caps - CAPABILITY_VOCABULARY
        if unknown:
            raise ValueError(f"unknown policy capabilities: {sorted(unknown)}")
        return cls(
            policy_id=str(data["policy_id"]),
            required_capabilities=caps,
            min_distinct_trust_domains=int(data["min_distinct_trust_domains"]),
            require_independent_verification=bool(
                data["require_independent_verification"]
            ),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class CheckerRunView:
    """Minimal checker run view for policy evaluation."""

    backend: str
    ok: bool
    skipped: bool = False
    status: str | None = None  # pass|fail|skipped|unknown when from authority refs

    @property
    def is_incomplete(self) -> bool:
        if self.skipped:
            return True
        if self.status in {"skipped", "unknown"}:
            return True
        return False

    @property
    def is_successful_run(self) -> bool:
        """Skipped/unknown never count as successful semantic checks."""

        if self.is_incomplete:
            return False
        if self.status == "fail":
            return False
        return bool(self.ok)


@dataclass(frozen=True)
class PolicyReportV1:
    """Acceptance report distinguishing redundancy from independent verification."""

    schema: str
    policy_id: str
    accepted: bool
    verification_kind: VerificationKind
    reason: str
    required_capabilities: tuple[str, ...]
    satisfied_capabilities: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    unadvertised_claims: tuple[str, ...]
    trust_domains: tuple[str, ...]
    independent_backends: tuple[str, ...]
    redundant_backends: tuple[str, ...]
    skipped_backends: tuple[str, ...]
    incomplete_backends: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "accepted": self.accepted,
            "verification_kind": self.verification_kind,
            "reason": self.reason,
            "required_capabilities": list(self.required_capabilities),
            "satisfied_capabilities": list(self.satisfied_capabilities),
            "missing_capabilities": list(self.missing_capabilities),
            "unadvertised_claims": list(self.unadvertised_claims),
            "trust_domains": list(self.trust_domains),
            "independent_backends": list(self.independent_backends),
            "redundant_backends": list(self.redundant_backends),
            "skipped_backends": list(self.skipped_backends),
            "incomplete_backends": list(self.incomplete_backends),
        }


# Built-in advertisements (registry JSON is the durable mirror).
_DEFAULT_ADVERTISEMENTS: tuple[CheckerAdvertisementV1, ...] = (
    CheckerAdvertisementV1(
        backend="python_structural",
        capabilities=frozenset({"structural_consistency"}),
        trust_domain=TRUST_DOMAIN_PYTHON_STRUCTURAL_FAMILY,
        role="redundant_conformance",
        required_evidence_inputs=("formal_object",),
        language_runtime="python_cpython",
        notes=(
            "List-based honesty/law encoding over FormalObjectV1. Shares "
            "parser/runtime/semantics with python_reference — not an "
            "independent trust root."
        ),
    ),
    CheckerAdvertisementV1(
        backend="python_reference",
        capabilities=frozenset({"structural_consistency"}),
        trust_domain=TRUST_DOMAIN_PYTHON_STRUCTURAL_FAMILY,
        role="redundant_conformance",
        required_evidence_inputs=("formal_object",),
        language_runtime="python_cpython",
        notes=(
            "Set/digest-first re-encoding of the same laws. Redundant "
            "conformance check in the shared structural family."
        ),
    ),
    CheckerAdvertisementV1(
        backend="python_replay",
        capabilities=frozenset({"semantic_replay", "certificate_checking"}),
        trust_domain=TRUST_DOMAIN_PYTHON_REPLAY,
        role="independent",
        required_evidence_inputs=(
            "support_certificate",
            "expander",
            "verifier",
        ),
        language_runtime="python_cpython",
        notes="Enumerative replay; skipped without live expander+verifier context.",
    ),
    CheckerAdvertisementV1(
        backend="lean_kernel",
        capabilities=frozenset(
            {
                "exact_proposition",
                "axiom_audit",
                "toolchain_lock",
                "source_identity",
                "module_binding",
            }
        ),
        trust_domain=TRUST_DOMAIN_LEAN4_KERNEL,
        role="independent",
        required_evidence_inputs=(
            "theorem_binding",
            "lean_source_tree",
            "lake_manifest",
        ),
        language_runtime="lean4",
        notes="Exact binding + axiom audit. Never sole semantic authority.",
    ),
    CheckerAdvertisementV1(
        backend="python_encoding_ref",
        capabilities=frozenset(
            {"encoding_correctness", "certificate_checking", "binding_identity"}
        ),
        trust_domain=TRUST_DOMAIN_ENCODING_BRIDGE,
        role="independent",
        required_evidence_inputs=(
            "original_problem",
            "encoded_problem",
            "certificate",
        ),
        language_runtime="python_cpython",
        notes="Verified encoding bridge (EVID-10).",
    ),
)

CONFORMANCE_POLICY = AcceptancePolicyV1(
    policy_id="formal_object_conformance/v1",
    required_capabilities=frozenset({"structural_consistency"}),
    min_distinct_trust_domains=1,
    require_independent_verification=False,
    notes=(
        "Structural/reference agreement is redundant conformance inside one "
        "trust domain — not independent semantic verification."
    ),
)

SEMANTIC_AUTHORITY_POLICY = AcceptancePolicyV1(
    policy_id="formal_semantic_authority/v1",
    required_capabilities=frozenset(),
    min_distinct_trust_domains=2,
    require_independent_verification=True,
    notes=(
        "Checked semantic authority needs an independent trust domain. Shared "
        "python_structural/python_reference conformance alone fails. A single "
        "independent backend (encoding/replay/lean) may authorize; two nominal "
        "backends in one domain never count as diversity."
    ),
)

DEFAULT_POLICIES: Mapping[str, AcceptancePolicyV1] = {
    CONFORMANCE_POLICY.policy_id: CONFORMANCE_POLICY,
    SEMANTIC_AUTHORITY_POLICY.policy_id: SEMANTIC_AUTHORITY_POLICY,
}


def default_advertisements() -> dict[str, CheckerAdvertisementV1]:
    return {item.backend: item for item in _DEFAULT_ADVERTISEMENTS}


def registry_path(*, root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parents[3]
    return base / REGISTRY_RELPATH


def load_registry(*, root: Path | None = None) -> dict[str, Any]:
    path = registry_path(root=root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CHECKER_CAPABILITY_SCHEMA:
        raise ValueError(f"expected schema {CHECKER_CAPABILITY_SCHEMA!r}")
    return payload


def advertisements_from_registry(
    *, root: Path | None = None
) -> dict[str, CheckerAdvertisementV1]:
    payload = load_registry(root=root)
    out: dict[str, CheckerAdvertisementV1] = {}
    for item in payload["checkers"]:
        adv = CheckerAdvertisementV1.from_dict(item)
        out[adv.backend] = adv
    return out


def trust_domain_for(
    backend: str,
    *,
    advertisements: Mapping[str, CheckerAdvertisementV1] | None = None,
) -> str:
    table = advertisements or default_advertisements()
    if backend in table:
        return table[backend].trust_domain
    return CHECKER_TRUST_DOMAINS.get(backend, f"unknown:{backend}")


def semantic_authority_requires_distinct_trust_domains(
    backends_ok: Sequence[str],
    *,
    min_domains: int = 2,
    advertisements: Mapping[str, CheckerAdvertisementV1] | None = None,
) -> bool:
    """Semantic authority needs distinct trust domains, not just backend names.

    ``python_structural`` and ``python_reference`` share a parser/runtime
    family — agreeing alone cannot confer independent semantic authority.
    """

    domains = {
        trust_domain_for(name, advertisements=advertisements) for name in backends_ok
    }
    return len(domains) >= min_domains


def capability_satisfied_by(
    capability: str,
    *,
    backend: str,
    advertisements: Mapping[str, CheckerAdvertisementV1] | None = None,
) -> bool:
    """A checker cannot satisfy a capability it does not explicitly advertise."""

    if capability not in CAPABILITY_VOCABULARY:
        return False
    table = advertisements or default_advertisements()
    adv = table.get(backend)
    if adv is None:
        return False
    return adv.advertises(capability)


def _partition_backends(
    runs: Sequence[CheckerRunView],
    *,
    advertisements: Mapping[str, CheckerAdvertisementV1],
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    set[str],
]:
    successful: list[str] = []
    skipped: list[str] = []
    incomplete: list[str] = []
    independent: list[str] = []
    redundant: list[str] = []
    domains: set[str] = set()
    for run in runs:
        if run.is_incomplete:
            incomplete.append(run.backend)
            if run.skipped or run.status in {"skipped", "unknown"}:
                skipped.append(run.backend)
            continue
        if not run.is_successful_run:
            continue
        if run.backend not in successful:
            successful.append(run.backend)
        adv = advertisements.get(run.backend)
        domain = trust_domain_for(run.backend, advertisements=advertisements)
        domains.add(domain)
        if adv is None:
            # Unknown backend: treat as independent unknown domain, never silent.
            if run.backend not in independent:
                independent.append(run.backend)
            continue
        if adv.role == "redundant_conformance":
            if run.backend not in redundant:
                redundant.append(run.backend)
        elif adv.role == "provenance_only":
            # Provenance-only never counts toward independent or conformance ok.
            continue
        else:
            if run.backend not in independent:
                independent.append(run.backend)
    return successful, independent, redundant, skipped, incomplete, domains


def evaluate_acceptance_policy(
    runs: Sequence[CheckerRunView],
    policy: AcceptancePolicyV1,
    *,
    claimed_capabilities: Iterable[str] | None = None,
    advertisements: Mapping[str, CheckerAdvertisementV1] | None = None,
) -> PolicyReportV1:
    """Evaluate runs against a policy.

    - Unadvertised capability claims are never satisfied.
    - Skipped/unknown runs are not successful semantic checks.
    - Reports distinguish redundant conformance from independent verification.
    """

    table = advertisements or default_advertisements()
    claimed = tuple(sorted(set(claimed_capabilities or ())))
    unadvertised: list[str] = []
    for cap in claimed:
        if cap not in CAPABILITY_VOCABULARY:
            unadvertised.append(f"unknown_vocab:{cap}")
            continue
        # A claim is unadvertised if *no* successful non-skipped backend advertises it.
        advertisers = [
            run.backend
            for run in runs
            if run.is_successful_run and capability_satisfied_by(cap, backend=run.backend, advertisements=table)
        ]
        if not advertisers:
            # Also flag if claim was made without any advertisement in registry.
            if not any(adv.advertises(cap) for adv in table.values()):
                unadvertised.append(cap)

    (
        _successful,
        independent,
        redundant,
        skipped,
        incomplete,
        domains,
    ) = _partition_backends(runs, advertisements=table)

    satisfied: list[str] = []
    missing: list[str] = []
    for cap in sorted(policy.required_capabilities):
        ok_backends = [
            run.backend
            for run in runs
            if run.is_successful_run
            and capability_satisfied_by(cap, backend=run.backend, advertisements=table)
        ]
        if ok_backends:
            satisfied.append(cap)
        else:
            missing.append(cap)

    domain_ok = len(domains) >= policy.min_distinct_trust_domains
    has_independent = bool(independent)
    only_redundant = bool(redundant) and not has_independent
    sole_independent = (
        has_independent
        and len(independent) == 1
        and not redundant
        and len(domains) == 1
    )
    if incomplete and policy.require_independent_verification:
        # Incomplete evidence cannot close a semantic policy.
        verification_kind: VerificationKind = "incomplete"
        accepted = False
        reason = (
            "incomplete/skipped checker evidence cannot satisfy semantic policy "
            f"({', '.join(incomplete)})"
        )
    elif missing or unadvertised:
        verification_kind = "rejected"
        accepted = False
        parts = []
        if missing:
            parts.append(f"missing capabilities: {', '.join(missing)}")
        if unadvertised:
            parts.append(f"unadvertised claims: {', '.join(unadvertised)}")
        reason = "; ".join(parts)
    elif policy.require_independent_verification and only_redundant:
        verification_kind = "redundant_conformance"
        accepted = False
        reason = (
            "redundant conformance checkers share a trust domain and cannot "
            f"authorize semantic verification ({', '.join(redundant)})"
        )
    elif policy.require_independent_verification and not has_independent:
        verification_kind = "rejected"
        accepted = False
        reason = "policy requires independent verification; no successful independent backend"
    elif policy.require_independent_verification and not domain_ok and not sole_independent:
        verification_kind = "rejected"
        accepted = False
        reason = (
            f"need {policy.min_distinct_trust_domains} distinct trust domains "
            f"(or one sole independent backend), got {len(domains)} "
            f"({', '.join(sorted(domains)) or 'none'})"
        )
    elif policy.require_independent_verification:
        verification_kind = "independent_verification"
        accepted = True
        reason = (
            "independent verification across trust domains: "
            + ", ".join(sorted(domains))
        )
    else:
        # Conformance policy: redundant same-domain agreement is allowed.
        if has_independent and domain_ok:
            verification_kind = "independent_verification"
        else:
            verification_kind = "redundant_conformance"
        accepted = True
        reason = (
            "conformance accepted "
            f"({verification_kind}; domains={', '.join(sorted(domains)) or 'none'})"
        )

    return PolicyReportV1(
        schema=POLICY_REPORT_SCHEMA,
        policy_id=policy.policy_id,
        accepted=accepted,
        verification_kind=verification_kind,
        reason=reason,
        required_capabilities=tuple(sorted(policy.required_capabilities)),
        satisfied_capabilities=tuple(satisfied),
        missing_capabilities=tuple(missing),
        unadvertised_claims=tuple(unadvertised),
        trust_domains=tuple(sorted(domains)),
        independent_backends=tuple(independent),
        redundant_backends=tuple(redundant),
        skipped_backends=tuple(dict.fromkeys(skipped)),
        incomplete_backends=tuple(dict.fromkeys(incomplete)),
    )


def runs_from_checker_results(results: Sequence[Any]) -> tuple[CheckerRunView, ...]:
    """Adapt CheckerResult / CheckerResultRefV1-like objects."""

    views: list[CheckerRunView] = []
    for item in results:
        if hasattr(item, "skipped"):
            views.append(
                CheckerRunView(
                    backend=str(item.backend),
                    ok=bool(item.ok),
                    skipped=bool(item.skipped),
                )
            )
            continue
        status = getattr(item, "status", None)
        if status is not None:
            status_s = str(status)
            views.append(
                CheckerRunView(
                    backend=str(item.backend),
                    ok=status_s == "pass",
                    skipped=status_s in {"skipped", "unknown"},
                    status=status_s,
                )
            )
            continue
        views.append(
            CheckerRunView(
                backend=str(item.backend),
                ok=bool(getattr(item, "ok", False)),
                skipped=False,
            )
        )
    return tuple(views)


def build_registry_payload() -> dict[str, Any]:
    """Canonical registry document (kept in sync with defaults)."""

    return {
        "schema": CHECKER_CAPABILITY_SCHEMA,
        "capability_vocabulary": sorted(CAPABILITY_VOCABULARY),
        "trust_domains": {
            TRUST_DOMAIN_PYTHON_STRUCTURAL_FAMILY: {
                "language_runtime": "python_cpython",
                "parser_family": "formal_object_v1_python",
                "kernel": None,
                "encoding_family": None,
                "notes": (
                    "Shared by python_structural and python_reference — "
                    "redundant conformance, not independent roots."
                ),
            },
            TRUST_DOMAIN_PYTHON_REPLAY: {
                "language_runtime": "python_cpython",
                "parser_family": "vss_support_certificate",
                "kernel": None,
                "encoding_family": None,
                "notes": "Enumerative replay path with expander+verifier.",
            },
            TRUST_DOMAIN_LEAN4_KERNEL: {
                "language_runtime": "lean4",
                "parser_family": "lean4_syntax",
                "kernel": "lean4",
                "encoding_family": None,
                "notes": "Lean 4 kernel + theorem binding.",
            },
            TRUST_DOMAIN_ENCODING_BRIDGE: {
                "language_runtime": "python_cpython",
                "parser_family": "cnf_ref_encoding",
                "kernel": None,
                "encoding_family": "cnf_ref",
                "notes": "Verified encoding adapter bridge.",
            },
        },
        "checkers": [item.to_dict() for item in _DEFAULT_ADVERTISEMENTS],
        "policies": [p.to_dict() for p in DEFAULT_POLICIES.values()],
    }


__all__ = [
    "CAPABILITY_VOCABULARY",
    "CHECKER_CAPABILITY_SCHEMA",
    "CHECKER_TRUST_DOMAINS",
    "CONFORMANCE_POLICY",
    "POLICY_REPORT_SCHEMA",
    "SEMANTIC_AUTHORITY_POLICY",
    "TRUST_DOMAIN_ENCODING_BRIDGE",
    "TRUST_DOMAIN_LEAN4_KERNEL",
    "TRUST_DOMAIN_PYTHON_REPLAY",
    "TRUST_DOMAIN_PYTHON_STRUCTURAL_FAMILY",
    "AcceptancePolicyV1",
    "CheckerAdvertisementV1",
    "CheckerCapabilityName",
    "CheckerRole",
    "CheckerRunView",
    "PolicyReportV1",
    "VerificationKind",
    "advertisements_from_registry",
    "build_registry_payload",
    "capability_satisfied_by",
    "default_advertisements",
    "evaluate_acceptance_policy",
    "load_registry",
    "registry_path",
    "runs_from_checker_results",
    "semantic_authority_requires_distinct_trust_domains",
    "trust_domain_for",
]
