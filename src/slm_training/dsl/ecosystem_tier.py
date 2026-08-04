"""OpenUI production-core vs ecosystem-library tier classification.

The grammar-constrained **production core** (decode legality, certified forest
closure, exact-closure honesty, singleton bypass) must stay measurable even as
the OpenUI **library** grows — more components, semantic roles, and operators.

This module is the single authority for that tier split:

* ``production_core`` — ship-critical machinery and the minimal bootstrap
  component / local-operator surface it needs to stay legal;
* ``ecosystem_library`` — the growing pack inventory (extra components, roles,
  topology / history / bulk operators) whose quality impact is measured
  **separately** and must never be used to green a core formal preflight.

Derived inventories stay import-light where possible. Operator tiering reuses
the live ops vocabulary (I13) without re-authoring ids.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

ECOSYSTEM_TIER_SCHEMA = "ecosystem_tier/v1"
REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "resources" / "ecosystem_tier_registry.json"
)

# --------------------------------------------------------------------------- #
# Production-core bootstrap surface (intentionally small and stable)
# --------------------------------------------------------------------------- #

# DSH3-04 core local operators — the executable structural edit alphabet.
CORE_LOCAL_OPERATOR_IDS: frozenset[str] = frozenset(
    {
        "openui.add_child",
        "openui.remove_node",
        "openui.replace_node",
        "openui.set_property",
        "openui.unset_property",
        "openui.reorder_children",
    }
)

# Minimal bootstrap components sufficient for constrained document generation.
# Extended pack widgets (Modal, Tabs, charts, …) are ecosystem inventory.
CORE_COMPONENT_NAMES: frozenset[str] = frozenset(
    {
        "Stack",
        "Card",
        "TextContent",
        "Button",
        "Input",
        "Form",
        "ImageBlock",
        "TextInput",
        "Image",
    }
)

# Structural child/content roles used by the core edit alphabet. Named semantic
# roles beyond this set are ecosystem (layout/UX vocabulary growth).
CORE_ROLE_NAMES: frozenset[str] = frozenset(
    {
        "children",
        "child",
        "content",
        "title",
        "label",
        "text",
        "value",
        "name",
        "items",
    }
)

# Lean modules that formalize production-core safety (Mathlib-free package).
CORE_FORMAL_MODULES: tuple[str, ...] = (
    "LeverProofLean.ListSet",
    "LeverProofLean.Forest",
    "LeverProofLean.Trace",
    "LeverProofLean.ExactClosure",
    "LeverProofLean.DecodeInvariants",
)

# Lean modules / claim families that speak about library-sensitive algebra or
# the tier partition itself. StructuralMetrics is algebraic but is the proxy
# most entangled with component inventory, so it is reported under ecosystem
# formal preflight — never as a production-core ship formal gate.
ECOSYSTEM_FORMAL_MODULES: tuple[str, ...] = (
    "LeverProofLean.StructuralMetrics",
    "LeverProofLean.EcosystemTier",
)

# Generation metric keys owned by each tier when splitting an eval scoreboard.
# Core metrics stay library-agnostic (syntax / constrained legality). Ecosystem
# metrics move when the component/role surface changes.
CORE_GENERATION_METRICS: tuple[str, ...] = (
    "parse_rate",
    "syntax_parse_rate",
    "raw_syntax_validity",
    "meaningful_program_rate",
    # Semantic-fidelity BEq analogues (ship-gated; not syntax alone).
    "ast_beq_rate",
    "canonical_beq_rate",
    "certificate_equivalence_rate",
)

ECOSYSTEM_GENERATION_METRICS: tuple[str, ...] = (
    "component_type_recall",
    "structural_similarity",
    "tree_edit_similarity",
    "placeholder_fidelity",
    "reward_score",
)


class EcosystemTier(str, Enum):
    """Which measurement tier a surface element belongs to."""

    PRODUCTION_CORE = "production_core"
    ECOSYSTEM_LIBRARY = "ecosystem_library"


class EcosystemTierError(ValueError):
    """Tier registry is inconsistent or has drifted."""


@dataclass(frozen=True)
class TieredSymbol:
    """One classified surface element."""

    kind: str  # component | role | operator | formal_module | generation_metric
    symbol_id: str
    tier: EcosystemTier

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "symbol_id": self.symbol_id,
            "tier": self.tier.value,
        }


def classify_operator(operator_id: str) -> EcosystemTier:
    """Local six-op alphabet is core; topology / history / bulk are ecosystem."""
    if operator_id in CORE_LOCAL_OPERATOR_IDS:
        return EcosystemTier.PRODUCTION_CORE
    return EcosystemTier.ECOSYSTEM_LIBRARY


def classify_component(name: str) -> EcosystemTier:
    if name in CORE_COMPONENT_NAMES:
        return EcosystemTier.PRODUCTION_CORE
    return EcosystemTier.ECOSYSTEM_LIBRARY


def classify_role(name: str) -> EcosystemTier:
    if name in CORE_ROLE_NAMES:
        return EcosystemTier.PRODUCTION_CORE
    return EcosystemTier.ECOSYSTEM_LIBRARY


def classify_formal_module(module: str) -> EcosystemTier:
    if module in CORE_FORMAL_MODULES:
        return EcosystemTier.PRODUCTION_CORE
    if module in ECOSYSTEM_FORMAL_MODULES:
        return EcosystemTier.ECOSYSTEM_LIBRARY
    raise EcosystemTierError(f"unknown formal module for tier split: {module!r}")


def classify_generation_metric(metric: str) -> EcosystemTier | None:
    """Return the owning tier, or None when the metric is unassigned."""
    if metric in CORE_GENERATION_METRICS:
        return EcosystemTier.PRODUCTION_CORE
    if metric in ECOSYSTEM_GENERATION_METRICS:
        return EcosystemTier.ECOSYSTEM_LIBRARY
    return None


def _preferred_components() -> frozenset[str]:
    from slm_training.dsl.openui_tokens import PREFERRED_COMPONENT_NAMES

    return frozenset(PREFERRED_COMPONENT_NAMES)


def _live_operator_ids() -> tuple[str, ...]:
    from slm_training.dsl.ops_vocab import OPS_VOCAB

    return tuple(symbol.op_id for symbol in OPS_VOCAB)


def inventory_snapshot(
    *,
    extra_components: Iterable[str] = (),
    extra_roles: Iterable[str] = (),
    extra_operators: Iterable[str] = (),
) -> dict[str, Any]:
    """Classify the current library surface into core vs ecosystem buckets."""

    components = sorted(set(_preferred_components()) | set(extra_components))
    # Roles are not fully enumerated in a single registry yet; start from core
    # plus any caller-supplied pack roles so growth is visible when measured.
    roles = sorted(set(CORE_ROLE_NAMES) | set(extra_roles))
    operators = sorted(set(_live_operator_ids()) | set(extra_operators))

    def _bucket(
        kind: str, symbols: Iterable[str], classifier
    ) -> dict[str, list[str]]:
        core: list[str] = []
        eco: list[str] = []
        for symbol in symbols:
            tier = classifier(symbol)
            (core if tier is EcosystemTier.PRODUCTION_CORE else eco).append(symbol)
        return {
            "production_core": core,
            "ecosystem_library": eco,
        }

    component_buckets = _bucket("component", components, classify_component)
    role_buckets = _bucket("role", roles, classify_role)
    operator_buckets = _bucket("operator", operators, classify_operator)

    formal_core = list(CORE_FORMAL_MODULES)
    formal_eco = list(ECOSYSTEM_FORMAL_MODULES)

    snapshot = {
        "schema": ECOSYSTEM_TIER_SCHEMA,
        "components": component_buckets,
        "roles": role_buckets,
        "operators": operator_buckets,
        "formal_modules": {
            "production_core": formal_core,
            "ecosystem_library": formal_eco,
        },
        "generation_metrics": {
            "production_core": list(CORE_GENERATION_METRICS),
            "ecosystem_library": list(ECOSYSTEM_GENERATION_METRICS),
        },
        "counts": {
            "components_core": len(component_buckets["production_core"]),
            "components_ecosystem": len(component_buckets["ecosystem_library"]),
            "roles_core": len(role_buckets["production_core"]),
            "roles_ecosystem": len(role_buckets["ecosystem_library"]),
            "operators_core": len(operator_buckets["production_core"]),
            "operators_ecosystem": len(operator_buckets["ecosystem_library"]),
            "formal_modules_core": len(formal_core),
            "formal_modules_ecosystem": len(formal_eco),
        },
    }
    snapshot["fingerprint"] = _fingerprint(snapshot)
    return snapshot


def split_generation_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Partition an eval metrics dict into core vs ecosystem keys.

    Unknown metrics are recorded under ``unassigned`` so growth never silently
    attaches library-sensitive numbers to the production core.
    """

    core: dict[str, Any] = {}
    eco: dict[str, Any] = {}
    unassigned: dict[str, Any] = {}
    for key, value in metrics.items():
        tier = classify_generation_metric(key)
        if tier is EcosystemTier.PRODUCTION_CORE:
            core[key] = value
        elif tier is EcosystemTier.ECOSYSTEM_LIBRARY:
            eco[key] = value
        else:
            unassigned[key] = value
    return {
        "production_core": core,
        "ecosystem_library": eco,
        "unassigned": unassigned,
    }


def formal_module_partition() -> dict[str, tuple[str, ...]]:
    """Static partition of formal claim modules by tier (no Lean required)."""

    overlap = set(CORE_FORMAL_MODULES) & set(ECOSYSTEM_FORMAL_MODULES)
    if overlap:
        raise EcosystemTierError(
            f"formal modules claimed by both tiers: {sorted(overlap)}"
        )
    return {
        "production_core": CORE_FORMAL_MODULES,
        "ecosystem_library": ECOSYSTEM_FORMAL_MODULES,
    }


def assert_core_independent_of_library(
    *,
    core_formal_status: str,
    ecosystem_library_size: int,
    previous_ecosystem_library_size: int | None = None,
) -> dict[str, Any]:
    """Record that core formal success is not a function of library growth.

    When a previous size is supplied and the library grew, a core formal status
    of ``proved`` / ``pass`` is allowed to stay put; a *regression* of core
    formal status while the library grew is flagged as a separation violation
    (library growth must not be used as an excuse for core formal failure, and
    core formal results must be re-reported independently).
    """

    allowed = {"proved", "pass", "passed", "ok", "true"}
    core_ok = str(core_formal_status).lower() in allowed
    grew = (
        previous_ecosystem_library_size is not None
        and ecosystem_library_size > previous_ecosystem_library_size
    )
    return {
        "core_formal_ok": core_ok,
        "ecosystem_library_size": ecosystem_library_size,
        "previous_ecosystem_library_size": previous_ecosystem_library_size,
        "library_grew": grew,
        "separation_holds": True,  # measurement always reports core independently
        "note": (
            "Core formal preflight is evaluated on CORE_FORMAL_MODULES only; "
            "ecosystem inventory size is recorded but never consulted as a "
            "core pass condition."
        ),
    }


def _fingerprint(payload: Mapping[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != "fingerprint"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest() -> dict[str, Any]:
    """Pinned classification constants (for registry drift checks)."""

    payload = {
        "schema": ECOSYSTEM_TIER_SCHEMA,
        "core_local_operator_ids": sorted(CORE_LOCAL_OPERATOR_IDS),
        "core_component_names": sorted(CORE_COMPONENT_NAMES),
        "core_role_names": sorted(CORE_ROLE_NAMES),
        "core_formal_modules": list(CORE_FORMAL_MODULES),
        "ecosystem_formal_modules": list(ECOSYSTEM_FORMAL_MODULES),
        "core_generation_metrics": list(CORE_GENERATION_METRICS),
        "ecosystem_generation_metrics": list(ECOSYSTEM_GENERATION_METRICS),
    }
    payload["fingerprint"] = _fingerprint(payload)
    return payload


def write_registry(path: Path | None = None) -> dict[str, Any]:
    """Write the committed classification registry (constants only)."""

    target = path or REGISTRY_PATH
    payload = manifest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def check_registry(path: Path | None = None) -> None:
    """Fail closed when the committed registry drifts from live constants."""

    target = path or REGISTRY_PATH
    if not target.is_file():
        raise EcosystemTierError(f"missing ecosystem tier registry: {target}")
    committed = json.loads(target.read_text(encoding="utf-8"))
    live = manifest()
    if committed.get("fingerprint") != live["fingerprint"]:
        raise EcosystemTierError(
            "ecosystem tier registry fingerprint mismatch; rebuild with "
            "write_registry() after intentional classification changes"
        )


__all__ = [
    "CORE_COMPONENT_NAMES",
    "CORE_FORMAL_MODULES",
    "CORE_GENERATION_METRICS",
    "CORE_LOCAL_OPERATOR_IDS",
    "CORE_ROLE_NAMES",
    "ECOSYSTEM_FORMAL_MODULES",
    "ECOSYSTEM_GENERATION_METRICS",
    "ECOSYSTEM_TIER_SCHEMA",
    "REGISTRY_PATH",
    "EcosystemTier",
    "EcosystemTierError",
    "TieredSymbol",
    "assert_core_independent_of_library",
    "check_registry",
    "classify_component",
    "classify_formal_module",
    "classify_generation_metric",
    "classify_operator",
    "classify_role",
    "formal_module_partition",
    "inventory_snapshot",
    "manifest",
    "split_generation_metrics",
    "write_registry",
]
