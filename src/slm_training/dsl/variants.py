"""VariantContractV1 (VAR0-01): scope facts as pack-, kernel-, or variant-owned.

The repository builds several variants of one base — the REPL/operator
surface, the tree-edit diffusion editor, and the TwoTower prompt->AST
denoiser — but "variant" was not a checked concept before this module. A
given fact (component inventory, op identity, action encoding, seed policy)
is either **shared across all variants** (pack-owned or kernel-owned) or
**owned by one variant** (variant-owned); mixing the two up is how a variant
re-declares pack facts as module constants (see ``tree_edit_diffusion.py``'s
``LEAF_COMPONENTS`` / ``CONTAINER_COMPONENTS``) instead of reading them from
their one authority.

Like :mod:`slm_training.dsl.ops_vocab`, this registry is **derived, not
authored**: :data:`VARIANTS` is built from each variant's live source files
(module-level action constants, the live operator registry, the shared
structural-token vocabulary) rather than hand-maintained, and is pinned
against ``resources/variant_registry.json`` by
``scripts/verify_decode_invariants.py::check_variant_contracts``. Building
this module must stay torch-free (`_ensure_builtin_packs` on
:mod:`slm_training.dsl.pack` deliberately keeps that module import-light) so
the decode-invariants CI job never needs a model backend.

``inventory_source`` records the current, honest state of the pack-owned
bucket for each variant -- ``"pack"`` when the variant's code actually reads
component inventory through a pack/kernel authority (checked statically:
does its source file import ``slm_training.dsl.pack``?), ``"module_constant"``
when it still hardcodes that inventory locally (a recorded violation, not a
certification), or ``"n/a"`` when the fact is genuinely ambiguous between the
two (see ``twotower_prompt_ast`` below). VAR0-03 is the follow-up that
migrates ``"module_constant"`` entries to ``"pack"``; this module only
classifies and gates, it does not refactor (see the issue's explicit
non-goals).
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from slm_training.dsl.ops_vocab import OPS_VOCAB

VARIANT_CONTRACT_SCHEMA = "variant_contract/v1"
REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "resources" / "variant_registry.json"
)

# src/slm_training/dsl/variants.py -> parents[2] is "src".
_SRC_ROOT = Path(__file__).resolve().parents[2]

_TREE_EDIT_DIFFUSION_SOURCE = "slm_training/models/tree_edit_diffusion.py"
_OPERATOR_REGISTRY_SOURCE = "slm_training/dsl/operators/registry.py"
_TWOTOWER_SOURCE = "slm_training/models/twotower.py"

_INVENTORY_SOURCES = frozenset({"pack", "module_constant", "n/a"})


class VariantContractError(ValueError):
    """A variant contract is internally inconsistent or has drifted."""


class FactBucket(str, Enum):
    """Which authority owns a fact: shared kernel, shared pack, or one variant."""

    PACK_OWNED = "pack_owned"
    KERNEL_OWNED = "kernel_owned"
    VARIANT_OWNED = "variant_owned"


# The three-bucket classification named verbatim in VAR0-01's implementation
# contract. Total by construction (tested in test_variants.py): every fact a
# variant might hardcode falls into exactly one bucket.
FACT_CLASSIFICATION: Mapping[str, FactBucket] = {
    # pack-owned: must be read from dsl/pack.py (or dsl/packs/), never
    # hardcoded in a variant module.
    "component_inventory": FactBucket.PACK_OWNED,
    "component_property_names": FactBucket.PACK_OWNED,
    "component_property_value_domains": FactBucket.PACK_OWNED,
    "component_arity": FactBucket.PACK_OWNED,
    # kernel-owned: must come from slm_training.dsl.ops_vocab.OPS_VOCAB.
    "op_identity": FactBucket.KERNEL_OWNED,
    "op_family": FactBucket.KERNEL_OWNED,
    "op_token_ids": FactBucket.KERNEL_OWNED,
    # variant-owned: may legitimately stay local to one variant.
    "action_encoding": FactBucket.VARIANT_OWNED,
    "edit_budget": FactBucket.VARIANT_OWNED,
    "decode_loop": FactBucket.VARIANT_OWNED,
    "search_strategy": FactBucket.VARIANT_OWNED,
    "seed_selection": FactBucket.VARIANT_OWNED,
}


@dataclass(frozen=True)
class VariantContractV1:
    """Scopes one model variant: what it claims to share vs. own locally."""

    variant_id: str
    pack_id: str
    action_alphabet_id: str
    action_alphabet_fingerprint: str
    kernel_ops: tuple[str, ...]
    seed_policy_id: str
    inventory_source: str
    source_path: str
    schema: str = VARIANT_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != VARIANT_CONTRACT_SCHEMA:
            raise VariantContractError(
                f"{self.variant_id}: schema must be {VARIANT_CONTRACT_SCHEMA!r}, "
                f"got {self.schema!r}"
            )
        if self.inventory_source not in _INVENTORY_SOURCES:
            raise VariantContractError(
                f"{self.variant_id}: inventory_source must be one of "
                f"{sorted(_INVENTORY_SOURCES)}, got {self.inventory_source!r}"
            )
        known_ops = {symbol.op_id for symbol in OPS_VOCAB}
        unknown_kernel_ops = sorted(set(self.kernel_ops) - known_ops)
        if unknown_kernel_ops:
            raise VariantContractError(
                f"{self.variant_id} claims kernel_ops absent from OPS_VOCAB: "
                + ", ".join(unknown_kernel_ops)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "variant_id": self.variant_id,
            "pack_id": self.pack_id,
            "action_alphabet_id": self.action_alphabet_id,
            "action_alphabet_fingerprint": self.action_alphabet_fingerprint,
            "kernel_ops": list(self.kernel_ops),
            "seed_policy_id": self.seed_policy_id,
            "inventory_source": self.inventory_source,
            "source_path": self.source_path,
        }


def _read(relative: str) -> str:
    return (_SRC_ROOT / relative).read_text(encoding="utf-8")


def _module(relative: str) -> ast.Module:
    return ast.parse(_read(relative), filename=relative)


def _names_pack_authority(module_name: str, target: str) -> bool:
    """``module_name`` is ``target`` itself or one of its submodules -- never
    a substring match, so e.g. ``slm_training.dsl.packaging`` cannot be
    mistaken for ``slm_training.dsl.pack``/``slm_training.dsl.packs``."""
    return module_name == target or module_name.startswith(f"{target}.")


def _imports_module(relative: str, target: str) -> bool:
    """Static check: does the module at ``relative`` import ``target`` or a submodule of it?"""
    for node in ast.walk(_module(relative)):
        if isinstance(node, ast.ImportFrom) and node.module:
            if _names_pack_authority(node.module, target):
                return True
        if isinstance(node, ast.Import) and any(
            _names_pack_authority(alias.name, target) for alias in node.names
        ):
            return True
    return False


def _tree_edit_action_names() -> tuple[str, ...]:
    """Module-level ``ACTION_*`` constants: the tree-edit variant's live action alphabet.

    Parsed statically (never imported) because ``tree_edit_diffusion.py``
    imports torch, and this registry must stay importable in the
    dependency-light decode-invariants CI job.
    """
    names = [
        target.id
        for node in _module(_TREE_EDIT_DIFFUSION_SOURCE).body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id.startswith("ACTION_")
    ]
    if not names:
        raise VariantContractError(
            f"{_TREE_EDIT_DIFFUSION_SOURCE} no longer defines any ACTION_* constants"
        )
    return tuple(sorted(names))


def _repl_operator_ids() -> tuple[str, ...]:
    """The reserved-ops kernel already derives this variant's legal action ids."""
    ids = tuple(sorted({symbol.op_id for symbol in OPS_VOCAB}))
    if not ids:
        raise VariantContractError("OPS_VOCAB is empty; repl_operators has no action alphabet")
    return ids


def _structural_token_names() -> tuple[str, ...]:
    from slm_training.dsl.openui_tokens import STRUCTURAL_TOKENS

    return tuple(sorted(STRUCTURAL_TOKENS))


def _alphabet_fingerprint(ids: tuple[str, ...]) -> str:
    payload = json.dumps(sorted(ids), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_inventory_source(entry: VariantContractV1) -> None:
    """Fail closed on a false pack-authority claim (not on an honest violation).

    A variant may honestly record ``inventory_source="module_constant"`` (or
    ``"n/a"``) -- that is this issue's survey, not a certification. What must
    never happen is claiming ``"pack"`` while the source file does not even
    import the pack authority it claims to defer to.
    """
    if entry.inventory_source != "pack":
        return
    if not _imports_module(entry.source_path, "slm_training.dsl.pack"):
        raise VariantContractError(
            f"{entry.variant_id} claims inventory_source='pack' (pack_id="
            f"{entry.pack_id!r}) but {entry.source_path} does not import "
            "slm_training.dsl.pack"
        )


def build_variant_contracts() -> tuple[VariantContractV1, ...]:
    """Derive the three registered variant contracts from their live sources."""
    from slm_training.dsl.pack import list_packs

    entries = (
        VariantContractV1(
            variant_id="repl_operators",
            pack_id="openui",
            action_alphabet_id="repl_operators.legal_actions",
            action_alphabet_fingerprint=_alphabet_fingerprint(_repl_operator_ids()),
            kernel_ops=(),
            seed_policy_id="repl_operators.caller_supplied_state",
            inventory_source="pack",
            source_path=_OPERATOR_REGISTRY_SOURCE,
        ),
        VariantContractV1(
            variant_id="tree_edit_diffusion",
            pack_id="openui",
            action_alphabet_id="tree_edit_diffusion.edit_actions",
            action_alphabet_fingerprint=_alphabet_fingerprint(_tree_edit_action_names()),
            kernel_ops=(),
            seed_policy_id="tree_edit_diffusion.minimal_valid_program_seed",
            inventory_source="pack",
            source_path=_TREE_EDIT_DIFFUSION_SOURCE,
        ),
        VariantContractV1(
            variant_id="twotower_prompt_ast",
            pack_id="openui",
            action_alphabet_id="twotower_prompt_ast.structural_tokens",
            action_alphabet_fingerprint=_alphabet_fingerprint(_structural_token_names()),
            kernel_ops=(),
            seed_policy_id="twotower_prompt_ast.full_mask_seed",
            inventory_source="n/a",
            source_path=_TWOTOWER_SOURCE,
        ),
    )
    known_packs = set(list_packs())
    for entry in entries:
        if entry.pack_id not in known_packs:
            raise VariantContractError(
                f"{entry.variant_id} declares pack_id {entry.pack_id!r} which is "
                f"not a registered pack (known: {sorted(known_packs)})"
            )
        _validate_inventory_source(entry)
    return tuple(sorted(entries, key=lambda entry: entry.variant_id))


VARIANTS: tuple[VariantContractV1, ...] = build_variant_contracts()


def fingerprint() -> str:
    """Content address over the ordered variant registry."""
    payload = json.dumps(
        [entry.to_dict() for entry in VARIANTS],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest() -> dict[str, Any]:
    return {
        "schema": VARIANT_CONTRACT_SCHEMA,
        "count": len(VARIANTS),
        "fingerprint": fingerprint(),
        "variants": [entry.to_dict() for entry in VARIANTS],
    }


def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def write_registry() -> Path:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return REGISTRY_PATH


__all__ = [
    "FACT_CLASSIFICATION",
    "FactBucket",
    "REGISTRY_PATH",
    "VARIANT_CONTRACT_SCHEMA",
    "VARIANTS",
    "VariantContractError",
    "VariantContractV1",
    "build_variant_contracts",
    "fingerprint",
    "load_registry",
    "manifest",
    "write_registry",
]
