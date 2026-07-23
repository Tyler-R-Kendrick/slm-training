"""OpenUI language-contract identity.

Pins and hashes the exact language surface this repo targets so every dataset can
be stamped with a stable ``contract_id``. A change to the OpenUI package versions,
the grammar, the prop-order table, or the output tokenizers yields a new
``contract_id`` — i.e. a new dataset version. Because a component's positional
argument order is derived from its schema, silently changing the component library
can change the language the model accepts; binding datasets to ``contract_id``
makes that break loud instead of silent.

Scope note: the installed OpenUI Lang is the **0.2.x subset** (``@openuidev/lang-core``
tops out at 0.2.9). Full Lang v0.5 (state / queries / mutations / actions / tools)
has no published package yet; when it ships, extending the grammar / codec /
tokenizer is a contract *version bump* here, not a redesign.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

# Language spec the repo currently targets (see module docstring).
LANG_SPEC = "openui-lang-0.2.x"
# v4 is intentionally checkpoint-incompatible: output targets may contain only
# grammar/AST literals and opaque ordinal placeholder symbols, never
# open-vocabulary strings. Persisted train/eval records and their structured
# metadata must already use the harness-owned ``:slot_<ordinal>`` identities.
OUTPUT_CONTRACT_VERSION = 4
OUTPUT_CONTRACT_NAME = "symbol_only"

# src/slm_training/dsl/language_contract.py -> repo root is parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BRIDGE_PACKAGE_JSON = _REPO_ROOT / "src" / "apps" / "openui_bridge" / "package.json"
_GRAMMAR_FILES = (
    _REPO_ROOT / "src" / "slm_training" / "dsl" / "grammars" / "openui.lark",
    _REPO_ROOT
    / "src"
    / "slm_training"
    / "dsl"
    / "grammars"
    / "openui_prop_order.json",
)
# The official OpenUI packages whose versions define the language surface.
_OPENUI_PACKAGES = (
    "@openuidev/lang-core",
    "@openuidev/react-lang",
    "@openuidev/react-ui",
    "@openuidev/react-headless",
)


def _sha256_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_openui_versions(package_json: Path) -> dict[str, str]:
    data = json.loads(package_json.read_text(encoding="utf-8"))
    deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
    return {name: str(deps[name]) for name in _OPENUI_PACKAGES if name in deps}


@dataclass(frozen=True)
class LanguageContract:
    """Immutable identity of the OpenUI language surface a dataset targets."""

    lang_spec: str
    openui_versions: tuple[tuple[str, str], ...]
    grammar_sha256: str
    tokenizer_version: int
    dsl_tokenizer_version: int
    output_contract_version: int = OUTPUT_CONTRACT_VERSION

    @property
    def contract_id(self) -> str:
        """Stable 16-hex identity of this contract."""
        payload = json.dumps(
            {
                "lang_spec": self.lang_spec,
                "openui_versions": list(self.openui_versions),
                "grammar_sha256": self.grammar_sha256,
                "tokenizer_version": self.tokenizer_version,
                "dsl_tokenizer_version": self.dsl_tokenizer_version,
                "output_contract_version": self.output_contract_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        return {
            "lang_spec": self.lang_spec,
            "openui_versions": dict(self.openui_versions),
            "grammar_sha256": self.grammar_sha256,
            "tokenizer_version": self.tokenizer_version,
            "dsl_tokenizer_version": self.dsl_tokenizer_version,
            "output_contract_version": self.output_contract_version,
            "contract_id": self.contract_id,
        }


@lru_cache(maxsize=1)
def current_contract() -> LanguageContract:
    """Build the contract from the repo's pinned OpenUI surface (offline, deterministic)."""
    # Lazy imports keep this lightweight module free of the tokenizers' heavy deps
    # and any import cycle with ``slm_training.models``.
    from slm_training.models.dsl_tokenizer import DSL_TOKENIZER_VERSION
    from slm_training.models.tokenizer import TOKENIZER_VERSION

    versions = _read_openui_versions(_BRIDGE_PACKAGE_JSON)
    return LanguageContract(
        lang_spec=LANG_SPEC,
        openui_versions=tuple(sorted(versions.items())),
        grammar_sha256=_sha256_files(_GRAMMAR_FILES),
        tokenizer_version=int(TOKENIZER_VERSION),
        dsl_tokenizer_version=int(DSL_TOKENIZER_VERSION),
    )


def contract_id() -> str:
    """Stable 16-hex identity of the current language contract."""
    return current_contract().contract_id


class OutputContractError(ValueError):
    """An OpenUI target contains text outside the symbol-only language."""


STRUCTURAL_ID_ATOMS = frozenset(f"${index}" for index in range(64))


@lru_cache(maxsize=1)
def grammar_string_literals() -> frozenset[str]:
    """Closed string atoms declared by the pinned component schema."""
    from slm_training.dsl.lang_core import library_schema

    values: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            enum = value.get("enum")
            if isinstance(enum, list):
                values.update(item for item in enum if isinstance(item, str))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(library_schema())
    # Structural spellings accepted by the parser but not consistently
    # represented as machine-readable schema enums.
    values.update({"column", "row", "horizontal", "vertical"})
    # Opaque identifiers satisfy required schema string fields without making
    # the model reproduce user-facing names or borrowed English enum values.
    values.update(STRUCTURAL_ID_ATOMS)
    return frozenset(values)


def output_contract_violations(
    source: str, *, output_kind: str | None = None
) -> tuple[str, ...]:
    """Return free-form string values in an OpenUI program, fail closed."""
    from slm_training.dsl.placeholders import is_placeholder
    from slm_training.dsl.production_codec import (
        LIT_PREFIX,
        encode_output,
        parse_statement_bindings,
    )

    kinds = (
        (output_kind,)
        if output_kind is not None
        else ("document", "statement", "expression", "lexical", "typed_node")
    )
    for kind in kinds:
        try:
            program = encode_output(source, output_kind=str(kind))
        except Exception:  # noqa: BLE001 - try the remaining validated surfaces
            continue
        violations: list[str] = []
        for token in program.tokens:
            if not token.startswith(f'{LIT_PREFIX}"'):
                continue
            value = json.loads(token[len(LIT_PREFIX) :])
            if not is_placeholder(value) and value not in grammar_string_literals():
                violations.append(value)
        return tuple(dict.fromkeys(violations))

    # Official document validation rejects content literals before encoding;
    # inspect that repairable AST to report the contract violation itself.
    bindings = parse_statement_bindings(source, validate=False)
    allowed = grammar_string_literals()
    violations: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            if not is_placeholder(value) and value not in allowed:
                violations.append(value)
            return
        if isinstance(value, list):
            for child in value:
                walk(child)
            return
        if not isinstance(value, dict):
            return
        kind = value.get("type")
        if kind == "element":
            for child in (value.get("props") or {}).values():
                walk(child)
        elif kind == "call":
            for child in value.get("args") or ():
                walk(child)
        elif kind in {"array", "object", "literal"}:
            for key, child in value.items():
                if key not in {"type", "name", "typeName"}:
                    walk(child)

    for node in bindings.values():
        walk(node)
    return tuple(dict.fromkeys(violations))


def assert_symbol_only_output(source: str, *, output_kind: str | None = None) -> None:
    """Reject targets that would make the model predict free-form text."""
    violations = output_contract_violations(source, output_kind=output_kind)
    if violations:
        preview = ", ".join(repr(value) for value in violations[:3])
        raise OutputContractError(
            f"output contract {OUTPUT_CONTRACT_NAME}/v{OUTPUT_CONTRACT_VERSION} "
            f"forbids free-form strings: {preview}"
        )


def require_current_output_contract(payload: dict[str, Any]) -> None:
    """Reject every pre-symbol-only checkpoint, without migration guesses."""
    found = int(payload.get("output_contract_version", 0))
    if found != OUTPUT_CONTRACT_VERSION:
        raise OutputContractError(
            f"checkpoint output contract v{found} is incompatible with required "
            f"{OUTPUT_CONTRACT_NAME}/v{OUTPUT_CONTRACT_VERSION}; retrain from "
            "symbol-only targets"
        )
