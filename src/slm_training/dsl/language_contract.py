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

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from slm_training.data.contract import GenerationRequest, RuntimeSymbol

# Language spec the repo currently targets (see module docstring).
LANG_SPEC = "openui-lang-0.2.x"
# v2 is intentionally checkpoint-incompatible: output targets may contain only
# grammar/AST literals and placeholder symbols, never open-vocabulary strings.
OUTPUT_CONTRACT_VERSION = 2
OUTPUT_CONTRACT_NAME = "symbol_only"
SYMBOLIC_SURFACE_POLICY_VERSION = "symbolic_surface_policy/v1"

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


class SurfaceCategory(str, Enum):
    GRAMMAR = "grammar_keyword_or_punctuation"
    CLOSED_VALUE = "closed_enum_or_primitive"
    OPEN_STRING = "open_string"
    OPEN_NUMBER = "open_number"
    BINDER = "binder"
    EXTERNAL_REF = "external_ref"
    STATE_REF = "state_ref"
    COMMENT_PROSE = "comment_or_prose"
    UNDECLARED_IDENTIFIER = "undeclared_identifier"


class SurfaceDecision(str, Enum):
    TEMPLATE = "template"
    REJECT = "reject"


@dataclass(frozen=True)
class SymbolicSurfaceViolationV1:
    start: int
    end: int
    surface: str
    category: SurfaceCategory
    pack_id: str
    pack_version: str
    policy_version: str
    decision: SurfaceDecision
    suggested_marker_role: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "span": {"start": self.start, "end": self.end},
            "surface": self.surface,
            "category": self.category.value,
            "pack_id": self.pack_id,
            "pack_version": self.pack_version,
            "policy_version": self.policy_version,
            "decision": self.decision.value,
            "suggested_marker_role": self.suggested_marker_role,
        }


@dataclass(frozen=True)
class SymbolicSurfaceReportV1:
    pack_id: str
    pack_version: str
    policy_version: str
    source_sha256: str
    violations: tuple[SymbolicSurfaceViolationV1, ...]

    @property
    def admitted(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        return {
            "pack_id": self.pack_id,
            "pack_version": self.pack_version,
            "policy_version": self.policy_version,
            "source_sha256": self.source_sha256,
            "admitted": self.admitted,
            "violations": [violation.to_dict() for violation in self.violations],
        }


_SURFACE_RE = re.compile(
    r"""
    (?P<comment>//[^\n]*|\#[^\n]*)
  | (?P<string>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')
  | (?P<state>\$[A-Za-z_][A-Za-z0-9_]*)
  | (?P<at>@[A-Z][A-Za-z0-9_]*)
  | (?P<upper>[A-Z][A-Za-z0-9_]*)
  | (?P<lower>[a-z_][A-Za-z0-9_]*)
  | (?P<number>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<punct>==|!=|>=|<=|&&|\|\||=|\(|\)|\[|\]|\{|\}|,|\.|:|\?|\+|-|\*|/|%|!|>|<)
  | (?P<space>\s+)
  | (?P<other>.)
    """,
    re.VERBOSE,
)


def _pack_surface_authority(pack: Any) -> tuple[frozenset[str], frozenset[str], str]:
    structural = set(pack.backend.structural_tokens())
    structural.update(pack.backend.component_names())
    closed_values: set[str] = {"true", "false", "null"}
    closed_identifiers: set[str] = set(structural)
    if pack.backend.info.kind == "graphql-js":
        structural.update({"!", "[", "]"})
        schema_path = pack.backend.info.grammar_path
        if schema_path is not None and schema_path.is_file():
            closed_identifiers.update(
                re.findall(r"[A-Za-z_][A-Za-z0-9_]*", schema_path.read_text())
            )
        closed_identifiers.update(structural)
    schema = pack.backend.library_schema()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            enum = value.get("enum")
            if isinstance(enum, list):
                closed_values.update(item for item in enum if isinstance(item, str))
            properties = value.get("properties")
            if isinstance(properties, dict):
                closed_identifiers.update(str(item) for item in properties)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)
    payload = json.dumps(
        {
            "pack_id": pack.pack_id,
            "backend_id": pack.backend.info.id,
            "structural": sorted(structural),
            "closed_values": sorted(closed_values),
            "closed_identifiers": sorted(closed_identifiers),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        frozenset(closed_values),
        frozenset(closed_identifiers),
        hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True)
class SymbolicSurfacePolicyV1:
    """Fail-closed staged-target policy over pack and request authority."""

    pack_id: str = "openui"
    policy_version: str = SYMBOLIC_SURFACE_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.policy_version != SYMBOLIC_SURFACE_POLICY_VERSION:
            raise ValueError(
                f"unsupported symbolic surface policy {self.policy_version!r}"
            )

    def evaluate(
        self,
        source: str,
        *,
        runtime_symbols: Iterable[RuntimeSymbol] = (),
    ) -> SymbolicSurfaceReportV1:
        from slm_training.dsl.pack import get_pack

        pack = get_pack(self.pack_id)
        closed_values, closed_identifiers, pack_version = _pack_surface_authority(
            pack
        )
        symbol_rows = tuple(runtime_symbols)
        symbols = {symbol.surface: symbol for symbol in symbol_rows}
        if len(symbols) != len(symbol_rows):
            raise ValueError("runtime symbol surfaces must be unique")
        matches = [
            match
            for match in _SURFACE_RE.finditer(source)
            if match.lastgroup != "space"
        ]
        binders = {
            match.group(0)
            for index, match in enumerate(matches[:-1])
            if match.lastgroup == "lower"
            and matches[index + 1].lastgroup == "punct"
            and matches[index + 1].group(0) == "="
        }
        binders.update(
            match.group(0)
            for index, match in enumerate(matches)
            if index
            and matches[index - 1].group(0) in {"query", "mutation", "fragment"}
            and match.lastgroup in {"lower", "upper"}
        )
        declared_binders = {
            surface
            for surface, symbol in symbols.items()
            if symbol.role in {"alpha_binder", "fresh_binder"}
        }
        violations: list[SymbolicSurfaceViolationV1] = []

        def reject(
            match: re.Match[str],
            category: SurfaceCategory,
            decision: SurfaceDecision,
            role: str | None = None,
        ) -> None:
            violations.append(
                SymbolicSurfaceViolationV1(
                    start=match.start(),
                    end=match.end(),
                    surface=match.group(0),
                    category=category,
                    pack_id=pack.pack_id,
                    pack_version=pack_version,
                    policy_version=self.policy_version,
                    decision=decision,
                    suggested_marker_role=role,
                )
            )

        for match in matches:
            group = match.lastgroup
            surface = match.group(0)
            if group == "comment":
                reject(
                    match,
                    SurfaceCategory.COMMENT_PROSE,
                    SurfaceDecision.REJECT,
                )
            elif group == "string":
                try:
                    value = ast.literal_eval(surface)
                except (SyntaxError, ValueError):
                    reject(
                        match,
                        SurfaceCategory.OPEN_STRING,
                        SurfaceDecision.REJECT,
                    )
                    continue
                if pack.placeholder_policy.is_placeholder(value):
                    symbol = symbols.get(value)
                    if symbol is None or symbol.role != "external_entity":
                        reject(
                            match,
                            SurfaceCategory.EXTERNAL_REF,
                            SurfaceDecision.REJECT,
                            "external_entity",
                        )
                elif value not in closed_values:
                    can_template_string = bool(pack.placeholder_policy.content_props)
                    reject(
                        match,
                        SurfaceCategory.OPEN_STRING,
                        (
                            SurfaceDecision.TEMPLATE
                            if can_template_string
                            else SurfaceDecision.REJECT
                        ),
                        "external_entity" if can_template_string else None,
                    )
            elif group == "number":
                reject(
                    match,
                    SurfaceCategory.OPEN_NUMBER,
                    SurfaceDecision.REJECT,
                )
            elif group == "state":
                symbol = symbols.get(surface)
                if symbol is None or symbol.role != "state":
                    reject(
                        match,
                        SurfaceCategory.STATE_REF,
                        SurfaceDecision.REJECT,
                        "state",
                    )
            elif group == "lower":
                if (
                    surface not in binders
                    and surface not in declared_binders
                    and surface not in closed_values
                    and surface not in closed_identifiers
                ):
                    reject(
                        match,
                        SurfaceCategory.UNDECLARED_IDENTIFIER,
                        SurfaceDecision.REJECT,
                        "alpha_binder",
                    )
            elif group in {"upper", "at", "punct"}:
                if surface not in binders and surface not in closed_identifiers:
                    reject(
                        match,
                        SurfaceCategory.UNDECLARED_IDENTIFIER,
                        SurfaceDecision.REJECT,
                    )
            elif group == "other":
                reject(
                    match,
                    SurfaceCategory.COMMENT_PROSE,
                    SurfaceDecision.REJECT,
                )

        return SymbolicSurfaceReportV1(
            pack_id=pack.pack_id,
            pack_version=pack_version,
            policy_version=self.policy_version,
            source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            violations=tuple(violations),
        )

    def require_admitted(
        self,
        source: str,
        *,
        runtime_symbols: Iterable[RuntimeSymbol] = (),
    ) -> SymbolicSurfaceReportV1:
        report = self.evaluate(source, runtime_symbols=runtime_symbols)
        if report.violations:
            preview = ", ".join(
                f"{item.category.value}:{item.surface!r}"
                for item in report.violations[:3]
            )
            raise OutputContractError(
                f"{self.policy_version} rejected staged target: {preview}"
            )
        return report

    def evaluate_request(
        self, source: str, request: GenerationRequest
    ) -> SymbolicSurfaceReportV1:
        """Evaluate against the request's explicit and legacy slot authority."""

        return self.evaluate(
            source,
            runtime_symbols=request.effective_runtime_symbols(),
        )


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
