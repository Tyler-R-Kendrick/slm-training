"""Closed, versioned Harness DSL for symbolic CAP0 task intent."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from slm_training.data.contract import RuntimeSymbol
from slm_training.dsl.language_contract import SymbolicSurfacePolicyV1
from slm_training.dsl.pack import DslPack, get_pack

HARNESS_SCHEMA = "harness_dsl/v1"
HARNESS_VERSION = "HARNESS_V1"
_GRAMMAR_PATH = Path(__file__).with_name("grammars") / "harness.lark"
_PAYLOAD_BEGIN = "PAYLOAD_BEGIN\n"
_PAYLOAD_END = "\nPAYLOAD_END"
_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_MARKER_RE = re.compile(r"[^\s]+\Z")


class HarnessDslError(ValueError):
    """A Harness prompt failed before it could become model input."""


class HarnessOperation(str, Enum):
    IDENTITY = "IDENTITY"
    CANONICALIZE = "CANONICALIZE"
    COMPLETE_SUFFIX = "COMPLETE_SUFFIX"
    COMPOSE = "COMPOSE"


class HarnessPayloadKind(str, Enum):
    DOCUMENT = "document"
    STATEMENT = "statement"
    EXPRESSION = "expression"
    LEXICAL = "lexical"
    NODE = "node"


_OUTPUT_KIND = {
    HarnessPayloadKind.DOCUMENT: "document",
    HarnessPayloadKind.STATEMENT: "statement",
    HarnessPayloadKind.EXPRESSION: "expression",
    HarnessPayloadKind.LEXICAL: "lexical",
    HarnessPayloadKind.NODE: "typed_node",
}
_ROLES = frozenset({"alpha_binder", "external_entity", "state", "fresh_binder"})


def harness_grammar_fingerprint() -> str:
    """Fingerprint the exact checked-in Harness grammar authority."""
    payload = f"{HARNESS_SCHEMA}\n{_GRAMMAR_PATH.read_text(encoding='utf-8')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HarnessTaskV1:
    operation: HarnessOperation
    pack_id: str
    payload_kind: HarnessPayloadKind
    grammar_category: str | None
    payload: str
    artifact_refs: tuple[str, ...] = ()
    runtime_symbols: tuple[RuntimeSymbol, ...] = ()
    schema: str = HARNESS_SCHEMA
    grammar_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.schema != HARNESS_SCHEMA:
            raise HarnessDslError(f"unsupported Harness schema {self.schema!r}")
        if not self.grammar_fingerprint:
            object.__setattr__(
                self, "grammar_fingerprint", harness_grammar_fingerprint()
            )
        elif self.grammar_fingerprint != harness_grammar_fingerprint():
            raise HarnessDslError("Harness grammar fingerprint mismatch")
        if not _SYMBOL_RE.fullmatch(self.pack_id):
            raise HarnessDslError(f"invalid pack symbol {self.pack_id!r}")
        if self.grammar_category is not None and not _SYMBOL_RE.fullmatch(
            self.grammar_category
        ):
            raise HarnessDslError(f"invalid grammar category {self.grammar_category!r}")
        if not self.payload:
            raise HarnessDslError("Harness payload must be non-empty")
        if _PAYLOAD_END in self.payload or _PAYLOAD_BEGIN in self.payload:
            raise HarnessDslError("Harness payload contains a reserved delimiter")
        if len(set(self.artifact_refs)) != len(self.artifact_refs):
            raise HarnessDslError("artifact refs must be unique")
        for ref in self.artifact_refs:
            if not _DIGEST_RE.fullmatch(ref):
                raise HarnessDslError(f"invalid artifact ref {ref!r}")
        surfaces = [symbol.surface for symbol in self.runtime_symbols]
        if len(set(surfaces)) != len(surfaces):
            raise HarnessDslError("runtime marker surfaces must be unique")
        for symbol in self.runtime_symbols:
            if symbol.role not in _ROLES or not _MARKER_RE.fullmatch(symbol.surface):
                raise HarnessDslError(f"invalid runtime marker {symbol.surface!r}")


def _declared_categories(pack: DslPack) -> frozenset[str]:
    authority = pack.grammar_capability_authority
    if authority is None:
        return frozenset()
    names = {str(item.name) for item in (authority.terminal_categories or ())}
    names.update(str(item.lhs) for item in (authority.productions or ()))
    names.update(str(item) for item in (authority.start_symbols or ()))
    names.update(item.value for item in HarnessPayloadKind)
    grammar_path = pack.backend.info.grammar_path
    if grammar_path is not None and grammar_path.is_file():
        names.update(
            re.findall(
                r"[A-Za-z_][A-Za-z0-9_]*",
                grammar_path.read_text(encoding="utf-8"),
            )
        )
    return frozenset(names)


def validate_harness_task(
    task: HarnessTaskV1, *, pack: DslPack | None = None
) -> HarnessTaskV1:
    """Validate the outer grammar, pack identity, fragment, and symbol policy."""
    resolved = pack or get_pack(task.pack_id)
    if resolved.pack_id != task.pack_id:
        raise HarnessDslError(
            f"pack identity mismatch: {task.pack_id!r} != {resolved.pack_id!r}"
        )
    if task.grammar_category is not None:
        declared = _declared_categories(resolved)
        if declared and task.grammar_category not in declared:
            raise HarnessDslError(
                f"pack {task.pack_id!r} does not declare grammar category "
                f"{task.grammar_category!r}"
            )
    fragment_parser = resolved.require("fragment_parser")
    try:
        fragment_parser(
            task.payload,
            _OUTPUT_KIND[task.payload_kind],
            task.grammar_category,
        )
        SymbolicSurfacePolicyV1(task.pack_id).require_admitted(
            task.payload,
            runtime_symbols=task.runtime_symbols,
        )
    except Exception as exc:  # noqa: BLE001 - normalize every pack failure
        raise HarnessDslError(
            f"invalid {task.pack_id} {task.payload_kind.value} payload: {exc}"
        ) from exc
    return task


def serialize_harness_task(task: HarnessTaskV1) -> str:
    """Emit one canonical, closed Harness prompt."""
    validate_harness_task(task)
    lines = [
        HARNESS_VERSION,
        f"OP {task.operation.value}",
        f"PACK {task.pack_id}",
        f"TYPE {task.payload_kind.value}",
        f"CATEGORY {task.grammar_category or '-'}",
    ]
    lines.extend(f"ARTIFACT {ref}" for ref in sorted(task.artifact_refs))
    lines.extend(
        f"MARKER {symbol.role} {symbol.surface}"
        for symbol in sorted(
            task.runtime_symbols, key=lambda item: (item.role, item.surface)
        )
    )
    lines.append("PAYLOAD_BEGIN")
    return "\n".join(lines) + "\n" + task.payload + _PAYLOAD_END


def parse_harness_task(source: str, *, pack: DslPack | None = None) -> HarnessTaskV1:
    """Parse the framing independently, then validate via the selected pack."""
    if "\r" in source or not source.startswith(f"{HARNESS_VERSION}\n"):
        raise HarnessDslError("invalid Harness version or newline framing")
    if source.count(_PAYLOAD_BEGIN) != 1 or source.count(_PAYLOAD_END) != 1:
        raise HarnessDslError("Harness prompt must contain exactly one payload")
    header, payload = source.split(_PAYLOAD_BEGIN, 1)
    payload, suffix = payload.rsplit(_PAYLOAD_END, 1)
    if suffix:
        raise HarnessDslError("trailing text after Harness payload")
    lines = header.removesuffix("\n").splitlines()
    if len(lines) < 5 or lines[0] != HARNESS_VERSION:
        raise HarnessDslError("incomplete Harness header")

    def field(index: int, name: str) -> str:
        prefix = f"{name} "
        line = lines[index]
        if not line.startswith(prefix) or not line[len(prefix) :]:
            raise HarnessDslError(f"expected one {name} field")
        return line[len(prefix) :]

    try:
        operation = HarnessOperation(field(1, "OP"))
        pack_id = field(2, "PACK")
        payload_kind = HarnessPayloadKind(field(3, "TYPE"))
        category_value = field(4, "CATEGORY")
    except (IndexError, ValueError) as exc:
        raise HarnessDslError(f"invalid reserved Harness field: {exc}") from exc
    refs: list[str] = []
    markers: list[RuntimeSymbol] = []
    for line in lines[5:]:
        if line.startswith("ARTIFACT "):
            refs.append(line.removeprefix("ARTIFACT "))
            continue
        if line.startswith("MARKER "):
            parts = line.split(" ")
            if len(parts) != 3:
                raise HarnessDslError("MARKER requires exactly role and surface")
            try:
                markers.append(RuntimeSymbol(surface=parts[2], role=parts[1]))
            except ValueError as exc:
                raise HarnessDslError(str(exc)) from exc
            continue
        raise HarnessDslError(f"unknown Harness header field {line!r}")
    task = HarnessTaskV1(
        operation=operation,
        pack_id=pack_id,
        payload_kind=payload_kind,
        grammar_category=None if category_value == "-" else category_value,
        payload=payload,
        artifact_refs=tuple(refs),
        runtime_symbols=tuple(markers),
    )
    validate_harness_task(task, pack=pack)
    if serialize_harness_task(task) != source:
        raise HarnessDslError("non-canonical Harness serialization")
    return task


def is_harness_prompt(source: str) -> bool:
    return source.startswith(f"{HARNESS_VERSION}\n")


def runtime_symbols_for_payload(payload: str) -> tuple[RuntimeSymbol, ...]:
    """Declare request-visible external/state markers found in a fragment."""
    from slm_training.dsl.placeholders import extract_placeholders

    surface_without_strings = re.sub(
        r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        "",
        payload,
    )
    symbols = [
        RuntimeSymbol(surface=value, role="external_entity")
        for value in sorted(set(extract_placeholders(payload)))
    ]
    symbols.extend(
        RuntimeSymbol(surface=value, role="state")
        for value in sorted(set(re.findall(r"\$[A-Za-z_][A-Za-z0-9_]*", payload)))
    )
    symbols.extend(
        RuntimeSymbol(surface=value, role="alpha_binder")
        for value in sorted(
            set(re.findall(r"\b[a-z_][A-Za-z0-9_]*\b", surface_without_strings))
            - {"false", "fragment", "mutation", "null", "query", "true"}
        )
    )
    return tuple(symbols)


__all__ = [
    "HARNESS_SCHEMA",
    "HARNESS_VERSION",
    "HarnessDslError",
    "HarnessOperation",
    "HarnessPayloadKind",
    "HarnessTaskV1",
    "harness_grammar_fingerprint",
    "is_harness_prompt",
    "parse_harness_task",
    "runtime_symbols_for_payload",
    "serialize_harness_task",
    "validate_harness_task",
]
