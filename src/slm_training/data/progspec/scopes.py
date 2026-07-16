"""Split-safe scope contracts and deterministic ScopeDiff derivatives."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from slm_training.data.progspec.schema import ProgramSpec, emit_record
from slm_training.data.verify import VerificationContext, verify_record
from slm_training.dsl.parser import validate
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord


class ScopeKind(str, Enum):
    COMPONENT_CALL = "component_call"
    STATEMENT = "statement"
    CHILD_LIST = "child_list"


SCOPE_DATA_FAMILIES = (
    "reconstruction",
    "local_repair",
    "boundary_counterfactual",
    "local_valid_global_invalid",
    "heterogeneous_multi_scope",
)


@dataclass(frozen=True)
class ScopeContract:
    """Inherited interface plus synthesized targets for one stable AST scope."""

    scope_id: str
    root_id: str
    kind: ScopeKind
    ast_path: tuple[str | int, ...]
    source_span: tuple[int, int]
    expected_role: str
    parser_entry: str
    parser_exit: str
    visible_binders: tuple[str, ...]
    visible_slots: tuple[str, ...]
    parent_role: str
    sibling_index: int
    cardinality: int
    depth_budget: int
    width_budget: int
    token_budget: int
    definitions: tuple[str, ...]
    uses: tuple[str, ...]
    synthesized_slots: tuple[str, ...]
    realized_size: int

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["ast_path"] = list(self.ast_path)
        data["source_span"] = list(self.source_span)
        for name in (
            "visible_binders",
            "visible_slots",
            "definitions",
            "uses",
            "synthesized_slots",
        ):
            data[name] = list(data[name])
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScopeContract:
        return cls(
            scope_id=str(data["scope_id"]),
            root_id=str(data["root_id"]),
            kind=ScopeKind(str(data["kind"])),
            ast_path=tuple(data.get("ast_path") or ()),
            source_span=tuple(int(v) for v in data.get("source_span") or (0, 0)),
            expected_role=str(data["expected_role"]),
            parser_entry=str(data["parser_entry"]),
            parser_exit=str(data["parser_exit"]),
            visible_binders=tuple(str(v) for v in data.get("visible_binders") or ()),
            visible_slots=tuple(str(v) for v in data.get("visible_slots") or ()),
            parent_role=str(data.get("parent_role") or "document"),
            sibling_index=int(data.get("sibling_index") or 0),
            cardinality=int(data.get("cardinality") or 0),
            depth_budget=int(data.get("depth_budget") or 0),
            width_budget=int(data.get("width_budget") or 0),
            token_budget=int(data.get("token_budget") or 0),
            definitions=tuple(str(v) for v in data.get("definitions") or ()),
            uses=tuple(str(v) for v in data.get("uses") or ()),
            synthesized_slots=tuple(
                str(v) for v in data.get("synthesized_slots") or ()
            ),
            realized_size=int(data.get("realized_size") or 0),
        )


@dataclass(frozen=True)
class ScopeOracleResult:
    ok: bool
    local_gates: tuple[str, ...]
    failing_gate: str | None = None


def dependency_closed_failure_cone(
    paths: list[tuple[str | int, ...]],
) -> tuple[str | int, ...]:
    """Return the smallest common AST ancestor covering every failing path."""
    if not paths:
        return ()
    prefix = list(paths[0])
    for path in paths[1:]:
        prefix = prefix[
            : next(
                (i for i, pair in enumerate(zip(prefix, path)) if pair[0] != pair[1]),
                min(len(prefix), len(path)),
            )
        ]
    return tuple(prefix)


def validate_scope_wrapper(contract: ScopeContract, openui: str) -> ScopeOracleResult:
    """Conservatively validate a scope inside a complete OpenUI wrapper."""
    validate(openui)
    record = ExampleRecord(id="scope-oracle", prompt="scope oracle", openui=openui)
    report = verify_record(record, VerificationContext(source_kind="deterministic"))
    local = tuple(result.gate.value for result in report.results[:5] if result.ok)
    failed = next(
        (result.gate.value for result in report.results[:5] if not result.ok), None
    )
    _ = contract
    return ScopeOracleResult(ok=failed is None, local_gates=local, failing_gate=failed)


def derive_scope_contracts(spec: ProgramSpec) -> tuple[ScopeContract, ...]:
    """Extract v1 component, statement, and child-list scopes from ProgramSpec AST."""
    binders = tuple(
        str(value)
        for value in _collect_key(spec.ast, "statementId")
        if value and value != "root"
    )
    all_slots = tuple(sorted(extract_placeholders(spec.canonical_openui)))
    contracts: list[ScopeContract] = []

    def add(
        node: Any,
        *,
        kind: ScopeKind,
        path: tuple[str | int, ...],
        parent_role: str,
        sibling_index: int,
    ) -> None:
        role = (
            str(node.get("typeName") or node.get("statementId") or parent_role)
            if isinstance(node, dict)
            else parent_role
        )
        definitions = tuple(
            sorted(
                str(value)
                for value in _collect_key(node, "statementId")
                if value and value != "root"
            )
        )
        values = set(_collect_strings(node))
        uses = tuple(
            sorted(
                value
                for value in binders
                if value in values and value not in definitions
            )
        )
        slots = tuple(sorted(value for value in values if value.startswith(":")))
        path_text = "/".join(_pointer_part(part) for part in path) or "root"
        scope_id = f"{spec.id}:{kind.value}:{path_text}"
        parser_state = hashlib.sha256(
            f"{spec.contract_id}:{kind.value}:{role}".encode("utf-8")
        ).hexdigest()[:12]
        contracts.append(
            ScopeContract(
                scope_id=scope_id,
                root_id=spec.id,
                kind=kind,
                ast_path=path,
                source_span=_source_span(spec.canonical_openui, node),
                expected_role=role,
                parser_entry=f"entry:{parser_state}",
                parser_exit=f"exit:{parser_state}",
                visible_binders=tuple(sorted(set(binders) - set(definitions))),
                visible_slots=all_slots,
                parent_role=parent_role,
                sibling_index=sibling_index,
                cardinality=len(node) if isinstance(node, list) else len(definitions),
                depth_budget=max(1, 32 - len(path)),
                width_budget=max(1, len(node) if isinstance(node, list) else 1),
                token_budget=max(8, len(json.dumps(node, sort_keys=True)) // 3),
                definitions=definitions,
                uses=uses,
                synthesized_slots=slots,
                realized_size=_node_count(node),
            )
        )

    def visit(value: Any, path: tuple[str | int, ...], parent_role: str) -> None:
        if isinstance(value, dict):
            role = str(value.get("typeName") or parent_role)
            if value.get("type") == "element":
                add(
                    value,
                    kind=ScopeKind.COMPONENT_CALL,
                    path=path,
                    parent_role=parent_role,
                    sibling_index=int(path[-1])
                    if path and isinstance(path[-1], int)
                    else 0,
                )
            if value.get("statementId"):
                add(
                    value,
                    kind=ScopeKind.STATEMENT,
                    path=path,
                    parent_role=parent_role,
                    sibling_index=int(path[-1])
                    if path and isinstance(path[-1], int)
                    else 0,
                )
            for key, child in value.items():
                child_path = (*path, str(key))
                if key == "children" and isinstance(child, list):
                    add(
                        child,
                        kind=ScopeKind.CHILD_LIST,
                        path=child_path,
                        parent_role=role,
                        sibling_index=0,
                    )
                visit(child, child_path, role)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, index), parent_role)

    visit(spec.ast, (), "document")
    return tuple(sorted(contracts, key=lambda item: item.scope_id))


def derive_scope_records(spec: ProgramSpec) -> tuple[ExampleRecord, ...]:
    """Project a balanced v1 ScopeDiff corpus without crossing split groups."""
    by_kind: dict[ScopeKind, ScopeContract] = {}
    for contract in derive_scope_contracts(spec):
        by_kind.setdefault(contract.kind, contract)
    contracts = tuple(by_kind[kind] for kind in ScopeKind if kind in by_kind)
    records: list[ExampleRecord] = []
    for contract in contracts:
        for family in SCOPE_DATA_FAMILIES:
            contract_data = contract.to_dict()
            gate_target = (
                0.0
                if family
                in {
                    "boundary_counterfactual",
                    "local_valid_global_invalid",
                }
                else 1.0
            )
            if family == "boundary_counterfactual":
                contract_data["expected_role"] = "__invalid_boundary__"
            cone = list(contract.ast_path) if gate_target == 0.0 else []
            prompt = (
                f"ScopeDiff {family}: satisfy this scope contract and emit the verified "
                f"OpenUI wrapper.\n{json.dumps(contract_data, sort_keys=True)}"
            )
            task = (
                "repair" if "repair" in family or gate_target == 0.0 else "generation"
            )
            record = emit_record(
                spec,
                prompt=prompt,
                task=task,
                record_id=f"{hashlib.sha256(f'{contract.scope_id}:{family}'.encode()).hexdigest()[:16]}_scope",
                source="scope_contract",
                meta={
                    "scope_contract": contract_data,
                    "scope_contract_fingerprint": contract.fingerprint,
                    "scope_family": family,
                    "scope_gate_target": gate_target,
                    "failure_cone": cone,
                    "source_kind": "deterministic",
                },
            )
            records.append(record)
    return tuple(records)


def _pointer_part(value: str | int) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _collect_key(value: Any, key: str) -> list[Any]:
    out: list[Any] = []
    if isinstance(value, dict):
        if key in value:
            out.append(value[key])
        for child in value.values():
            out.extend(_collect_key(child, key))
    elif isinstance(value, list):
        for child in value:
            out.extend(_collect_key(child, key))
    return out


def _collect_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _collect_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _collect_strings(child)]
    return []


def _node_count(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + sum(_node_count(child) for child in value.values())
    if isinstance(value, list):
        return 1 + sum(_node_count(child) for child in value)
    return 1


def _source_span(openui: str, node: Any) -> tuple[int, int]:
    statement = node.get("statementId") if isinstance(node, dict) else None
    if statement:
        offset = 0
        for line in openui.splitlines(keepends=True):
            if line.startswith(f"{statement} ="):
                return (offset, offset + len(line.rstrip("\n")))
            offset += len(line)
    return (0, len(openui))


__all__ = [
    "SCOPE_DATA_FAMILIES",
    "ScopeContract",
    "ScopeKind",
    "ScopeOracleResult",
    "dependency_closed_failure_cone",
    "derive_scope_contracts",
    "derive_scope_records",
    "validate_scope_wrapper",
]
