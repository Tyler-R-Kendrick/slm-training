"""Production codec: OpenUI ↔ compact grammar-native token sequence + slot pointers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from slm_training.data.contract import canonical_slot_contract
from slm_training.data.structure import strip_style_literals
from slm_training.dsl.lang_core import ParseError, Program
from slm_training.dsl.placeholders import is_placeholder
from slm_training.grammar_backends.ast_utils import map_positional_props
from slm_training.grammar_backends.types import GRAMMARS_DIR

OPEN_PREFIX = "+"
CLOSE = "-"
DIR_PREFIX = "^"
SLOT_PREFIX = "@"
REF_PREFIX = "&"
LIT_PREFIX = "#"
LIST_OPEN = "["
LIST_CLOSE = "]"
STMT = "="

_DIRECTIONS = frozenset({"column", "row"})
_STMT_RE = re.compile(r"(?m)^([a-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")


@dataclass(frozen=True)
class ProductionProgram:
    """Compact production token sequence with parallel slot contract."""

    tokens: tuple[str, ...]
    slot_contract: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": list(self.tokens),
            "slot_contract": list(self.slot_contract),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProductionProgram:
        return cls(
            tokens=tuple(data.get("tokens") or ()),
            slot_contract=tuple(data.get("slot_contract") or ()),
        )


@dataclass
class ProductionVocab:
    """Grammar-closed vocabulary built from a production-token corpus."""

    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"
    _SPECIAL = (PAD, BOS, EOS, UNK)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @classmethod
    def build(cls, programs: Iterable[ProductionProgram | Iterable[str]]) -> ProductionVocab:
        vocab = list(cls._SPECIAL)
        seen = set(vocab)
        for item in programs:
            tokens = item.tokens if isinstance(item, ProductionProgram) else tuple(item)
            for tok in tokens:
                if tok not in seen:
                    seen.add(tok)
                    vocab.append(tok)
        token_to_id = {t: i for i, t in enumerate(vocab)}
        return cls(token_to_id=token_to_id, id_to_token={i: t for t, i in token_to_id.items()})

    def encode(self, tokens: Iterable[str], *, add_special: bool = True) -> list[int]:
        unk = self.token_to_id[self.UNK]
        ids = [self.token_to_id.get(t, unk) for t in tokens]
        if add_special:
            return [self.token_to_id[self.BOS], *ids, self.token_to_id[self.EOS]]
        return ids

    def decode_ids(self, ids: Iterable[int], *, skip_special: bool = True) -> list[str]:
        special = {
            self.token_to_id[self.PAD],
            self.token_to_id[self.BOS],
            self.token_to_id[self.EOS],
        }
        out: list[str] = []
        for i in ids:
            if skip_special and i in special:
                continue
            out.append(self.id_to_token.get(i, self.UNK))
        return out


def _prop_order() -> dict[str, list[str]]:
    path = GRAMMARS_DIR / "openui_prop_order.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_program(source: str) -> Program:
    from slm_training.grammar_backends import get_backend

    text = strip_style_literals(source or "").strip()
    if not text:
        raise ParseError("empty OpenUI source")
    return get_backend("openui").validate(text)


def _collect_bindings(source: str) -> dict[str, Any]:
    program = _parse_program(source)
    bindings = (program.meta or {}).get("bindings")
    if isinstance(bindings, list):
        names = list(bindings)
        return {name: None for name in names}
    root = program.root
    if not isinstance(root, dict):
        raise ParseError("missing root AST")
    names: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            sid = node.get("statementId")
            if isinstance(sid, str) and sid not in names:
                names.append(sid)
            props = node.get("props")
            if isinstance(props, dict):
                for value in props.values():
                    walk(value)
            for key in ("children", "root"):
                child = node.get(key)
                if isinstance(child, list):
                    for item in child:
                        walk(item)
                elif child is not None:
                    walk(child)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(root)
    for match in _STMT_RE.finditer(source):
        name = match.group(1)
        if name not in names:
            names.append(name)
    if "root" not in names:
        names.insert(0, "root")
    return {name: None for name in names}


def _parse_bindings(source: str) -> dict[str, Any]:
    """Return resolved element AST nodes keyed by binder name."""
    from slm_training.grammar_backends import get_backend

    backend = get_backend("openui")
    program = backend.validate(source)
    bindings: dict[str, Any] = {}

    for match in _STMT_RE.finditer(source):
        name = match.group(1)
        rhs = match.group(2).strip()
        probe = f"__probe_{name}__"
        mini = f"{probe} = {rhs}\nroot = {probe}\n"
        mini_program = backend.validate(mini)
        root = mini_program.root
        if isinstance(root, dict) and root.get("type") == "element":
            bindings[name] = root
            continue
        if isinstance(root, dict) and root.get("type") == "ref":
            target = str(root["name"])
            if target in bindings:
                bindings[name] = bindings[target]
            else:
                raise ParseError(f"unresolved binding {name!r}")
            continue
        bindings[name] = root

    if "root" not in bindings and program.root is not None:
        bindings["root"] = program.root
    return bindings


def _statement_order(source: str, bindings: dict[str, Any]) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()

    def visit_ref(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        match = re.search(
            rf"(?m)^{re.escape(name)}\s*=\s*(.+?)\s*$",
            source,
        )
        if not match:
            return
        rhs = match.group(1)
        for ref in re.findall(r"\b([a-z_][A-Za-z0-9_]*)\b", rhs):
            if ref in {"true", "false", "null"}:
                continue
            if re.search(rf"(?m)^{re.escape(ref)}\s*=", source):
                visit_ref(ref)
        order.append(name)

    visit_ref("root")
    for match in _STMT_RE.finditer(source):
        name = match.group(1)
        if name not in seen:
            visit_ref(name)
    return order


def _resolve_binding(bindings: dict[str, Any], name: str) -> Any:
    if name not in bindings:
        raise ParseError(f"undefined binding {name!r}")
    return bindings[name]


def encode_openui(
    source: str,
    *,
    slot_contract: Iterable[str] | None = None,
) -> ProductionProgram:
    """Parse OpenUI and emit a compact production token sequence."""
    scrubbed = strip_style_literals(source or "").strip()
    _parse_program(scrubbed)
    bindings = _parse_bindings(scrubbed)
    contract = (
        canonical_slot_contract(scrubbed, declared=slot_contract)
        if slot_contract is not None
        else canonical_slot_contract(scrubbed)
    )
    slot_index = {ph: i for i, ph in enumerate(contract)}
    stmt_order = _statement_order(scrubbed, bindings)
    stmt_index = {name: i for i, name in enumerate(stmt_order)}

    tokens: list[str] = []
    for name in stmt_order:
        tokens.append(STMT)
        expr = _resolve_binding(bindings, name)
        tokens.extend(
            _encode_expr(expr, slot_index=slot_index, stmt_index=stmt_index)
        )
    return ProductionProgram(tokens=tuple(tokens), slot_contract=tuple(contract))


def decode_productions(
    tokens: Iterable[str],
    slot_contract: Iterable[str],
    *,
    root_name: str = "root",
) -> str:
    """Reconstruct deterministic OpenUI source from production tokens + contract."""
    contract = tuple(slot_contract)
    stream = list(tokens)
    pos = 0
    statements: list[tuple[str, str]] = []
    generated_names: list[str] = []
    stmt_count = sum(1 for tok in stream if tok == STMT)

    def peek() -> str | None:
        return stream[pos] if pos < len(stream) else None

    def pop() -> str:
        nonlocal pos
        if pos >= len(stream):
            raise ParseError("unexpected end of production stream")
        tok = stream[pos]
        pos += 1
        return tok

    def decode_expr() -> str:
        tok = pop()
        if tok == LIST_OPEN:
            parts: list[str] = []
            while peek() != LIST_CLOSE:
                parts.append(decode_expr())
            pop()
            return "[" + ", ".join(parts) + "]"
        if tok.startswith(OPEN_PREFIX):
            comp = tok[len(OPEN_PREFIX) :]
            args: list[str] = []
            while peek() != CLOSE:
                args.append(decode_expr())
            pop()
            return f"{comp}({', '.join(args)})"
        if tok.startswith(SLOT_PREFIX):
            idx = int(tok[len(SLOT_PREFIX) :])
            if idx < 0 or idx >= len(contract):
                raise ParseError(f"slot pointer out of range: {tok}")
            return json.dumps(contract[idx])
        if tok.startswith(DIR_PREFIX):
            return json.dumps(tok[len(DIR_PREFIX) :])
        if tok.startswith(REF_PREFIX):
            idx = int(tok[len(REF_PREFIX) :])
            if idx < 0 or idx >= len(generated_names):
                raise ParseError(f"statement ref out of range: {tok}")
            return generated_names[idx]
        if tok.startswith(LIT_PREFIX):
            return _decode_literal(tok[len(LIT_PREFIX) :])
        raise ParseError(f"unknown production token: {tok}")

    while pos < len(stream):
        if pop() != STMT:
            raise ParseError("expected statement marker '='")
        stmt_idx = len(generated_names)
        binder = root_name if stmt_idx == stmt_count - 1 else f"v{stmt_idx}"
        expr = decode_expr()
        generated_names.append(binder)
        statements.append((binder, expr))

    if pos != len(stream):
        raise ParseError("trailing production tokens")

    root_lines = [f"{name} = {expr}" for name, expr in statements if name == root_name]
    other_lines = [f"{name} = {expr}" for name, expr in statements if name != root_name]
    return "\n".join(root_lines + other_lines)


def roundtrip_openui(
    source: str,
    *,
    slot_contract: Iterable[str] | None = None,
) -> tuple[ProductionProgram, str]:
    """Encode then decode; returns (program, reconstructed OpenUI)."""
    program = encode_openui(source, slot_contract=slot_contract)
    decoded = decode_productions(program.tokens, program.slot_contract)
    return program, decoded


def build_vocab_from_corpus(
    sources: Iterable[str],
    *,
    slot_contracts: Iterable[Iterable[str]] | None = None,
) -> ProductionVocab:
    """Grammar-closed vocab over production tokens observed in a corpus."""
    contracts = list(slot_contracts) if slot_contracts is not None else None
    programs: list[ProductionProgram] = []
    for i, src in enumerate(sources):
        contract = contracts[i] if contracts and i < len(contracts) else None
        programs.append(encode_openui(src, slot_contract=contract))
    return ProductionVocab.build(programs)


def _encode_expr(
    node: Any,
    *,
    slot_index: dict[str, int],
    stmt_index: dict[str, int],
) -> list[str]:
    if node is None:
        return [_literal_token(None)]
    if isinstance(node, bool):
        return [_literal_token(node)]
    if isinstance(node, (int, float)):
        return [_literal_token(node)]
    if isinstance(node, str):
        if is_placeholder(node):
            if node not in slot_index:
                raise ParseError(f"placeholder {node!r} missing from slot_contract")
            return [f"{SLOT_PREFIX}{slot_index[node]}"]
        if node in _DIRECTIONS:
            return [f"{DIR_PREFIX}{node}"]
        return [_literal_token(node)]
    if isinstance(node, list):
        out = [LIST_OPEN]
        for item in node:
            out.extend(_encode_expr(item, slot_index=slot_index, stmt_index=stmt_index))
        out.append(LIST_CLOSE)
        return out
    if isinstance(node, dict) and node.get("type") == "ref":
        name = str(node["name"])
        if name not in stmt_index:
            raise ParseError(f"unbound ref {name!r}")
        return [f"{REF_PREFIX}{stmt_index[name]}"]
    if isinstance(node, dict) and node.get("type") == "element":
        type_name = str(node.get("typeName") or "")
        props = dict(node.get("props") or {})
        tokens = [f"{OPEN_PREFIX}{type_name}"]
        tokens.extend(
            _encode_component_props(
                type_name,
                props,
                slot_index=slot_index,
                stmt_index=stmt_index,
            )
        )
        tokens.append(CLOSE)
        return tokens
    if isinstance(node, dict) and node.get("type") == "call":
        # Toy-layout style calls — treat as component with positional args.
        name = str(node.get("name") or "")
        args = list(node.get("args") or [])
        props = map_positional_props(name, args, _prop_order())
        tokens = [f"{OPEN_PREFIX}{name}"]
        tokens.extend(
            _encode_component_props(
                name,
                props,
                slot_index=slot_index,
                stmt_index=stmt_index,
            )
        )
        tokens.append(CLOSE)
        return tokens
    raise ParseError(f"unsupported AST node: {node!r}")


def _encode_component_props(
    type_name: str,
    props: dict[str, Any],
    *,
    slot_index: dict[str, int],
    stmt_index: dict[str, int],
) -> list[str]:
    order = list(_prop_order().get(type_name) or [])
    tokens: list[str] = []
    emitted_children = False

    if "children" in props and props["children"] is not None:
        tokens.extend(
            _encode_expr(
                props["children"],
                slot_index=slot_index,
                stmt_index=stmt_index,
            )
        )
        emitted_children = True

    for key in order:
        if key == "children" and emitted_children:
            continue
        if key not in props:
            continue
        value = props[key]
        if value is None:
            tokens.append(_literal_token(None))
            continue
        tokens.extend(
            _encode_expr(value, slot_index=slot_index, stmt_index=stmt_index)
        )

    extra = props.get("_args") or []
    for value in extra:
        tokens.extend(
            _encode_expr(value, slot_index=slot_index, stmt_index=stmt_index)
        )
    return tokens


def _literal_token(value: Any) -> str:
    if value is None:
        return f"{LIT_PREFIX}null"
    if isinstance(value, bool):
        return f"{LIT_PREFIX}{str(value).lower()}"
    if isinstance(value, (int, float)):
        return f"{LIT_PREFIX}{value}"
    return f"{LIT_PREFIX}{json.dumps(value)}"


def _decode_literal(payload: str) -> str:
    if payload == "null":
        return "null"
    if payload in {"true", "false"}:
        return payload
    try:
        if "." in payload:
            float(payload)
        else:
            int(payload)
        return payload
    except ValueError:
        pass
    return json.dumps(json.loads(payload))


__all__ = [
    "CLOSE",
    "DIR_PREFIX",
    "LIST_CLOSE",
    "LIST_OPEN",
    "LIT_PREFIX",
    "OPEN_PREFIX",
    "REF_PREFIX",
    "SLOT_PREFIX",
    "STMT",
    "ProductionProgram",
    "ProductionVocab",
    "build_vocab_from_corpus",
    "decode_productions",
    "encode_openui",
    "roundtrip_openui",
]
