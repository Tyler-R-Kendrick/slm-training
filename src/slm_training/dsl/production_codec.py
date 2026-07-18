"""Production codec: OpenUI ↔ compact grammar-native token sequence + slot pointers."""

from __future__ import annotations

import json
from functools import lru_cache
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from slm_training.data.contract import canonical_slot_contract
from slm_training.data.structure import strip_style_literals
from slm_training.dsl.lang_core import ParseError, Program
from slm_training.dsl.placeholders import is_placeholder
from slm_training.dsl.grammar.backends.ast_utils import map_positional_props
from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR

OPEN_PREFIX = "+"
CLOSE = "-"
DIR_PREFIX = "^"
SLOT_PREFIX = "@"
REF_PREFIX = "&"
REL_REF_PREFIX = "~"
LIT_PREFIX = "#"
LIST_OPEN = "["
LIST_CLOSE = "]"
STMT = "="
V05 = "!v0.5"
ROOT_STMT = "r="
STATE_STMT = "$="
QUERY_STMT = "q="
MUTATION_STMT = "m="
ACTION_STMT = "a="
EOL = ";"
STATE_REF_PREFIX = "$@"
BUILTIN_PREFIX = "*"
NAME_PREFIX = "n:"
PUNCT_PREFIX = "p:"
FRAGMENT_MARKERS = {
    "lexical": "!lexical",
    "expression": "!expression",
    "statement": "!statement",
}
FRAGMENT_CHUNK = "!fragment_chunk"

_DIRECTIONS = frozenset({"column", "row"})
_STMT_RE = re.compile(r"(?m)^([a-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
_V05_STMT_RE = re.compile(
    r"^\s*(\$?[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$"
)
_V05_LEX_RE = re.compile(
    r'''//[^\n]*|\#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|'''
    r"\$[A-Za-z_][A-Za-z0-9_]*|@[A-Z][A-Za-z0-9_]*|"
    r"[A-Za-z_][A-Za-z0-9_]*|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|"
    r"==|!=|>=|<=|&&|\|\||[=()\[\]{},.:?+*/%!<>-]"
)


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


# Backend ids that share the OpenUI language (and its prop-order file).
_OPENUI_DSL_KEYS = frozenset(
    {"openui", "openui-lark", "openui-langcore", "lark-openui", "default", "auto"}
)


@lru_cache(maxsize=8)
def _prop_order(dsl: str | None = None) -> dict[str, list[str]]:
    key = (dsl or "openui").strip().lower()
    if key in _OPENUI_DSL_KEYS:
        path = GRAMMARS_DIR / "openui_prop_order.json"
        return json.loads(path.read_text(encoding="utf-8"))
    from slm_training.dsl.grammar.backends import get_backend

    order = getattr(get_backend(key), "prop_order", None)
    if callable(order):
        return dict(order())
    raise ParseError(f"no prop order available for dsl {key!r}")


def _parse_program(source: str, dsl: str | None = None) -> Program:
    from slm_training.dsl.grammar.backends import get_backend

    text = strip_style_literals(source or "").strip()
    if not text:
        raise ParseError("empty source")
    return get_backend(dsl or "openui").validate(text)


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


def _split_top_level_args(inner: str) -> list[str]:
    """Split a comma-separated arg list respecting brackets, parens, and quotes."""
    parts: list[str] = []
    buf: list[str] = []
    depth_paren = 0
    depth_brack = 0
    in_string = False
    escape = False
    i = 0
    while i < len(inner):
        ch = inner[i]
        if in_string:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth_paren += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            buf.append(ch)
            i += 1
            continue
        if ch == "[":
            depth_brack += 1
            buf.append(ch)
            i += 1
            continue
        if ch == "]":
            depth_brack = max(0, depth_brack - 1)
            buf.append(ch)
            i += 1
            continue
        if ch == "," and depth_paren == 0 and depth_brack == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_rhs_value(token: str, dsl: str | None = None) -> Any:
    token = token.strip()
    if not token:
        raise ParseError("empty RHS value")
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_parse_rhs_value(part, dsl) for part in _split_top_level_args(inner)]
    if token.startswith('"') and token.endswith('"'):
        try:
            return json.loads(token)
        except json.JSONDecodeError as exc:
            raise ParseError(f"invalid string literal: {token}") from exc
    if token in {"true", "false"}:
        return token == "true"
    if token == "null":
        return None
    if _RHS_NUMBER_RE.fullmatch(token):
        return float(token) if "." in token else int(token)
    if _LOWER_IDENT_RE.fullmatch(token):
        return {"type": "ref", "name": token}
    match = _RHS_CALL_RE.match(token)
    if match:
        type_name = match.group(1)
        args = [
            _parse_rhs_value(part, dsl)
            for part in _split_top_level_args(match.group(2))
        ]
        props = map_positional_props(type_name, args, _prop_order(dsl))
        return {"type": "element", "typeName": type_name, "props": props}
    raise ParseError(f"unsupported RHS value: {token!r}")


def _parse_bindings(source: str, dsl: str | None = None) -> dict[str, Any]:
    """Return statement-level AST nodes keyed by binder name (refs preserved)."""
    _parse_program(source, dsl)
    bindings: dict[str, Any] = {}
    for match in _STMT_RE.finditer(source):
        name = match.group(1)
        rhs = match.group(2).strip()
        bindings[name] = _parse_rhs_value(rhs, dsl)
    if "root" not in bindings:
        raise ParseError("missing root binding")
    return bindings


def _ordered_prop_values(
    type_name: str, props: dict[str, Any], dsl: str | None = None
) -> list[Any]:
    """Prop values in declared positional order (absent middle props -> None).

    The decoder reconstructs a positional surface, so emission must follow the
    declared prop order exactly — otherwise the parser reassigns values to the
    wrong props on reparse (e.g. Modal children landing in ``title``).
    """
    order = list(_prop_order(dsl).get(type_name) or [])
    unknown = [key for key in props if key != "_args" and key not in order]
    if unknown:
        raise ParseError(
            f"component {type_name!r} props outside positional order: {unknown}"
        )
    present = [i for i, key in enumerate(order) if key in props]
    last = present[-1] if present else -1
    values: list[Any] = [props.get(key) for key in order[: last + 1]]
    values.extend(props.get("_args") or [])
    return values


def _expr_refs(node: Any, dsl: str | None = None) -> list[str]:
    """Statement refs in emission order (mirrors ``_encode_expr`` traversal)."""
    refs: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
            return
        if not isinstance(value, dict):
            return
        kind = value.get("type")
        if kind == "ref":
            refs.append(str(value["name"]))
            return
        if kind == "element":
            props = dict(value.get("props") or {})
            for item in _ordered_prop_values(
                str(value.get("typeName") or ""), props, dsl
            ):
                walk(item)
            return
        if kind == "call":
            name = str(value.get("name") or "")
            props = map_positional_props(
                name, list(value.get("args") or []), _prop_order(dsl)
            )
            for item in _ordered_prop_values(name, props, dsl):
                walk(item)

    walk(node)
    return refs


def _statement_order(bindings: dict[str, Any], dsl: str | None = None) -> list[str]:
    """Emission order: root's dependencies (DFS), remaining statements, root last.

    Refs are read from the parsed statement ASTs rather than surface text so
    the order is invariant under binder renaming: the decoder's alpha-renamed
    output re-encodes to the identical production stream, and the decoder's
    "last statement is root" naming convention always holds.
    """
    order: list[str] = []
    seen: set[str] = {"root"}

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        for ref in _expr_refs(bindings[name], dsl):
            if ref in bindings:
                visit(ref)
        order.append(name)

    for ref in _expr_refs(bindings["root"], dsl):
        if ref in bindings:
            visit(ref)
    for name in bindings:
        visit(name)
    order.append("root")
    return order


def _resolve_binding(bindings: dict[str, Any], name: str) -> Any:
    if name not in bindings:
        raise ParseError(f"undefined binding {name!r}")
    return bindings[name]


def _requires_v05_codec(source: str) -> bool:
    for piece in _V05_LEX_RE.findall(source):
        if piece.startswith(('"', "'", "//", "#")):
            continue
        if piece.startswith(("$", "@")) or piece in {"Query", "Mutation", "Action"}:
            return True
        if piece in {
            "{",
            "}",
            ".",
            ":",
            "?",
            "+",
            "-",
            "*",
            "/",
            "%",
            "!",
            "==",
            "!=",
            ">",
            "<",
            ">=",
            "<=",
            "&&",
            "||",
        }:
            return True
    return False


def _v05_statements(source: str) -> list[tuple[str, str]]:
    statements: list[tuple[str, str]] = []
    for line in source.splitlines():
        clean = line.strip()
        if not clean or clean.startswith(("//", "#")):
            continue
        match = _V05_STMT_RE.fullmatch(clean)
        if not match:
            raise ParseError(f"unsupported v0.5 statement: {line!r}")
        statements.append((match.group(1), match.group(2)))
    if not any(name == "root" for name, _ in statements):
        raise ParseError("missing root binding")
    return statements


def _v05_marker(name: str, rhs: str) -> str:
    if name == "root":
        return ROOT_STMT
    if name.startswith("$"):
        return STATE_STMT
    call = re.match(r"(Query|Mutation|Action)\s*\(", rhs)
    if call:
        return {
            "Query": QUERY_STMT,
            "Mutation": MUTATION_STMT,
            "Action": ACTION_STMT,
        }[call.group(1)]
    return STMT


def _encode_v05(
    source: str,
    *,
    slot_contract: Iterable[str] | None,
) -> ProductionProgram:
    _parse_program(source)
    statements = _v05_statements(source)
    contract = (
        canonical_slot_contract(source, declared=slot_contract)
        if slot_contract is not None
        else canonical_slot_contract(source)
    )
    slot_index = {placeholder: i for i, placeholder in enumerate(contract)}
    binder_index = {name: i for i, (name, _) in enumerate(statements)}
    state_names = [name for name, _ in statements if name.startswith("$")]
    state_index = {name: i for i, name in enumerate(state_names)}
    tokens = [V05]

    for name, rhs in statements:
        tokens.append(_v05_marker(name, rhs))
        pieces = [piece for piece in _V05_LEX_RE.findall(rhs) if not piece.startswith(("//", "#"))]
        for i, piece in enumerate(pieces):
            if piece.startswith(('"', "'")):
                value = json.loads(piece) if piece.startswith('"') else piece[1:-1]
                if isinstance(value, str) and is_placeholder(value):
                    if value not in slot_index:
                        raise ParseError(
                            f"placeholder {value!r} missing from slot_contract"
                        )
                    tokens.append(f"{SLOT_PREFIX}{slot_index[value]}")
                else:
                    tokens.append(_literal_token(value))
                continue
            if piece.startswith("$"):
                if piece not in state_index:
                    state_index[piece] = len(state_index)
                tokens.append(f"{STATE_REF_PREFIX}{state_index[piece]}")
                continue
            if piece.startswith("@"):
                tokens.append(f"{BUILTIN_PREFIX}{piece[1:]}")
                continue
            if _CHOICE_NUMBER_RE.fullmatch(piece):
                tokens.append(f"{LIT_PREFIX}{piece}")
                continue
            if piece in {"true", "false", "null"}:
                tokens.append(f"{LIT_PREFIX}{piece}")
                continue
            if _UPPER_IDENT_RE.fullmatch(piece):
                tokens.append(f"{OPEN_PREFIX}{piece}")
                continue
            if _LOWER_IDENT_RE.fullmatch(piece):
                literal_name = (
                    (i + 1 < len(pieces) and pieces[i + 1] == ":")
                    or (i > 0 and pieces[i - 1] == ".")
                )
                if piece in binder_index and not literal_name:
                    tokens.append(f"{REF_PREFIX}{binder_index[piece]}")
                else:
                    tokens.append(f"{NAME_PREFIX}{piece}")
                continue
            tokens.append(f"{PUNCT_PREFIX}{piece}")
        tokens.append(EOL)

    return ProductionProgram(tokens=tuple(tokens), slot_contract=tuple(contract))


def _with_relative_refs(
    program: ProductionProgram, *, relative_refs: bool
) -> ProductionProgram:
    if not relative_refs:
        return program
    return ProductionProgram(
        tokens=to_relative_refs(program.tokens),
        slot_contract=program.slot_contract,
    )


def encode_openui(
    source: str,
    *,
    slot_contract: Iterable[str] | None = None,
    relative_refs: bool = False,
    dsl: str | None = None,
) -> ProductionProgram:
    """Parse source and emit a compact production token sequence.

    ``dsl`` selects the grammar backend + prop order (default: OpenUI). The
    v0.5 typed path is OpenUI-specific and ignores non-OpenUI ``dsl`` values.
    When ``relative_refs`` is set, statement references are emitted as
    scope-relative De Bruijn deltas (C1) instead of absolute slot indices.
    """
    scrubbed = strip_style_literals(source or "").strip()
    if _requires_v05_codec(scrubbed):
        return _with_relative_refs(
            _encode_v05(scrubbed, slot_contract=slot_contract),
            relative_refs=relative_refs,
        )
    _parse_program(scrubbed, dsl)
    bindings = _parse_bindings(scrubbed, dsl)
    contract = (
        canonical_slot_contract(scrubbed, declared=slot_contract)
        if slot_contract is not None
        else canonical_slot_contract(scrubbed)
    )
    slot_index = {ph: i for i, ph in enumerate(contract)}
    stmt_order = _statement_order(bindings, dsl)
    stmt_index = {name: i for i, name in enumerate(stmt_order)}

    tokens: list[str] = []
    for position, name in enumerate(stmt_order):
        tokens.append(STMT)
        expr = _resolve_binding(bindings, name)
        tokens.extend(
            _encode_expr(
                expr,
                slot_index=slot_index,
                stmt_index=stmt_index,
                stmt_limit=position,
                dsl=dsl,
            )
        )
    return _with_relative_refs(
        ProductionProgram(tokens=tuple(tokens), slot_contract=tuple(contract)),
        relative_refs=relative_refs,
    )


def encode_output(
    source: str,
    *,
    output_kind: str = "document",
    slot_contract: Iterable[str] | None = None,
    relative_refs: bool = False,
) -> ProductionProgram:
    """Encode a full document or a validated compact output surface."""
    if output_kind == "document":
        return encode_openui(
            source, slot_contract=slot_contract, relative_refs=relative_refs
        )
    from slm_training.dsl.parser import lexical_tokens, validate_output

    validate_output(source, output_kind)  # type: ignore[arg-type]
    contract = canonical_slot_contract(source, declared=slot_contract)
    slot_index = {placeholder: i for i, placeholder in enumerate(contract)}
    tokens = [FRAGMENT_MARKERS[output_kind]]
    for piece in lexical_tokens(source):
        if piece.startswith(('"', "'")):
            value = json.loads(piece) if piece.startswith('"') else piece[1:-1]
            if isinstance(value, str) and is_placeholder(value):
                tokens.append(f"{SLOT_PREFIX}{slot_index[value]}")
            else:
                tokens.append(_literal_token(value))
        elif _CHOICE_NUMBER_RE.fullmatch(piece):
            tokens.append(f"{LIT_PREFIX}{piece}")
        elif piece in {"true", "false", "null"}:
            tokens.append(f"{LIT_PREFIX}{piece}")
        elif piece.startswith("@"):
            tokens.append(f"{BUILTIN_PREFIX}{piece[1:]}")
        elif piece[:1].isupper() and piece.isidentifier():
            tokens.append(f"{OPEN_PREFIX}{piece}")
        elif piece.isidentifier():
            tokens.append(f"{NAME_PREFIX}{piece}")
        else:
            tokens.append(f"{PUNCT_PREFIX}{piece}")
    return ProductionProgram(tuple(tokens), tuple(contract))


def _decode_output_tokens(tokens: list[str], contract: tuple[str, ...]) -> str:
    marker = tokens.pop(0)

    def surface(token: str) -> str:
        if token.startswith(SLOT_PREFIX):
            return json.dumps(contract[int(token[len(SLOT_PREFIX) :])])
        if token.startswith(BUILTIN_PREFIX):
            return f"@{token[len(BUILTIN_PREFIX):]}"
        if token.startswith(OPEN_PREFIX):
            return token[len(OPEN_PREFIX) :]
        if token.startswith(NAME_PREFIX):
            return token[len(NAME_PREFIX) :]
        if token.startswith(PUNCT_PREFIX):
            return token[len(PUNCT_PREFIX) :]
        if token.startswith(LIT_PREFIX):
            return _decode_literal(token[len(LIT_PREFIX) :])
        raise ParseError(f"unknown fragment production token: {token}")

    pieces = [surface(token) for token in tokens]
    text = ""
    for piece in pieces:
        if piece == ",":
            text = text.rstrip() + ", "
        elif piece == "=":
            text = text.rstrip() + " = "
        else:
            text += piece
    from slm_training.dsl.parser import validate_output

    kind = next(kind for kind, value in FRAGMENT_MARKERS.items() if value == marker)
    return validate_output(text.strip(), kind)  # type: ignore[arg-type]


def decode_productions(
    tokens: Iterable[str],
    slot_contract: Iterable[str],
    *,
    root_name: str = "root",
) -> str:
    """Reconstruct deterministic OpenUI source from production tokens + contract."""
    contract = tuple(slot_contract)
    stream = list(tokens)
    # De Bruijn refs (C1) are self-describing; restore absolute indices first.
    if any(tok.startswith(REL_REF_PREFIX) for tok in stream):
        stream = list(from_relative_refs(stream))
    if stream[:1] and stream[0] in FRAGMENT_MARKERS.values():
        return _decode_output_tokens(stream, contract)
    if stream[:1] == [V05]:
        return _decode_v05(stream, contract, root_name=root_name)
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


def _decode_v05(
    stream: list[str],
    contract: tuple[str, ...],
    *,
    root_name: str,
) -> str:
    """Decode the typed v0.5 lexical stream with deterministic alpha-renaming."""
    markers = {ROOT_STMT, STATE_STMT, QUERY_STMT, MUTATION_STMT, ACTION_STMT, STMT}
    sections: list[tuple[str, list[str]]] = []
    pos = 1
    while pos < len(stream):
        marker = stream[pos]
        pos += 1
        if marker not in markers:
            raise ParseError(f"expected v0.5 statement marker, got {marker!r}")
        end = pos
        while end < len(stream) and stream[end] != EOL:
            end += 1
        if end == len(stream):
            raise ParseError("unterminated v0.5 production statement")
        sections.append((marker, stream[pos:end]))
        pos = end + 1

    binder_names: list[str] = []
    counters = {STATE_STMT: 0, QUERY_STMT: 0, MUTATION_STMT: 0, ACTION_STMT: 0}
    prefixes = {QUERY_STMT: "q", MUTATION_STMT: "m", ACTION_STMT: "a"}
    for index, (marker, _) in enumerate(sections):
        if marker == ROOT_STMT:
            name = root_name
        elif marker == STATE_STMT:
            name = f"$s{counters[STATE_STMT]}"
            counters[STATE_STMT] += 1
        elif marker in prefixes:
            name = f"{prefixes[marker]}{counters[marker]}"
            counters[marker] += 1
        else:
            name = f"v{index}"
        binder_names.append(name)

    state_names = [
        name
        for (marker, _), name in zip(sections, binder_names)
        if marker == STATE_STMT
    ]

    def surface(token: str) -> str:
        if token.startswith(SLOT_PREFIX):
            index = int(token[len(SLOT_PREFIX) :])
            if index < 0 or index >= len(contract):
                raise ParseError(f"slot pointer out of range: {token}")
            return json.dumps(contract[index])
        if token.startswith(STATE_REF_PREFIX):
            index = int(token[len(STATE_REF_PREFIX) :])
            return state_names[index] if index < len(state_names) else f"$s{index}"
        if token.startswith(REF_PREFIX):
            index = int(token[len(REF_PREFIX) :])
            if index < 0 or index >= len(binder_names):
                raise ParseError(f"statement ref out of range: {token}")
            return binder_names[index]
        if token.startswith(BUILTIN_PREFIX):
            return f"@{token[len(BUILTIN_PREFIX):]}"
        if token.startswith(OPEN_PREFIX):
            return token[len(OPEN_PREFIX) :]
        if token.startswith(NAME_PREFIX):
            return token[len(NAME_PREFIX) :]
        if token.startswith(PUNCT_PREFIX):
            return token[len(PUNCT_PREFIX) :]
        if token.startswith(LIT_PREFIX):
            return _decode_literal(token[len(LIT_PREFIX) :])
        raise ParseError(f"unknown v0.5 production token: {token}")

    def pretty(tokens: list[str]) -> str:
        pieces = [surface(token) for token in tokens]
        out: list[str] = []
        operators = {"?", "+", "-", "*", "/", "%", "==", "!=", ">", "<", ">=", "<=", "&&", "||"}
        for piece in pieces:
            if piece == ",":
                out.append(", ")
            elif piece == ":":
                out.append(": ")
            elif piece in operators:
                out.append(f" {piece} ")
            elif piece == "!":
                out.append(piece)
            else:
                out.append(piece)
        return "".join(out)

    # Preserve stream order: the v0.5 encoder keeps source statement order, so
    # reordering here would break encode→decode→encode token idempotence.
    lines = [
        f"{name} = {pretty(tokens)}"
        for name, (_, tokens) in zip(binder_names, sections)
    ]
    return "\n".join(lines)


_V05_MARKERS = frozenset(
    {ROOT_STMT, STATE_STMT, QUERY_STMT, MUTATION_STMT, ACTION_STMT, STMT}
)


def _statement_markers(stream: list[str]) -> frozenset[str]:
    """Tokens that open a new statement (used to index De Bruijn ref distances)."""
    if stream[:1] == [V05]:
        return _V05_MARKERS
    return frozenset({STMT})


def to_relative_refs(tokens: Iterable[str]) -> tuple[str, ...]:
    """Rewrite absolute statement refs (``&i``) as scope-relative De Bruijn deltas.

    A reference token ``&i`` inside the statement at index ``cur`` becomes
    ``~{cur - i}`` — the signed distance, in canonical statement order, from the
    use site back to the binder's definition. This makes references
    translation-invariant: inserting or deleting an unrelated earlier statement
    renumbers absolute slots but leaves the local ``def→use`` distance unchanged,
    so a diffusion edit near one binder does not perturb refs elsewhere.

    Fragment streams carry no refs and are returned unchanged. See C1 /
    ``docs/design/iter-relative-index-refs-20260717.md``.
    """
    stream = list(tokens)
    markers = _statement_markers(stream)
    out: list[str] = []
    cur = -1
    for tok in stream:
        if tok in markers:
            cur += 1
        if tok.startswith(REF_PREFIX):
            idx = int(tok[len(REF_PREFIX):])
            out.append(f"{REL_REF_PREFIX}{cur - idx}")
        else:
            out.append(tok)
    return tuple(out)


def from_relative_refs(tokens: Iterable[str]) -> tuple[str, ...]:
    """Inverse of :func:`to_relative_refs` (``~delta`` → ``&i``), verifier-enforced.

    Legality is enforced here, not learned: a delta that resolves to a negative
    (undefined) statement index raises :class:`ParseError`. The absolute-ref
    decoders already range-check the upper bound against the decoded binder set.
    """
    stream = list(tokens)
    markers = _statement_markers(stream)
    out: list[str] = []
    cur = -1
    for tok in stream:
        if tok in markers:
            cur += 1
        if tok.startswith(REL_REF_PREFIX):
            delta = int(tok[len(REL_REF_PREFIX):])
            idx = cur - delta
            if idx < 0:
                raise ParseError(f"relative ref {tok} resolves before scope start")
            out.append(f"{REF_PREFIX}{idx}")
        else:
            out.append(tok)
    return tuple(out)


def _expr_spans(stream: list[str]) -> list[tuple[int, int]]:
    """Half-open spans of consecutive top-level expressions in a document stream."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(stream)
    while i < n:
        start = i
        stack: list[str] = []
        while True:
            if i >= n:
                raise ParseError("unterminated expression in choice stream")
            tok = stream[i]
            if tok == LIST_OPEN:
                stack.append(LIST_CLOSE)
            elif tok.startswith(OPEN_PREFIX):
                stack.append(CLOSE)
            elif tok in (LIST_CLOSE, CLOSE):
                if not stack or stack[-1] != tok:
                    raise ParseError(f"unbalanced {tok!r} in choice stream")
                stack.pop()
            i += 1
            if not stack:
                break
        spans.append((start, i))
    return spans


def to_choice_stream(tokens: Iterable[str]) -> tuple[str, ...]:
    """Drop grammar-forced framing tokens, leaving only semantic choices (B1).

    Document streams lose the ``=`` statement markers — top-level expressions
    are self-delimiting, so the marker carries zero bits. v0.5 streams lose the
    ``;`` terminators — the next typed statement marker is the boundary (the
    typed markers themselves stay: statement *kind* is a semantic choice).
    Fragment streams carry no framing and pass through unchanged.

    Unlike relative refs, a choice stream is NOT self-describing — decode it
    via :func:`from_choice_stream` / ``ProductionCodec(choice_stream=True)``,
    never by sniffing. Composes with relative refs: convert refs first (the
    delta arithmetic counts statement markers), then elide the framing.
    """
    stream = list(tokens)
    if not stream:
        return ()
    if stream[0] == V05:
        return tuple(tok for tok in stream if tok != EOL)
    if stream[0] in FRAGMENT_MARKERS.values():
        return tuple(stream)
    return tuple(tok for tok in stream if tok != STMT)


def from_choice_stream(tokens: Iterable[str]) -> tuple[str, ...]:
    """Inverse of :func:`to_choice_stream`: reinsert the deterministic framing.

    Framing is reconstructed, not predicted: document statement boundaries come
    from walking the self-delimiting expression grammar, v0.5 boundaries from
    the typed statement markers. A stream whose framing cannot be reconstructed
    (unbalanced or truncated frames) raises :class:`ParseError` — fail closed.
    """
    stream = list(tokens)
    if not stream:
        return ()
    if stream[0] in FRAGMENT_MARKERS.values():
        return tuple(stream)
    if stream[0] == V05:
        out = [V05]
        i = 1
        n = len(stream)
        while i < n:
            marker = stream[i]
            if marker not in _V05_MARKERS:
                raise ParseError(
                    f"expected v0.5 statement marker in choice stream, got {marker!r}"
                )
            out.append(marker)
            i += 1
            while i < n and stream[i] not in _V05_MARKERS:
                out.append(stream[i])
                i += 1
            out.append(EOL)
        return tuple(out)
    out: list[str] = []
    for start, end in _expr_spans(stream):
        out.append(STMT)
        out.extend(stream[start:end])
    return tuple(out)


def roundtrip_openui(
    source: str,
    *,
    slot_contract: Iterable[str] | None = None,
    relative_refs: bool = False,
    dsl: str | None = None,
) -> tuple[ProductionProgram, str]:
    """Encode then decode; returns (program, reconstructed OpenUI)."""
    program = encode_openui(
        source, slot_contract=slot_contract, relative_refs=relative_refs, dsl=dsl
    )
    decoded = decode_productions(program.tokens, program.slot_contract)
    return program, decoded


# ---------------------------------------------------------------------------
# B1 (SLM-42): pure grammar-choice stream.
#
# ``encode_choices`` emits ONLY semantic decisions — every token answers
# "which production applies here?" or "which filler goes in this slot?".
# All grammar-forced surface syntax (statement markers, call parens, commas,
# object braces' colons, statement terminators) is reconstructed by the
# deterministic detokenizer ``decode_choices``, which routes the final text
# through the official lang-core serializer (fail-closed on invalid
# reconstruction).
#
# Per-sigil keep/drop rationale vs the production stream:
#
#   token          keep?  why
#   -------------  -----  ----------------------------------------------------
#   +Comp          KEEP   production choice: which component/call head.
#   *Builtin       KEEP   production choice: which builtin action/aggregate.
#   -  (CLOSE)     KEEP   arity decision: prop/arg lists are variable length
#                         (openui_prop_order.json declares order, not arity),
#                         so "stop filling" is a real choice at every position.
#   [ / ]          KEEP   shape decision: the grammar admits scalar or list in
#                         any value position (prop order carries no types), so
#                         list-vs-scalar and list length are genuine choices.
#   { / }          KEEP   shape decision: object literal vs other expr, and
#                         entry count, are genuine choices (v0.5).
#   @i             KEEP   slot filler choice (which placeholder).
#   &i             KEEP   reference choice (which earlier statement).
#   $@i            KEEP   state reference choice.
#   #lit           KEEP   literal filler choice.
#   ^dir           KEEP   enum filler choice (column/row).
#   n:name         KEEP   object-key / unbound-identifier choice (which key).
#                         Irreducible: keys are user content, not grammar.
#   .name          KEEP   member-access choice (that access happens + which
#                         member) — one fused token (v0.5).
#   o:<op>         KEEP   operator choice (which operator applies IS a
#                         decision); operands are positional so no parens.
#   r=/$=/q=/m=/a= KEEP   statement-production choice (v0.5 only): state vs
#                         binder is not derivable from the RHS ("$x = 0" vs
#                         "x = 0"); root position is source order in v0.5.
#                         q=/m=/a= are technically derivable from the RHS head
#                         but kept so deterministic alpha-naming stays local.
#   =  (STMT)      DROP   structural path: statements are self-delimiting
#                         (one top-level expression each), so the marker is
#                         grammar-forced. v0.5 keeps its markers (above).
#   ;  (EOL)       DROP   grammar-forced: a statement ends exactly when its
#                         expression completes.
#   !v0.5          DROP   derivable: a stream starting with a statement
#                         marker is v0.5; otherwise structural.
#   p:punct        DROP   all of it: parens/commas/colons/braces' punctuation
#                         is reconstructed from the grammar; operator choices
#                         survive as ``o:<op>`` tokens instead.
# ---------------------------------------------------------------------------

OP_PREFIX = "o:"
MEMBER_PREFIX = "."
OBJ_OPEN = "{"
OBJ_CLOSE = "}"
TERNARY_OP = f"{OP_PREFIX}?:"
NOT_OP = f"{OP_PREFIX}!"
NEG_OP = f"{OP_PREFIX}neg"
INDEX_OP = f"{OP_PREFIX}[]"

_V05_BINARY_LEVELS: tuple[tuple[str, ...], ...] = (
    ("||",),
    ("&&",),
    ("==", "!="),
    (">=", "<=", ">", "<"),
    ("+", "-"),
    ("*", "/", "%"),
)
_CHOICE_BINARY_OPS = frozenset(op for level in _V05_BINARY_LEVELS for op in level)
CHOICE_STMT_MARKERS = (
    ROOT_STMT,
    STATE_STMT,
    QUERY_STMT,
    MUTATION_STMT,
    ACTION_STMT,
    STMT,
)
# Binary operator precedence (ternary=1 .. unary=8, postfix=9, atoms=10).
_CHOICE_PREC = {
    "||": 2,
    "&&": 3,
    "==": 4,
    "!=": 4,
    ">": 5,
    "<": 5,
    ">=": 5,
    "<=": 5,
    "+": 6,
    "-": 6,
    "*": 7,
    "/": 7,
    "%": 7,
}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LOWER_IDENT_RE = re.compile(r"[a-z_][A-Za-z0-9_]*")
_UPPER_IDENT_RE = re.compile(r"[A-Z][A-Za-z0-9_]*")
_CHOICE_NUMBER_RE = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_RHS_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_RHS_CALL_RE = re.compile(r"^([A-Z][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)


def _v05_normalize_pieces(pieces: list[str]) -> list[str]:
    """Split ``-5`` into ``-``, ``5`` when it follows a value (binary minus)."""
    out: list[str] = []
    for piece in pieces:
        prev = out[-1] if out else None
        if (
            piece.startswith("-")
            and _CHOICE_NUMBER_RE.fullmatch(piece)
            and prev is not None
            and (
                prev in {")", "]", "}"}
                or prev.startswith(('"', "'", "$"))
                or prev in {"true", "false", "null"}
                or _IDENT_RE.fullmatch(prev)
                or _CHOICE_NUMBER_RE.fullmatch(prev)
            )
        ):
            out.append("-")
            out.append(piece[1:])
            continue
        out.append(piece)
    return out


class _V05ExprParser:
    """Pratt parser for v0.5 RHS expressions (mirrors openui.lark precedence)."""

    def __init__(self, pieces: list[str]) -> None:
        self._pieces = pieces
        self._pos = 0

    def parse(self) -> Any:
        node = self._ternary()
        if self._pos != len(self._pieces):
            raise ParseError(
                f"trailing v0.5 expression input: {self._pieces[self._pos:]!r}"
            )
        return node

    def _peek(self) -> str | None:
        return self._pieces[self._pos] if self._pos < len(self._pieces) else None

    def _pop(self) -> str:
        if self._pos >= len(self._pieces):
            raise ParseError("unexpected end of v0.5 expression")
        piece = self._pieces[self._pos]
        self._pos += 1
        return piece

    def _expect(self, piece: str) -> None:
        got = self._pop()
        if got != piece:
            raise ParseError(f"expected {piece!r}, got {got!r}")

    def _ternary(self) -> Any:
        cond = self._binary(0)
        if self._peek() == "?":
            self._pop()
            then = self._ternary()
            self._expect(":")
            other = self._ternary()
            return ("ternary", cond, then, other)
        return cond

    def _binary(self, level: int) -> Any:
        if level >= len(_V05_BINARY_LEVELS):
            return self._unary()
        node = self._binary(level + 1)
        while self._peek() in _V05_BINARY_LEVELS[level]:
            op = self._pop()
            node = ("binary", op, node, self._binary(level + 1))
        return node

    def _unary(self) -> Any:
        piece = self._peek()
        if piece == "!":
            self._pop()
            return ("unary", "!", self._unary())
        if piece == "-":
            self._pop()
            return ("unary", "neg", self._unary())
        return self._postfix()

    def _postfix(self) -> Any:
        node = self._primary()
        while True:
            nxt = self._peek()
            if nxt == ".":
                self._pop()
                name = self._pop()
                if not _IDENT_RE.fullmatch(name or ""):
                    raise ParseError(f"invalid member name {name!r}")
                node = ("member", node, name)
                continue
            if nxt == "[":
                self._pop()
                index = self._ternary()
                self._expect("]")
                node = ("index", node, index)
                continue
            return node

    def _call_args(self) -> list[Any]:
        self._expect("(")
        args: list[Any] = []
        if self._peek() == ")":
            self._pop()
            return args
        args.append(self._ternary())
        while self._peek() == ",":
            self._pop()
            args.append(self._ternary())
        self._expect(")")
        return args

    def _primary(self) -> Any:
        piece = self._pop()
        if piece.startswith(('"', "'")):
            value = json.loads(piece) if piece.startswith('"') else piece[1:-1]
            return ("str", value)
        if _CHOICE_NUMBER_RE.fullmatch(piece):
            return ("num", piece)
        if piece in {"true", "false", "null"}:
            return ("kw", piece)
        if piece.startswith("$"):
            return ("state", piece)
        if piece.startswith("@"):
            return ("builtin", piece[1:], self._call_args())
        if _UPPER_IDENT_RE.fullmatch(piece):
            return ("call", piece, self._call_args())
        if _LOWER_IDENT_RE.fullmatch(piece):
            return ("ident", piece)
        if piece == "(":
            node = self._ternary()
            self._expect(")")
            return node
        if piece == "[":
            items: list[Any] = []
            if self._peek() == "]":
                self._pop()
                return ("array", items)
            items.append(self._ternary())
            while self._peek() == ",":
                self._pop()
                items.append(self._ternary())
            self._expect("]")
            return ("array", items)
        if piece == "{":
            entries: list[tuple[str, Any]] = []
            if self._peek() == "}":
                self._pop()
                return ("object", entries)
            while True:
                key = self._pop()
                if not _IDENT_RE.fullmatch(key or ""):
                    raise ParseError(f"unsupported v0.5 object key {key!r}")
                self._expect(":")
                entries.append((key, self._ternary()))
                if self._peek() == ",":
                    self._pop()
                    continue
                break
            self._expect("}")
            return ("object", entries)
        raise ParseError(f"unsupported v0.5 expression piece {piece!r}")


def _parse_v05_expr(rhs: str) -> Any:
    pieces = [
        piece
        for piece in _V05_LEX_RE.findall(rhs)
        if not piece.startswith(("//", "#"))
    ]
    return _V05ExprParser(_v05_normalize_pieces(pieces)).parse()


def _emit_choice_expr(
    node: Any,
    *,
    slot_index: dict[str, int],
    binder_index: dict[str, int],
    state_index: dict[str, int],
) -> list[str]:
    def emit(child: Any) -> list[str]:
        return _emit_choice_expr(
            child,
            slot_index=slot_index,
            binder_index=binder_index,
            state_index=state_index,
        )

    kind = node[0]
    if kind == "str":
        value = node[1]
        if isinstance(value, str) and is_placeholder(value):
            if value not in slot_index:
                raise ParseError(f"placeholder {value!r} missing from slot_contract")
            return [f"{SLOT_PREFIX}{slot_index[value]}"]
        return [_literal_token(value)]
    if kind in {"num", "kw"}:
        return [f"{LIT_PREFIX}{node[1]}"]
    if kind == "state":
        name = node[1]
        if name not in state_index:
            state_index[name] = len(state_index)
        return [f"{STATE_REF_PREFIX}{state_index[name]}"]
    if kind == "ident":
        name = node[1]
        if name in binder_index:
            return [f"{REF_PREFIX}{binder_index[name]}"]
        return [f"{NAME_PREFIX}{name}"]
    if kind == "member":
        return [f"{MEMBER_PREFIX}{node[2]}", *emit(node[1])]
    if kind == "index":
        return [INDEX_OP, *emit(node[1]), *emit(node[2])]
    if kind == "array":
        out = [LIST_OPEN]
        for item in node[1]:
            out.extend(emit(item))
        out.append(LIST_CLOSE)
        return out
    if kind == "object":
        out = [OBJ_OPEN]
        for key, value in node[1]:
            out.append(f"{NAME_PREFIX}{key}")
            out.extend(emit(value))
        out.append(OBJ_CLOSE)
        return out
    if kind == "call":
        out = [f"{OPEN_PREFIX}{node[1]}"]
        for arg in node[2]:
            out.extend(emit(arg))
        out.append(CLOSE)
        return out
    if kind == "builtin":
        out = [f"{BUILTIN_PREFIX}{node[1]}"]
        for arg in node[2]:
            out.extend(emit(arg))
        out.append(CLOSE)
        return out
    if kind == "unary":
        return [NOT_OP if node[1] == "!" else NEG_OP, *emit(node[2])]
    if kind == "binary":
        return [f"{OP_PREFIX}{node[1]}", *emit(node[2]), *emit(node[3])]
    if kind == "ternary":
        return [TERNARY_OP, *emit(node[1]), *emit(node[2]), *emit(node[3])]
    raise ParseError(f"unsupported v0.5 AST node: {node!r}")


def _encode_choices_v05(
    source: str,
    *,
    slot_contract: Iterable[str] | None,
) -> ProductionProgram:
    _parse_program(source)
    statements = _v05_statements(source)
    contract = (
        canonical_slot_contract(source, declared=slot_contract)
        if slot_contract is not None
        else canonical_slot_contract(source)
    )
    slot_index = {placeholder: i for i, placeholder in enumerate(contract)}
    binder_index = {name: i for i, (name, _) in enumerate(statements)}
    state_names = [name for name, _ in statements if name.startswith("$")]
    state_index = {name: i for i, name in enumerate(state_names)}
    tokens: list[str] = []
    for name, rhs in statements:
        tokens.append(_v05_marker(name, rhs))
        tokens.extend(
            _emit_choice_expr(
                _parse_v05_expr(rhs),
                slot_index=slot_index,
                binder_index=binder_index,
                state_index=state_index,
            )
        )
    return ProductionProgram(tokens=tuple(tokens), slot_contract=tuple(contract))


def encode_choices(
    source: str,
    *,
    slot_contract: Iterable[str] | None = None,
    dsl: str | None = None,
) -> ProductionProgram:
    """Encode OpenUI into the pure grammar-choice stream (semantic decisions only).

    The choice stream lives in **canonical space**: with the lang-core bridge
    available the input is first collapsed to the official serializer form
    (which may reorder statements, prune dead bindings / trailing ``null``
    props, and drop redundant parens). This keeps the token stream a fixed
    point of ``decode_choices`` — decode routes through the same serializer —
    and keeps the v0.5 path's codec-mode dispatch stable under decode.
    """
    from slm_training.dsl.lang_core import bridge_available

    scrubbed = strip_style_literals(source or "").strip()
    program = _parse_program(scrubbed, dsl)
    if bridge_available():
        serialized = strip_style_literals(program.serialized or "").strip()
        if serialized:
            scrubbed = serialized
    if _requires_v05_codec(scrubbed):
        return _encode_choices_v05(scrubbed, slot_contract=slot_contract)
    bindings = _parse_bindings(scrubbed, dsl)
    contract = (
        canonical_slot_contract(scrubbed, declared=slot_contract)
        if slot_contract is not None
        else canonical_slot_contract(scrubbed)
    )
    slot_index = {ph: i for i, ph in enumerate(contract)}
    stmt_order = _statement_order(bindings, dsl)
    stmt_index = {name: i for i, name in enumerate(stmt_order)}
    tokens: list[str] = []
    for position, name in enumerate(stmt_order):
        # No STMT marker: statements are self-delimiting expressions.
        tokens.extend(
            _encode_expr(
                _resolve_binding(bindings, name),
                slot_index=slot_index,
                stmt_index=stmt_index,
                stmt_limit=position,
                dsl=dsl,
            )
        )
    return ProductionProgram(tokens=tuple(tokens), slot_contract=tuple(contract))


def _canonicalize_choice_text(text: str) -> str:
    """Validate + serialize the reconstruction through the official pipeline.

    Fail closed: an unparseable reconstruction raises ``ParseError``. With the
    lang-core bridge available the returned text is the serializer fixed point.
    """
    from slm_training.dsl.lang_core import bridge_available
    from slm_training.dsl.parser import validate

    try:
        program = validate(text)
    except ParseError:
        raise
    except Exception as exc:  # noqa: BLE001 - backend-specific errors
        raise ParseError(f"invalid choice reconstruction: {exc}") from exc
    if bridge_available():
        serialized = (program.serialized or "").strip()
        if not serialized:
            raise ParseError("choice reconstruction produced no canonical form")
        return serialized
    return text


def _parse_choice_node(
    stream: list[str], pos: int, *, n_refs_available: int
) -> tuple[Any, int]:
    """Parse one choice-stream expression into a render node."""

    def parse(pos: int) -> tuple[Any, int]:
        if pos >= len(stream):
            raise ParseError("unexpected end of choice stream")
        tok = stream[pos]
        pos += 1
        if tok == LIST_OPEN:
            items: list[Any] = []
            while pos < len(stream) and stream[pos] != LIST_CLOSE:
                node, pos = parse(pos)
                items.append(node)
            if pos >= len(stream):
                raise ParseError("unterminated choice list")
            return ("array", items), pos + 1
        if tok == OBJ_OPEN:
            entries: list[tuple[str, Any]] = []
            while pos < len(stream) and stream[pos] != OBJ_CLOSE:
                key_tok = stream[pos]
                if not key_tok.startswith(NAME_PREFIX):
                    raise ParseError(f"expected object key token, got {key_tok!r}")
                pos += 1
                value, pos = parse(pos)
                entries.append((key_tok[len(NAME_PREFIX) :], value))
            if pos >= len(stream):
                raise ParseError("unterminated choice object")
            return ("object", entries), pos + 1
        if tok == TERNARY_OP:
            cond, pos = parse(pos)
            then, pos = parse(pos)
            other, pos = parse(pos)
            return ("ternary", cond, then, other), pos
        if tok in {NOT_OP, NEG_OP}:
            child, pos = parse(pos)
            return ("unary", "!" if tok == NOT_OP else "neg", child), pos
        if tok == INDEX_OP:
            base, pos = parse(pos)
            index, pos = parse(pos)
            return ("index", base, index), pos
        if tok.startswith(OP_PREFIX):
            op = tok[len(OP_PREFIX) :]
            if op not in _CHOICE_BINARY_OPS:
                raise ParseError(f"unknown choice operator token: {tok}")
            left, pos = parse(pos)
            right, pos = parse(pos)
            return ("binary", op, left, right), pos
        if tok.startswith(MEMBER_PREFIX):
            base, pos = parse(pos)
            return ("member", base, tok[len(MEMBER_PREFIX) :]), pos
        if tok.startswith(OPEN_PREFIX):
            args: list[Any] = []
            while pos < len(stream) and stream[pos] != CLOSE:
                node, pos = parse(pos)
                args.append(node)
            if pos >= len(stream):
                raise ParseError("unterminated choice call")
            return ("call", tok[len(OPEN_PREFIX) :], args), pos + 1
        if tok.startswith(BUILTIN_PREFIX):
            args = []
            while pos < len(stream) and stream[pos] != CLOSE:
                node, pos = parse(pos)
                args.append(node)
            if pos >= len(stream):
                raise ParseError("unterminated choice builtin call")
            return ("builtin", tok[len(BUILTIN_PREFIX) :], args), pos + 1
        if tok.startswith(STATE_REF_PREFIX):
            return ("stateref", int(tok[len(STATE_REF_PREFIX) :])), pos
        if tok.startswith(SLOT_PREFIX):
            return ("slot", int(tok[len(SLOT_PREFIX) :])), pos
        if tok.startswith(REF_PREFIX):
            index = int(tok[len(REF_PREFIX) :])
            if index < 0 or index >= n_refs_available:
                raise ParseError(f"statement ref out of range: {tok}")
            return ("ref", index), pos
        if tok.startswith(DIR_PREFIX):
            return ("atom", json.dumps(tok[len(DIR_PREFIX) :])), pos
        if tok.startswith(LIT_PREFIX):
            return ("atom", _decode_literal(tok[len(LIT_PREFIX) :])), pos
        if tok.startswith(NAME_PREFIX):
            return ("atom", tok[len(NAME_PREFIX) :]), pos
        raise ParseError(f"unknown choice token: {tok}")

    return parse(pos)


def _render_choice_node(
    node: Any,
    *,
    contract: tuple[str, ...],
    binder_names: list[str],
    state_names: list[str],
    min_prec: int = 0,
) -> str:
    def render(child: Any, min_prec: int = 0) -> str:
        return _render_choice_node(
            child,
            contract=contract,
            binder_names=binder_names,
            state_names=state_names,
            min_prec=min_prec,
        )

    kind = node[0]
    if kind == "atom":
        return node[1]
    if kind == "slot":
        index = node[1]
        if index < 0 or index >= len(contract):
            raise ParseError(f"slot pointer out of range: @{index}")
        return json.dumps(contract[index])
    if kind == "ref":
        index = node[1]
        if index < 0 or index >= len(binder_names):
            raise ParseError(f"statement ref out of range: &{index}")
        return binder_names[index]
    if kind == "stateref":
        index = node[1]
        return state_names[index] if index < len(state_names) else f"$s{index}"
    if kind == "array":
        return "[" + ", ".join(render(item) for item in node[1]) + "]"
    if kind == "object":
        return (
            "{"
            + ", ".join(f"{key}: {render(value)}" for key, value in node[1])
            + "}"
        )
    if kind == "call":
        return f"{node[1]}({', '.join(render(arg) for arg in node[2])})"
    if kind == "builtin":
        return f"@{node[1]}({', '.join(render(arg) for arg in node[2])})"
    if kind == "member":
        return f"{render(node[1], 9)}.{node[2]}"
    if kind == "index":
        return f"{render(node[1], 9)}[{render(node[2])}]"
    if kind == "unary":
        rendered = ("!" if node[1] == "!" else "-") + render(node[2], 8)
        return f"({rendered})" if min_prec > 8 else rendered
    if kind == "binary":
        prec = _CHOICE_PREC[node[1]]
        rendered = f"{render(node[2], prec)} {node[1]} {render(node[3], prec + 1)}"
        return f"({rendered})" if prec < min_prec else rendered
    if kind == "ternary":
        rendered = f"{render(node[1], 2)} ? {render(node[2], 1)} : {render(node[3], 1)}"
        return f"({rendered})" if min_prec > 1 else rendered
    raise ParseError(f"unsupported choice render node: {node!r}")


def _decode_choices_structural(
    stream: list[str], contract: tuple[str, ...], *, root_name: str
) -> str:
    nodes: list[Any] = []
    pos = 0
    while pos < len(stream):
        node, pos = _parse_choice_node(stream, pos, n_refs_available=len(nodes))
        nodes.append(node)
    if not nodes:
        raise ParseError("empty choice stream")
    # Forward references are unrepresentable (the encoder enforces
    # stmt_limit), so renaming the last statement to root after the fact is
    # exact: no earlier expression can reference it.
    binder_names = [f"v{i}" for i in range(len(nodes) - 1)] + [root_name]
    exprs = [
        _render_choice_node(
            node, contract=contract, binder_names=binder_names, state_names=[]
        )
        for node in nodes
    ]
    root_line = f"{root_name} = {exprs[-1]}"
    other_lines = [
        f"{binder_names[i]} = {exprs[i]}" for i in range(len(nodes) - 1)
    ]
    return "\n".join([root_line, *other_lines])


def _decode_choices_v05(
    stream: list[str], contract: tuple[str, ...], *, root_name: str
) -> str:
    markers = set(CHOICE_STMT_MARKERS)
    sections: list[tuple[str, Any]] = []
    pos = 0
    while pos < len(stream):
        marker = stream[pos]
        if marker not in markers:
            raise ParseError(f"expected choice statement marker, got {marker!r}")
        pos += 1
        node, pos = _parse_choice_node(stream, pos, n_refs_available=1 << 30)
        sections.append((marker, node))

    binder_names: list[str] = []
    counters = {STATE_STMT: 0, QUERY_STMT: 0, MUTATION_STMT: 0, ACTION_STMT: 0}
    prefixes = {QUERY_STMT: "q", MUTATION_STMT: "m", ACTION_STMT: "a"}
    for index, (marker, _) in enumerate(sections):
        if marker == ROOT_STMT:
            name = root_name
        elif marker == STATE_STMT:
            name = f"$s{counters[STATE_STMT]}"
            counters[STATE_STMT] += 1
        elif marker in prefixes:
            name = f"{prefixes[marker]}{counters[marker]}"
            counters[marker] += 1
        else:
            name = f"v{index}"
        binder_names.append(name)
    state_names = [
        name
        for (marker, _), name in zip(sections, binder_names)
        if marker == STATE_STMT
    ]
    lines = []
    for (marker, node), name in zip(sections, binder_names):
        rendered = _render_choice_node(
            node,
            contract=contract,
            binder_names=binder_names,
            state_names=state_names,
        )
        lines.append(f"{name} = {rendered}")
    return "\n".join(lines)


def decode_choices(
    tokens: Iterable[str],
    slot_contract: Iterable[str],
    *,
    root_name: str = "root",
) -> str:
    """Deterministically reconstruct canonical OpenUI from a choice stream.

    The mode is derivable: a stream starting with a v0.5 statement marker is
    v0.5, anything else is the structural path (whose statements carry no
    markers). The reconstruction is routed through the official lang-core
    serializer and fails closed on invalid input.
    """
    contract = tuple(slot_contract)
    stream = list(tokens)
    if not stream:
        raise ParseError("empty choice stream")
    if stream[0] in CHOICE_STMT_MARKERS:
        text = _decode_choices_v05(stream, contract, root_name=root_name)
    else:
        text = _decode_choices_structural(stream, contract, root_name=root_name)
    return _canonicalize_choice_text(text)


def roundtrip_choices(
    source: str,
    *,
    slot_contract: Iterable[str] | None = None,
    dsl: str | None = None,
) -> tuple[ProductionProgram, str]:
    """Encode to the choice stream then decode; returns (program, canonical text)."""
    program = encode_choices(source, slot_contract=slot_contract, dsl=dsl)
    decoded = decode_choices(program.tokens, program.slot_contract)
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
    stmt_limit: int | None = None,
    dsl: str | None = None,
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
            out.extend(
                _encode_expr(
                    item,
                    slot_index=slot_index,
                    stmt_index=stmt_index,
                    stmt_limit=stmt_limit,
                    dsl=dsl,
                )
            )
        out.append(LIST_CLOSE)
        return out
    if isinstance(node, dict) and node.get("type") == "ref":
        name = str(node["name"])
        if name not in stmt_index:
            raise ParseError(f"unbound ref {name!r}")
        index = stmt_index[name]
        if stmt_limit is not None and index >= stmt_limit:
            raise ParseError(
                f"forward reference {name!r} is not representable in the "
                "production stream"
            )
        return [f"{REF_PREFIX}{index}"]
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
                stmt_limit=stmt_limit,
                dsl=dsl,
            )
        )
        tokens.append(CLOSE)
        return tokens
    if isinstance(node, dict) and node.get("type") == "call":
        # Toy-layout style calls — treat as component with positional args.
        name = str(node.get("name") or "")
        args = list(node.get("args") or [])
        props = map_positional_props(name, args, _prop_order(dsl))
        tokens = [f"{OPEN_PREFIX}{name}"]
        tokens.extend(
            _encode_component_props(
                name,
                props,
                slot_index=slot_index,
                stmt_index=stmt_index,
                stmt_limit=stmt_limit,
                dsl=dsl,
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
    stmt_limit: int | None = None,
    dsl: str | None = None,
) -> list[str]:
    tokens: list[str] = []
    for value in _ordered_prop_values(type_name, props, dsl):
        if value is None:
            tokens.append(_literal_token(None))
            continue
        tokens.extend(
            _encode_expr(
                value,
                slot_index=slot_index,
                stmt_index=stmt_index,
                stmt_limit=stmt_limit,
                dsl=dsl,
            )
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


@dataclass
class ProductionCodec:
    """Grammar-native production codec with parallel slot pointers for diffusion."""

    production_to_id: dict[str, int] = field(default_factory=dict)
    id_to_production: dict[int, str] = field(default_factory=dict)
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    mask_id: int = 3
    unk_id: int = 4
    slot_none_id: int = 0
    relative_refs: bool = False
    choice_stream: bool = False

    _SPECIALS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<mask>", "<unk>")

    @classmethod
    def build(
        cls,
        texts: list[str],
        output_kinds: list[str] | None = None,
        *,
        relative_refs: bool = False,
        choice_stream: bool = False,
    ) -> ProductionCodec:
        vocab: dict[str, int] = {tok: i for i, tok in enumerate(cls._SPECIALS)}
        if output_kinds and any(kind != "document" for kind in output_kinds):
            vocab[FRAGMENT_CHUNK] = len(vocab)
        for index, text in enumerate(texts):
            kind = output_kinds[index] if output_kinds else "document"
            program = encode_output(
                text, output_kind=kind, relative_refs=relative_refs
            )
            tokens = (
                to_choice_stream(program.tokens) if choice_stream else program.tokens
            )
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        inv = {i: t for t, i in vocab.items()}
        return cls(
            production_to_id=vocab,
            id_to_production=inv,
            pad_id=vocab["<pad>"],
            bos_id=vocab["<bos>"],
            eos_id=vocab["<eos>"],
            mask_id=vocab["<mask>"],
            unk_id=vocab["<unk>"],
            relative_refs=relative_refs,
            choice_stream=choice_stream,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.production_to_id)

    def encode(
        self,
        openui: str,
        slot_inventory: list[str] | None = None,
        *,
        max_len: int = 256,
        output_kind: str = "document",
    ) -> tuple[list[int], list[int]]:
        contract = list(
            canonical_slot_contract(openui, declared=slot_inventory)
            if slot_inventory is not None
            else canonical_slot_contract(openui)
        )
        program = encode_output(
            openui,
            output_kind=output_kind,
            slot_contract=contract,
            relative_refs=self.relative_refs,
        )
        tokens = (
            to_choice_stream(program.tokens)
            if self.choice_stream
            else program.tokens
        )
        prod_ids = [self.bos_id]
        slot_ids = [self.slot_none_id]
        for tok in tokens:
            pid = self.production_to_id.get(tok)
            if pid is None:
                pid = self.production_to_id.setdefault(tok, len(self.production_to_id))
                self.id_to_production[pid] = tok
            prod_ids.append(pid)
            if tok.startswith(SLOT_PREFIX):
                slot_ids.append(int(tok[len(SLOT_PREFIX) :]) + 1)
            else:
                slot_ids.append(self.slot_none_id)
        prod_ids.append(self.eos_id)
        slot_ids.append(self.slot_none_id)
        if max_len > 0 and len(prod_ids) > max_len:
            prod_ids = prod_ids[:max_len]
            prod_ids[-1] = self.eos_id
            slot_ids = slot_ids[:max_len]
            slot_ids[-1] = self.slot_none_id
        if len(slot_ids) < len(prod_ids):
            slot_ids.extend([self.slot_none_id] * (len(prod_ids) - len(slot_ids)))
        return prod_ids, slot_ids

    def decode(
        self,
        production_ids: list[int],
        slot_ids: list[int],
        slot_inventory: list[str],
        *,
        stop_at_mask: bool = False,
    ) -> str:
        inventory = [
            ph if ph.startswith(":") else f":{ph}" for ph in slot_inventory
        ]
        tokens: list[str] = []
        hit_mask = False
        for pid, sid in zip(production_ids, slot_ids):
            if pid in {self.pad_id, self.bos_id}:
                continue
            if pid == self.eos_id:
                break
            if pid == self.mask_id:
                hit_mask = True
                break
            tok = self.id_to_production.get(pid, self._SPECIALS[-1])
            if tok.startswith(SLOT_PREFIX) and int(sid) > 0:
                tok = f"{SLOT_PREFIX}{int(sid) - 1}"
            tokens.append(tok)
        text = ""
        if tokens:
            try:
                if self.choice_stream:
                    tokens = list(from_choice_stream(tokens))
                text = decode_productions(tokens, inventory)
            except ParseError:
                text = ""
        if hit_mask and stop_at_mask:
            return f"{text}<mask>" if text else "<mask>"
        if text and not text.endswith("\n"):
            return text + "\n"
        return text


__all__ = [
    "ACTION_STMT",
    "BUILTIN_PREFIX",
    "CHOICE_STMT_MARKERS",
    "CLOSE",
    "DIR_PREFIX",
    "INDEX_OP",
    "LIST_CLOSE",
    "LIST_OPEN",
    "LIT_PREFIX",
    "MEMBER_PREFIX",
    "MUTATION_STMT",
    "NAME_PREFIX",
    "NEG_OP",
    "NOT_OP",
    "OBJ_CLOSE",
    "OBJ_OPEN",
    "OPEN_PREFIX",
    "OP_PREFIX",
    "REF_PREFIX",
    "REL_REF_PREFIX",
    "ROOT_STMT",
    "SLOT_PREFIX",
    "STMT",
    "STATE_REF_PREFIX",
    "STATE_STMT",
    "QUERY_STMT",
    "TERNARY_OP",
    "V05",
    "ProductionCodec",
    "ProductionProgram",
    "ProductionVocab",
    "build_vocab_from_corpus",
    "decode_choices",
    "decode_productions",
    "encode_choices",
    "encode_openui",
    "from_choice_stream",
    "from_relative_refs",
    "roundtrip_choices",
    "roundtrip_openui",
    "to_choice_stream",
    "to_relative_refs",
]
