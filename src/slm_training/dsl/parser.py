"""Minimal OpenUI subset parser, validator, and serializer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from slm_training.dsl.placeholders import CONTENT_PROPS, extract_placeholders, is_placeholder

COMPONENTS = frozenset({"Stack", "Card", "Text", "Button"})
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ParseError(ValueError):
    """Raised when OpenUI source is syntactically or semantically invalid."""


@dataclass
class Token:
    kind: str
    value: str
    pos: int


@dataclass
class Program:
    statements: list[tuple[str, Any]] = field(default_factory=list)
    placeholders: list[str] = field(default_factory=list)


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch.isspace() or ch == "\n" or ch == "\r":
            i += 1
            continue
        if ch in "()=,":
            tokens.append(Token(ch, ch, i))
            i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and source[j] != '"':
                if source[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                j += 1
            if j >= n:
                raise ParseError(f"unterminated string at position {i}")
            tokens.append(Token("STRING", source[i + 1 : j], i))
            i = j + 1
            continue
        if ch == ":":
            m = re.match(r":[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", source[i:])
            if not m:
                raise ParseError(f"invalid placeholder at position {i}")
            tokens.append(Token("PLACEHOLDER", m.group(0), i))
            i += len(m.group(0))
            continue
        if ch.isdigit() or (ch == "-" and i + 1 < n and source[i + 1].isdigit()):
            j = i + 1
            while j < n and (source[j].isdigit() or source[j] == "."):
                j += 1
            tokens.append(Token("NUMBER", source[i:j], i))
            i = j
            continue
        m = IDENT_RE.match(source, i)
        if m:
            ident = m.group(0)
            if ident in {"true", "false"}:
                tokens.append(Token("BOOL", ident, i))
            else:
                tokens.append(Token("IDENT", ident, i))
            i = m.end()
            continue
        raise ParseError(f"unexpected character {ch!r} at position {i}")
    tokens.append(Token("EOF", "", i))
    return tokens


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.i = 0

    def peek(self) -> Token:
        return self.tokens[self.i]

    def advance(self) -> Token:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def expect(self, kind: str) -> Token:
        tok = self.advance()
        if tok.kind != kind:
            raise ParseError(
                f"expected {kind} at position {tok.pos}, got {tok.kind} ({tok.value!r})"
            )
        return tok

    def parse_program(self) -> Program:
        statements: list[tuple[str, Any]] = []
        while self.peek().kind != "EOF":
            statements.append(self.parse_statement())
        if not statements:
            raise ParseError("program must contain at least one statement")
        source_repr = serialize_ast(statements)
        return Program(statements=statements, placeholders=extract_placeholders(source_repr))

    def parse_statement(self) -> tuple[str, Any]:
        name = self.expect("IDENT").value
        self.expect("=")
        expr = self.parse_expr()
        return name, expr

    def parse_expr(self) -> Any:
        tok = self.peek()
        if tok.kind == "PLACEHOLDER":
            self.advance()
            return {"type": "placeholder", "name": tok.value}
        if tok.kind == "IDENT":
            # Component call or reference
            if tok.value in COMPONENTS and self._lookahead_is("("):
                return self.parse_component()
            self.advance()
            return {"type": "ref", "name": tok.value}
        raise ParseError(f"expected expression at position {tok.pos}")

    def _lookahead_is(self, kind: str) -> bool:
        if self.i + 1 >= len(self.tokens):
            return False
        return self.tokens[self.i + 1].kind == kind

    def parse_component(self) -> dict[str, Any]:
        name = self.expect("IDENT").value
        if name not in COMPONENTS:
            raise ParseError(f"unknown component {name!r}")
        self.expect("(")
        args: dict[str, Any] = {}
        if self.peek().kind != ")":
            while True:
                key = self.expect("IDENT").value
                self.expect("=")
                args[key] = self.parse_value(prop_name=key)
                if self.peek().kind == ",":
                    self.advance()
                    continue
                break
        self.expect(")")
        return {"type": "component", "name": name, "args": args}

    def parse_value(self, prop_name: str) -> Any:
        tok = self.peek()
        if tok.kind == "STRING":
            self.advance()
            if prop_name in CONTENT_PROPS:
                raise ParseError(
                    f"content prop {prop_name!r} must be a placeholder, got string literal"
                )
            return {"type": "string", "value": tok.value}
        if tok.kind == "NUMBER":
            self.advance()
            raw = tok.value
            num: float | int = float(raw) if "." in raw else int(raw)
            return {"type": "number", "value": num}
        if tok.kind == "BOOL":
            self.advance()
            return {"type": "bool", "value": tok.value == "true"}
        if tok.kind == "PLACEHOLDER":
            self.advance()
            return {"type": "placeholder", "name": tok.value}
        if tok.kind == "IDENT":
            return self.parse_expr()
        raise ParseError(f"expected value at position {tok.pos}")


def parse(source: str) -> Program:
    tokens = tokenize(source)
    return _Parser(tokens).parse_program()


def validate(source: str) -> Program:
    """Parse and enforce semantic rules (content props, known components)."""
    program = parse(source)
    for _name, expr in program.statements:
        _validate_expr(expr)
    # Ensure content props that appear use placeholders (already enforced in parse_value).
    # Also reject placeholder-shaped strings nowhere else needed.
    if not program.placeholders and _has_content_component(program):
        # Text/Button/Card with content props should have produced placeholders;
        # allow structure-only programs.
        pass
    return program


def _has_content_component(program: Program) -> bool:
    def walk(node: Any) -> bool:
        if isinstance(node, dict):
            if node.get("type") == "component":
                args = node.get("args") or {}
                if CONTENT_PROPS.intersection(args):
                    return True
            return any(walk(v) for v in node.values())
        if isinstance(node, list):
            return any(walk(v) for v in node)
        return False

    return any(walk(expr) for _, expr in program.statements)


def _validate_expr(expr: Any) -> None:
    if not isinstance(expr, dict):
        return
    if expr.get("type") == "component":
        if expr["name"] not in COMPONENTS:
            raise ParseError(f"unknown component {expr['name']!r}")
        for key, val in (expr.get("args") or {}).items():
            if key in CONTENT_PROPS:
                if not (isinstance(val, dict) and val.get("type") == "placeholder"):
                    raise ParseError(f"content prop {key!r} must be a placeholder")
                if not is_placeholder(val["name"]):
                    raise ParseError(f"invalid placeholder {val['name']!r}")
            _validate_expr(val)
    elif expr.get("type") in {"ref", "placeholder", "string", "number", "bool"}:
        return
    else:
        for val in expr.values():
            _validate_expr(val)


def serialize_ast(statements: list[tuple[str, Any]]) -> str:
    lines = [f"{name} = {_serialize_expr(expr)}" for name, expr in statements]
    return "\n".join(lines)


def serialize(program: Program) -> str:
    return serialize_ast(program.statements)


def _serialize_expr(expr: Any) -> str:
    if not isinstance(expr, dict):
        return str(expr)
    kind = expr.get("type")
    if kind == "placeholder":
        return expr["name"]
    if kind == "ref":
        return expr["name"]
    if kind == "string":
        escaped = expr["value"].replace('"', '\\"')
        return f'"{escaped}"'
    if kind == "number":
        return str(expr["value"])
    if kind == "bool":
        return "true" if expr["value"] else "false"
    if kind == "component":
        args = expr.get("args") or {}
        if not args:
            return f"{expr['name']}()"
        inner = ", ".join(f"{k}={_serialize_expr(v)}" for k, v in args.items())
        return f"{expr['name']}({inner})"
    raise ParseError(f"cannot serialize node {expr!r}")
