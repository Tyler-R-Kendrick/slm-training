"""Generic Lark-file grammar backend for arbitrary DSLs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from lark import Lark, Token, Transformer, Tree, UnexpectedEOF, UnexpectedInput

from slm_training.dsl.lang_core import ParseError, Program
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.stream_types import StreamStatus
from slm_training.grammar_backends.ast_utils import (
    collect_placeholders_from_ast,
    map_positional_props,
)
from slm_training.grammar_backends.types import GrammarInfo


class _GenericTransformer(Transformer):
    """Build ElementNode-like trees from a simple call/list/string grammar."""

    def __init__(
        self,
        *,
        call_as_component: bool = True,
        prop_order: dict[str, list[str]] | None = None,
        component_rule: str = "component",
    ) -> None:
        super().__init__()
        self.call_as_component = call_as_component
        self.prop_order = prop_order or {}
        self.component_rule = component_rule
        self.bindings: dict[str, Any] = {}

    def STRING(self, tok: Token) -> str:
        raw = str(tok)
        return raw[1:-1].encode("utf-8").decode("unicode_escape")

    def NUMBER(self, tok: Token) -> float | int:
        text = str(tok)
        return float(text) if "." in text else int(text)

    def BOOL(self, tok: Token) -> bool:
        return str(tok) == "true"

    def NAME(self, tok: Token) -> str:
        return str(tok)

    def COMPONENT(self, tok: Token) -> str:
        return str(tok)

    def ref(self, items: list[Any]) -> dict[str, Any]:
        return {"type": "ref", "name": items[0]}

    def list(self, items: list[Any]) -> list[Any]:
        return list(items)

    def arg_list(self, items: list[Any]) -> list[Any]:
        return list(items)

    def _make_call(self, name: Any, args: list[Any]) -> dict[str, Any]:
        type_name = str(name)
        if self.call_as_component:
            props = map_positional_props(type_name, args, self.prop_order)
            return {
                "type": "element",
                "typeName": type_name,
                "props": props,
                "partial": False,
            }
        return {"type": "call", "name": type_name, "args": args}

    def call(self, items: list[Any]) -> dict[str, Any]:
        name = items[0]
        args = items[1] if len(items) > 1 else []
        if not isinstance(args, list):
            args = [args] if args is not None else []
        return self._make_call(name, args)

    def component(self, items: list[Any]) -> dict[str, Any]:
        return self.call(items)

    def statement(self, items: list[Any]) -> tuple[str, Any]:
        name, expr = items[0], items[1]
        self.bindings[str(name)] = expr
        return str(name), expr

    def start(self, items: list[Any]) -> dict[str, Any]:
        def resolve(node: Any) -> Any:
            if isinstance(node, dict) and node.get("type") == "ref":
                target = self.bindings.get(node["name"])
                if target is None:
                    return node
                resolved = resolve(target)
                if isinstance(resolved, dict) and resolved.get("type") == "element":
                    out = dict(resolved)
                    out["statementId"] = node["name"]
                    return out
                return resolved
            if isinstance(node, dict):
                return {k: resolve(v) for k, v in node.items()}
            if isinstance(node, list):
                return [resolve(x) for x in node]
            return node

        root_expr = self.bindings.get("root")
        if root_expr is None and items:
            root_expr = items[0][1] if isinstance(items[0], tuple) else None
        root = resolve(root_expr) if root_expr is not None else None
        if isinstance(root, dict) and root.get("type") == "element":
            root = {**root, "statementId": root.get("statementId") or "root"}
        return {
            "bindings": {k: resolve(v) for k, v in self.bindings.items()},
            "root": root,
        }


class LarkFileBackend:
    """Parse any DSL described by a `.lark` file into ElementNode-like ASTs."""

    name: str = "lark-file"
    dsl_id: str = "lark-file"

    def __init__(
        self,
        *,
        dsl_id: str,
        grammar_path: Path,
        description: str = "",
        root_name: str = "root",
        call_as_component: bool = True,
        start: str = "start",
        prop_order: dict[str, list[str]] | None = None,
        prop_order_path: Path | None = None,
        structural_extras: frozenset[str] | None = None,
    ) -> None:
        self._id = dsl_id
        self.dsl_id = dsl_id
        self.name = dsl_id
        self._path = Path(grammar_path)
        self._description = description or f"Lark grammar {self._path.name}"
        self._root_name = root_name
        self._call_as_component = call_as_component
        self._start = start
        self._prop_order = prop_order
        self._prop_order_path = Path(prop_order_path) if prop_order_path else None
        self._structural_extras = structural_extras or frozenset()
        self._parser: Lark | None = None
        self._loaded_prop_order: dict[str, list[str]] | None = None

    @property
    def info(self) -> GrammarInfo:
        return GrammarInfo(
            id=self._id,
            kind="lark",
            description=self._description,
            grammar_path=self._path,
            root_component=self._root_name,
        )

    def available(self) -> bool:
        return self._path.is_file()

    def is_available(self) -> bool:
        return self.available()

    def _order(self) -> dict[str, list[str]]:
        if self._prop_order is not None:
            return self._prop_order
        if self._loaded_prop_order is not None:
            return self._loaded_prop_order
        if self._prop_order_path and self._prop_order_path.is_file():
            self._loaded_prop_order = json.loads(
                self._prop_order_path.read_text(encoding="utf-8")
            )
            return self._loaded_prop_order
        self._loaded_prop_order = {}
        return self._loaded_prop_order

    def _lark(self) -> Lark:
        if self._parser is None:
            grammar = self._path.read_text(encoding="utf-8")
            self._parser = Lark(
                grammar,
                start=self._start,
                parser="lalr",
                maybe_placeholders=False,
            )
        return self._parser

    def _transform(self, tree: Tree) -> dict[str, Any]:
        transformer = _GenericTransformer(
            call_as_component=self._call_as_component,
            prop_order=self._order(),
        )
        return transformer.transform(tree)

    def parse(self, source: str) -> Program:
        text = source if source.endswith("\n") else source + "\n"
        try:
            tree = self._lark().parse(text)
            data = self._transform(tree)
        except UnexpectedInput as exc:
            raise ParseError(str(exc)) from exc
        root = data.get("root")
        ph = collect_placeholders_from_ast(root) or extract_placeholders(source)
        return Program(
            source=source,
            root=root if isinstance(root, dict) else None,
            placeholders=ph,
            meta={
                "backend": self._id,
                "kind": "lark",
                "bindings": list((data.get("bindings") or {}).keys()),
            },
            serialized=source.strip(),
        )

    def validate(self, source: str) -> Program:
        program = self.parse(source)
        if program.root is None:
            raise ParseError(f"{self._id}: missing root expression")
        return program

    def serialize(self, program: Program) -> str:
        return (program.serialized or program.source).strip()

    def stream_check(self, source: str) -> StreamStatus:
        text = source if source.endswith("\n") else source + "\n"
        try:
            tree = self._lark().parse(text)
            data = self._transform(tree)
            root = data.get("root")
            has_root = isinstance(root, dict)
            return StreamStatus(
                ok=True,
                incomplete=False,
                has_root=has_root,
                error_codes=(),
                unresolved=(),
                serialized=source.strip() if has_root else None,
            )
        except UnexpectedEOF:
            return StreamStatus(
                ok=True,
                incomplete=True,
                has_root="root" in source,
                error_codes=(),
                unresolved=(),
                serialized=None,
            )
        except UnexpectedInput as exc:
            # Treat parse failures at the interactive frontier as incomplete
            # when the buffer looks like a truncated program (unbalanced delimiters
            # or error near the end).
            msg = str(exc)
            open_delims = source.count("(") + source.count("[")
            close_delims = source.count(")") + source.count("]")
            unbalanced = open_delims > close_delims
            near_end = getattr(exc, "pos_in_stream", None)
            at_frontier = near_end is None or near_end >= max(0, len(source) - 3)
            if unbalanced or "Unexpected EOF" in msg or "Unexpected end" in msg:
                return StreamStatus(
                    ok=True,
                    incomplete=True,
                    has_root="root" in source,
                    error_codes=(),
                    unresolved=(),
                    serialized=None,
                )
            if at_frontier and not source.strip().endswith((")", "]")):
                return StreamStatus(
                    ok=True,
                    incomplete=True,
                    has_root="root" in source,
                    error_codes=(),
                    unresolved=(),
                    serialized=None,
                )
            return StreamStatus(
                ok=False,
                incomplete=False,
                has_root="root" in source,
                error_codes=("unexpected-token", msg[:80]),
                unresolved=(),
                serialized=None,
            )

    def structural_tokens(self) -> frozenset[str]:
        text = self._path.read_text(encoding="utf-8")
        names = set(re.findall(r"\b([A-Z][A-Za-z0-9]+)\b", text))
        names.update({"root", "(", ")", "[", "]", ",", "=", '"', "true", "false"})
        names.update(self._structural_extras)
        names.update(self._order().keys())
        return frozenset(names)

    def component_names(self) -> frozenset[str]:
        order = self._order()
        if order:
            return frozenset(order.keys())
        return frozenset(
            n for n in self.structural_tokens() if n[:1].isupper() and n.isidentifier()
        )

    def content_props(self) -> frozenset[str]:
        return frozenset(
            {
                "text",
                "label",
                "title",
                "body",
                "content",
                "placeholder",
                "alt",
                "hint",
                "description",
                "trigger",
            }
        )

    def library_schema(self) -> dict[str, Any]:
        return {
            "dsl": self._id,
            "grammar": str(self._path),
            "components": sorted(self.component_names()),
            "prop_order": self._order(),
        }

    def generate_system_prompt(self, **options: Any) -> str:
        _ = options
        comps = ", ".join(sorted(self.component_names())[:40])
        return (
            f"Generate a program in the {self._id} DSL.\n"
            f"Root binding must be `{self._root_name}`.\n"
            f"Known symbols: {comps}\n"
            "Use placeholders like :ns.slot for user-facing strings.\n"
        )
