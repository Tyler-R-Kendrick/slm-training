"""GraphQL query backend (F2 / SLM-43): schema-native, graphql-core oracle.

Implements the GrammarBackend Protocol directly on graphql-core instead of a
Lark grammar: `parse`/`print` come from the reference implementation, and
`validate` runs full schema validation against a committed SDL fixture — the
introspection schema literally supplies the symbol table (`component_names`
= schema type names, `structural_tokens` from the query surface). Stated
boundary: graphql-core tracks the GraphQL spec like graphql-js but byte
parity with graphql-js is a non-goal (recorded in grammar-backends.md).

All graphql imports are lazy so registration never requires the optional
dependency; `available()` gates tests the same way `bridge_available` does.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from slm_training.bridge_utils import repo_root
from slm_training.dsl.grammar.backends.types import GrammarBackend, GrammarInfo
from slm_training.dsl.lang_core import ParseError, Program
from slm_training.dsl.stream_types import StreamStatus

DEFAULT_SCHEMA_PATH = (
    repo_root() / "src" / "slm_training" / "resources" / "graphql" / "shop_schema.graphql"
)

_STRUCTURAL = frozenset(
    {"{", "}", "(", ")", ":", ",", "$", "!", "[", "]", "query", "...", "="}
)


def graphql_available() -> bool:
    try:
        import graphql  # noqa: F401

        return DEFAULT_SCHEMA_PATH.is_file()
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=4)
def _schema(schema_path: str):
    from graphql import build_schema

    return build_schema(Path(schema_path).read_text(encoding="utf-8"))


class GraphQLQueryBackend:
    """Query-side GraphQL backend over a fixed SDL fixture schema."""

    def __init__(self, schema_path: Path | None = None) -> None:
        self._schema_path = Path(schema_path or DEFAULT_SCHEMA_PATH)

    @property
    def info(self) -> GrammarInfo:
        return GrammarInfo(
            id="graphql",
            kind="schema",
            description="GraphQL queries validated against a fixture "
            "introspection schema (graphql-core oracle)",
            grammar_path=None,
            root_component="query",
        )

    def available(self) -> bool:
        return graphql_available() and self._schema_path.is_file()

    def is_available(self) -> bool:
        return self.available()

    def _parse_document(self, source: str):
        from graphql import GraphQLSyntaxError, parse

        try:
            return parse(source)
        except GraphQLSyntaxError as exc:
            raise ParseError(f"graphql: {exc.message}") from exc

    def parse(self, source: str) -> Program:
        document = self._parse_document(source)
        from graphql import print_ast

        operations = [
            getattr(defn, "operation", None)
            for defn in document.definitions
        ]
        root = {
            "type": "document",
            "operations": [
                getattr(op, "value", None) for op in operations if op is not None
            ],
        }
        return Program(
            source=source,
            root=root,
            placeholders=[],
            meta={"backend": "graphql", "kind": "schema"},
            serialized=print_ast(document).strip(),
        )

    def validate(self, source: str) -> Program:
        from graphql import validate as gql_validate

        document = self._parse_document(source)
        errors = gql_validate(_schema(str(self._schema_path)), document)
        if errors:
            raise ParseError(
                "graphql schema validation: "
                + "; ".join(error.message for error in errors[:3])
            )
        return self.parse(source)

    def serialize(self, program: Program) -> str:
        return (program.serialized or program.source).strip()

    def stream_check(self, source: str) -> StreamStatus:
        try:
            self._parse_document(source)
            return StreamStatus(
                ok=True,
                incomplete=False,
                has_root=True,
                error_codes=(),
                unresolved=(),
            )
        except ParseError as exc:
            # Unbalanced braces at end-of-input = plausibly incomplete prefix.
            open_braces = source.count("{") - source.count("}")
            incomplete = open_braces > 0 or "EOF" in str(exc)
            return StreamStatus(
                ok=False,
                incomplete=incomplete,
                has_root=False,
                error_codes=("parse-error",),
                unresolved=(),
            )

    def structural_tokens(self) -> frozenset[str]:
        return _STRUCTURAL

    def component_names(self) -> frozenset[str]:
        # The schema IS the symbol table: type names are the fixed alphabet.
        if not self.available():
            return frozenset()
        schema = _schema(str(self._schema_path))
        return frozenset(
            name for name in schema.type_map if not name.startswith("__")
        )

    def content_props(self) -> frozenset[str]:
        return frozenset()

    def library_schema(self) -> dict[str, Any]:
        if not self.available():
            return {}
        schema = _schema(str(self._schema_path))
        out: dict[str, Any] = {}
        for name, gql_type in schema.type_map.items():
            if name.startswith("__"):
                continue
            fields = getattr(gql_type, "fields", None)
            out[name] = sorted(fields) if fields else []
        return out

    def generate_system_prompt(self, **options: Any) -> str:
        return (
            "Emit a single GraphQL query valid against the shop schema; "
            "fields and arguments must exist on their parent types."
        )


assert isinstance(GraphQLQueryBackend(), GrammarBackend)
