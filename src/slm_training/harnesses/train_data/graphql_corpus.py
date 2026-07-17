"""F2 (SLM-43): deterministic typed-query generator for the GraphQL pack.

The generator walks the schema-as-symbol-table (``schema_symbols``) and emits
selection sets whose every field provably exists on its parent type — typed-AST
generation in the pack-contract sense: outputs pass the pack's own validity
oracle by construction, and the prompt names the selected fields so content
coverage is checkable.
"""

from __future__ import annotations

from slm_training.dsl.schema import ExampleRecord


def _selection(
    types: dict[str, list[str]],
    field_types: dict[str, dict[str, str]],
    type_name: str,
    depth: int,
) -> list[str]:
    lines: list[str] = []
    for field in types.get(type_name, []):
        child = field_types.get(type_name, {}).get(field)
        if child and child in types and depth > 0:
            inner = _selection(types, field_types, child, depth - 1)
            if inner:
                lines.append(f"{field} {{")
                lines.extend(f"  {line}" for line in inner)
                lines.append("}")
            continue
        if not child or child not in types:
            lines.append(field)
    return lines


def build_graphql_corpus(
    *,
    root_id: str,
    schema_sdl: str | None = None,
    split: str = "train",
    depth: int = 1,
    max_records: int = 32,
) -> list[ExampleRecord]:
    """One record per Query root field, selection sets resolved to ``depth``."""
    from slm_training.dsl.grammar.backends.graphql_js import (
        default_schema_sdl,
        schema_symbols,
    )

    sdl = schema_sdl or default_schema_sdl()
    types = schema_symbols(sdl)
    field_types = _parse_field_types(sdl)
    required_args = _parse_required_args(sdl)
    query_fields = types.get("Query", [])
    records: list[ExampleRecord] = []
    for field in query_fields[:max_records]:
        child = field_types.get("Query", {}).get(field)
        # Required arguments are satisfied through variables — GraphQL's
        # routed-content channel (values never appear as literals).
        args = required_args.get("Query", {}).get(field, {})
        variables = {name: f"${field}_{name.lstrip('$')}" for name in args}
        header = (
            "query("
            + ", ".join(f"{variables[n]}: {t}" for n, t in args.items())
            + ")"
            if args
            else "query"
        )
        call = (
            f"{field}(" + ", ".join(f"{n}: {variables[n]}" for n in args) + ")"
            if args
            else field
        )
        if child and child in types:
            inner = _selection(types, field_types, child, depth)
            body = "\n".join(f"    {line}" for line in inner)
            query = f"{header} {{\n  {call} {{\n{body}\n  }}\n}}"
            selected = ", ".join(types[child])
        else:
            query = f"{header} {{\n  {call}\n}}"
            selected = field
        records.append(
            ExampleRecord(
                id=f"{root_id}-{field}",
                prompt=f"Fetch {field} with {selected}.",
                openui=query,
                placeholders=sorted(variables.values()),
                split=split,
                source="graphql-typed-generator",
                meta={"dsl": "graphql", "root_field": field, "depth": depth},
            )
        )
    return records


def _parse_field_types(sdl: str) -> dict[str, dict[str, str]]:
    """Field → named result type, from the SDL text (wrappers stripped).

    A deliberately small SDL reader for generator use only — legality is
    always re-certified by graphql-js, never by this parser.
    """
    import re

    field_types: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw in sdl.splitlines():
        line = raw.strip()
        m = re.match(r"^(?:type|interface)\s+(\w+)", line)
        if m:
            current = m.group(1)
            field_types[current] = {}
            continue
        if line.startswith("}"):
            current = None
            continue
        if current:
            fm = re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+?)\s*$", line)
            if fm:
                named = re.sub(r"[\[\]!\s]", "", fm.group(2))
                field_types[current][fm.group(1)] = named
    return field_types


def _parse_required_args(sdl: str) -> dict[str, dict[str, dict[str, str]]]:
    """type → field → {arg name: SDL type} for non-null (required) args."""
    import re

    out: dict[str, dict[str, dict[str, str]]] = {}
    current: str | None = None
    for raw in sdl.splitlines():
        line = raw.strip()
        m = re.match(r"^(?:type|interface)\s+(\w+)", line)
        if m:
            current = m.group(1)
            out[current] = {}
            continue
        if line.startswith("}"):
            current = None
            continue
        if current:
            fm = re.match(r"^(\w+)\s*\(([^)]*)\)\s*:", line)
            if fm:
                args: dict[str, str] = {}
                for part in fm.group(2).split(","):
                    am = re.match(r"^\s*(\w+)\s*:\s*(\S+!)\s*$", part)
                    if am:
                        args[am.group(1)] = am.group(2)
                if args:
                    out[current][fm.group(1)] = args
    return out


__all__ = ["build_graphql_corpus"]
