"""Declared grammar capabilities exposed through :class:`DslPack`.

The adapter deliberately has no example/corpus fallback. A pack either declares
an authority for a capability or receives a typed ``UNSUPPORTED`` result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from lark import Lark, Tree
from lark.grammar import Terminal

if TYPE_CHECKING:
    from slm_training.dsl.pack import DslPack


@dataclass(frozen=True)
class UnsupportedCapabilityV1:
    capability: str
    reason: str
    status: str = "UNSUPPORTED"


@dataclass(frozen=True)
class GrammarSymbolV1:
    name: str
    kind: str


@dataclass(frozen=True)
class ProductionAlternativeV1:
    lhs: str
    rhs: tuple[GrammarSymbolV1, ...]


@dataclass(frozen=True)
class TerminalCategoryV1:
    name: str
    kind: str
    pattern: str


@dataclass(frozen=True)
class NonterminalAnalysisV1:
    name: str
    reachable: bool
    productive: bool
    nullable: bool
    recursive: bool


@dataclass(frozen=True)
class ProductionOccurrenceV1:
    production: ProductionAlternativeV1
    ast_path: tuple[int, ...]


@dataclass(frozen=True)
class GrammarWitnessCandidateV1:
    source: str
    runtime_symbols: tuple[Any, ...] = ()


@dataclass(frozen=True)
class GrammarCapabilityAuthorityV1:
    """Pack-owned declarations; grammar structure must not come from examples."""

    start_symbols: tuple[str, ...] | None = None
    productions: tuple[ProductionAlternativeV1, ...] | None = None
    terminal_categories: tuple[TerminalCategoryV1, ...] | None = None
    fragment_parse: Callable[[str, str], Any] | None = None
    canonical_serialize: Callable[[str], str] | None = None
    static_validate: Callable[[str], Any] | None = None
    scope_policy: Callable[[str], Any] | None = None
    completion_frontier: Callable[[str], frozenset[str]] | None = None
    production_trace: Callable[[str, str], tuple[ProductionOccurrenceV1, ...]] | None = None
    witness_candidates: Callable[[], Iterable[GrammarWitnessCandidateV1]] | None = None
    unsupported_alternatives: Mapping[str, str] | None = None


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def production_id(production: ProductionAlternativeV1) -> str:
    payload = {
        "lhs": str(production.lhs),
        "rhs": [
            {"kind": symbol.kind, "name": str(symbol.name)}
            for symbol in production.rhs
        ],
    }
    return f"{production.lhs}:{_canonical_hash(payload)[:16]}"


def _unsupported(capability: str, pack_id: str) -> UnsupportedCapabilityV1:
    return UnsupportedCapabilityV1(
        capability=capability,
        reason=f"DSL pack {pack_id!r} does not declare {capability!r}",
    )


def _analysis(
    starts: Sequence[str], productions: Sequence[ProductionAlternativeV1]
) -> tuple[NonterminalAnalysisV1, ...]:
    names = {production.lhs for production in productions}
    edges = {
        name: {
            symbol.name
            for production in productions
            if production.lhs == name
            for symbol in production.rhs
            if symbol.kind == "nonterminal" and symbol.name in names
        }
        for name in names
    }
    reachable = set(starts) & names
    frontier = list(reachable)
    while frontier:
        current = frontier.pop()
        for target in edges[current] - reachable:
            reachable.add(target)
            frontier.append(target)

    nullable: set[str] = set()
    productive: set[str] = set()
    changed = True
    while changed:
        changed = False
        for production in productions:
            if (
                production.lhs not in nullable
                and all(
                    symbol.kind == "nonterminal" and symbol.name in nullable
                    for symbol in production.rhs
                )
            ):
                nullable.add(production.lhs)
                changed = True
            if (
                production.lhs not in productive
                and all(
                    symbol.kind == "terminal" or symbol.name in productive
                    for symbol in production.rhs
                )
            ):
                productive.add(production.lhs)
                changed = True

    def recursive(name: str) -> bool:
        pending = list(edges[name])
        seen: set[str] = set()
        while pending:
            target = pending.pop()
            if target == name:
                return True
            if target not in seen:
                seen.add(target)
                pending.extend(edges.get(target, ()))
        return False

    return tuple(
        NonterminalAnalysisV1(
            name=name,
            reachable=name in reachable,
            productive=name in productive,
            nullable=name in nullable,
            recursive=recursive(name),
        )
        for name in sorted(names)
    )


class GrammarCapabilityAdapterV1:
    """Read exact grammar capabilities from one resolved :class:`DslPack`."""

    def __init__(self, pack: DslPack) -> None:
        self.pack = pack
        self.authority = pack.grammar_capability_authority

    def _declared(self, field: str, capability: str) -> Any:
        if self.authority is None:
            return _unsupported(capability, self.pack.pack_id)
        value = getattr(self.authority, field)
        return (
            value
            if value is not None
            else _unsupported(capability, self.pack.pack_id)
        )

    @property
    def start_symbols(self) -> tuple[str, ...] | UnsupportedCapabilityV1:
        return self._declared("start_symbols", "start_symbols")

    @property
    def production_alternatives(
        self,
    ) -> tuple[ProductionAlternativeV1, ...] | UnsupportedCapabilityV1:
        return self._declared("productions", "production_alternatives")

    @property
    def terminal_categories(
        self,
    ) -> tuple[TerminalCategoryV1, ...] | UnsupportedCapabilityV1:
        return self._declared("terminal_categories", "terminal_categories")

    @property
    def nonterminal_analysis(
        self,
    ) -> tuple[NonterminalAnalysisV1, ...] | UnsupportedCapabilityV1:
        starts = self.start_symbols
        productions = self.production_alternatives
        if isinstance(starts, UnsupportedCapabilityV1) or isinstance(
            productions, UnsupportedCapabilityV1
        ):
            return _unsupported("nonterminal_analysis", self.pack.pack_id)
        return _analysis(starts, productions)

    def fragment_parse(
        self, start_symbol: str, source: str
    ) -> Any | UnsupportedCapabilityV1:
        parser = self._declared("fragment_parse", "fragment_parse")
        if isinstance(parser, UnsupportedCapabilityV1):
            return parser
        return parser(start_symbol, source)

    def canonical_serialize(self, source: str) -> str | UnsupportedCapabilityV1:
        serializer = self._declared("canonical_serialize", "canonical_serialize")
        if isinstance(serializer, UnsupportedCapabilityV1):
            return serializer
        return serializer(source)

    def static_validate(self, source: str) -> Any | UnsupportedCapabilityV1:
        validator = self._declared("static_validate", "static_validate")
        if isinstance(validator, UnsupportedCapabilityV1):
            return validator
        return validator(source)

    def scope_policy(self, source: str) -> Any | UnsupportedCapabilityV1:
        policy = self._declared("scope_policy", "scope_policy")
        if isinstance(policy, UnsupportedCapabilityV1):
            return policy
        return policy(source)

    def completion_frontier(
        self, prefix: str
    ) -> frozenset[str] | UnsupportedCapabilityV1:
        frontier = self._declared("completion_frontier", "completion_frontier")
        if isinstance(frontier, UnsupportedCapabilityV1):
            return frontier
        return frontier(prefix)

    @property
    def is_complete(self) -> bool:
        if self.authority is None:
            return False
        return bool(
            self.authority.start_symbols
            and self.authority.productions
            and self.authority.terminal_categories
        ) and all(
            getattr(self.authority, field) is not None
            for field in (
                "fragment_parse",
                "canonical_serialize",
                "static_validate",
                "scope_policy",
                "completion_frontier",
            )
        )

    @property
    def authority_fingerprints(self) -> Mapping[str, str]:
        backend = self.pack.backend
        placeholder = self.pack.placeholder_policy
        values = {
            "grammar": {
                "starts": self.start_symbols,
                "productions": self.production_alternatives,
                "terminals": self.terminal_categories,
            },
            "backend": {
                "class": f"{type(backend).__module__}.{type(backend).__qualname__}",
                "info": backend.info,
                "available": backend.available(),
            },
            "schema": backend.library_schema(),
            "property_order": (
                self.pack.prop_order() if self.pack.prop_order is not None else None
            ),
            "placeholder": {
                "pattern": placeholder.placeholder_re.pattern,
                "flags": placeholder.placeholder_re.flags,
                "content_props": sorted(placeholder.content_props),
            },
        }
        fingerprints = {name: _canonical_hash(value) for name, value in values.items()}
        fingerprints["combined"] = _canonical_hash(fingerprints)
        return fingerprints


def lark_authority(
    *,
    grammar_path: Path,
    start_symbols: Sequence[str],
    canonical_serialize: Callable[[str], str],
    static_validate: Callable[[str], Any],
    scope_policy: Callable[[str], Any],
    completion_frontier: Callable[[str], frozenset[str]],
    witness_candidates: Callable[[], Iterable[GrammarWitnessCandidateV1]]
    | None = None,
    unsupported_alternatives: Mapping[str, str] | None = None,
) -> GrammarCapabilityAuthorityV1:
    """Build authority from a declared Lark grammar file."""

    grammar = Path(grammar_path).read_text(encoding="utf-8")
    parser = Lark(
        grammar,
        start=list(start_symbols),
        parser="lalr",
        maybe_placeholders=False,
    )
    productions = tuple(
        ProductionAlternativeV1(
            lhs=str(rule.origin.name),
            rhs=tuple(
                GrammarSymbolV1(
                    name=str(symbol.name),
                    kind=(
                        "terminal" if isinstance(symbol, Terminal) else "nonterminal"
                    ),
                )
                for symbol in rule.expansion
            ),
        )
        for rule in parser.rules
    )
    terminals = tuple(
        TerminalCategoryV1(
            name=str(terminal.name),
            kind=terminal.pattern.type,
            pattern=terminal.pattern.value,
        )
        for terminal in parser.terminals
    )

    def parse_fragment(start_symbol: str, source: str) -> Any:
        if start_symbol not in start_symbols:
            raise ValueError(
                f"undeclared start symbol {start_symbol!r}; "
                f"declared={sorted(start_symbols)!r}"
            )
        text = source if source.endswith("\n") else source + "\n"
        return parser.parse(text, start=start_symbol)

    def trace_productions(
        start_symbol: str, source: str
    ) -> tuple[ProductionOccurrenceV1, ...]:
        if start_symbol not in start_symbols:
            raise ValueError(
                f"undeclared start symbol {start_symbol!r}; "
                f"declared={sorted(start_symbols)!r}"
            )
        traced = Lark(
            grammar,
            start=list(start_symbols),
            parser="lalr",
            maybe_placeholders=False,
        )
        callbacks = traced.parser.parser.parser.callbacks
        occurrences: list[tuple[ProductionAlternativeV1, int]] = []
        for rule, callback in tuple(callbacks.items()):
            production = ProductionAlternativeV1(
                lhs=str(rule.origin.name),
                rhs=tuple(
                    GrammarSymbolV1(
                        name=str(symbol.name),
                        kind=(
                            "terminal"
                            if isinstance(symbol, Terminal)
                            else "nonterminal"
                        ),
                    )
                    for symbol in rule.expansion
                ),
            )

            def record(
                children: list[Any],
                *,
                item: ProductionAlternativeV1 = production,
                inner: Callable[[list[Any]], Any] = callback,
            ) -> Any:
                result = inner(children)
                occurrences.append((item, id(result)))
                return result

            callbacks[rule] = record

        tree = traced.parse(source, start=start_symbol)
        paths: dict[int, tuple[int, ...]] = {}

        def visit(node: Any, path: tuple[int, ...]) -> None:
            paths.setdefault(id(node), path)
            if isinstance(node, Tree):
                for index, child in enumerate(node.children):
                    visit(child, (*path, index))
            elif isinstance(node, (list, tuple)):
                for index, child in enumerate(node):
                    visit(child, (*path, index))

        visit(tree, ())
        return tuple(
            ProductionOccurrenceV1(production=item, ast_path=paths.get(identity, ()))
            for item, identity in occurrences
        )

    return GrammarCapabilityAuthorityV1(
        start_symbols=tuple(start_symbols),
        productions=productions,
        terminal_categories=terminals,
        fragment_parse=parse_fragment,
        canonical_serialize=canonical_serialize,
        static_validate=static_validate,
        scope_policy=scope_policy,
        completion_frontier=completion_frontier,
        production_trace=trace_productions,
        witness_candidates=witness_candidates,
        unsupported_alternatives=unsupported_alternatives,
    )


__all__ = [
    "GrammarCapabilityAdapterV1",
    "GrammarCapabilityAuthorityV1",
    "GrammarSymbolV1",
    "GrammarWitnessCandidateV1",
    "NonterminalAnalysisV1",
    "ProductionOccurrenceV1",
    "ProductionAlternativeV1",
    "TerminalCategoryV1",
    "UnsupportedCapabilityV1",
    "lark_authority",
    "production_id",
]
