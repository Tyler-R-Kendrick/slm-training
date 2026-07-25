"""Deterministic minimal witnesses for declared grammar alternatives."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from slm_training.data.contract import RuntimeSymbol
from slm_training.dsl.grammar_capabilities import (
    GrammarCapabilityAdapterV1,
    GrammarWitnessCandidateV1,
    ProductionAlternativeV1,
    ProductionOccurrenceV1,
    UnsupportedCapabilityV1,
    production_id,
)
from slm_training.dsl.language_contract import SymbolicSurfacePolicyV1
from slm_training.dsl.pack import DslPack


class UnexplainedAlternativeGap(RuntimeError):
    """A reachable/productive alternative has neither witness nor explanation."""


@dataclass(frozen=True, order=True)
class WitnessCostV1:
    ast_nodes: int
    productions: int
    optional_nodes: int
    markers: int
    surface_tokens: int


@dataclass(frozen=True)
class AlternativeWitnessV1:
    alternative_id: str
    start_symbol: str
    production: ProductionAlternativeV1
    source: str
    canonical_source: str
    focus_ast_path: tuple[int, ...]
    cost: WitnessCostV1
    runtime_symbols: tuple[Any, ...]


@dataclass(frozen=True)
class UnsupportedAlternativeV1:
    alternative_id: str
    start_symbol: str
    production: ProductionAlternativeV1
    reason: str
    status: str = "UNSUPPORTED"


@dataclass(frozen=True)
class MinimalWitnessBasisV1:
    pack_id: str
    authority_fingerprint: str
    seed: int
    witnesses: tuple[AlternativeWitnessV1, ...]
    unsupported: tuple[UnsupportedAlternativeV1, ...]

    @property
    def identity(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ast_nodes(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + sum(_ast_nodes(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return 1 + sum(_ast_nodes(child) for child in value)
    return 1


_SURFACE_TOKEN_RE = re.compile(
    r"[A-Za-z_$:@][A-Za-z0-9_.$:@]*|==|!=|>=|<=|&&|\|\||\S"
)


def _cost(
    pack: DslPack,
    source: str,
    trace: tuple[ProductionOccurrenceV1, ...],
    candidate: GrammarWitnessCandidateV1,
) -> WitnessCostV1:
    root = pack.backend.parse(source).root
    return WitnessCostV1(
        ast_nodes=_ast_nodes(root),
        productions=len(trace),
        optional_nodes=sum(
            occurrence.production.lhs.startswith("__")
            or not occurrence.production.rhs
            for occurrence in trace
        ),
        markers=len(candidate.runtime_symbols),
        surface_tokens=len(_SURFACE_TOKEN_RE.findall(source)),
    )


def _reachable_productive(
    adapter: GrammarCapabilityAdapterV1,
    start_symbol: str,
) -> tuple[ProductionAlternativeV1, ...]:
    productions = adapter.production_alternatives
    analysis = adapter.nonterminal_analysis
    if isinstance(productions, UnsupportedCapabilityV1) or isinstance(
        analysis, UnsupportedCapabilityV1
    ):
        raise UnexplainedAlternativeGap(
            f"{adapter.pack.pack_id}: grammar analysis is unsupported"
        )
    rows = {item.name: item for item in analysis}
    by_lhs: dict[str, list[ProductionAlternativeV1]] = {}
    for production in productions:
        by_lhs.setdefault(production.lhs, []).append(production)
    reachable = {start_symbol}
    pending = [start_symbol]
    while pending:
        current = pending.pop()
        for production in by_lhs.get(current, ()):
            for symbol in production.rhs:
                if symbol.kind == "nonterminal" and symbol.name not in reachable:
                    reachable.add(symbol.name)
                    pending.append(symbol.name)
    return tuple(
        production
        for production in productions
        if production.lhs in reachable
        and all(
            symbol.kind == "terminal"
            or (
                symbol.name in rows
                and rows[symbol.name].productive
            )
            for symbol in production.rhs
        )
    )


def _admit_candidate(
    pack: DslPack,
    adapter: GrammarCapabilityAdapterV1,
    start_symbol: str,
    candidate: GrammarWitnessCandidateV1,
) -> tuple[
    str,
    tuple[ProductionOccurrenceV1, ...],
    WitnessCostV1,
] | None:
    authority = adapter.authority
    if authority is None or authority.production_trace is None:
        raise UnexplainedAlternativeGap(
            f"{pack.pack_id}: production trace authority is unsupported"
        )
    try:
        adapter.fragment_parse(start_symbol, candidate.source)
        adapter.static_validate(candidate.source)
        if not adapter.scope_policy(candidate.source):
            return None
        SymbolicSurfacePolicyV1(pack.pack_id).require_admitted(
            candidate.source,
            runtime_symbols=candidate.runtime_symbols,
        )
        canonical = adapter.canonical_serialize(candidate.source)
        if not isinstance(canonical, str):
            return None
        adapter.static_validate(canonical)
        if not adapter.scope_policy(canonical):
            return None
        canonical_symbols = tuple(
            symbol
            for symbol in candidate.runtime_symbols
            if getattr(symbol, "role", None) != "state"
        ) + tuple(
            RuntimeSymbol(surface=surface, role="state")
            for surface in sorted(
                set(re.findall(r"\$[A-Za-z_][A-Za-z0-9_]*", canonical))
            )
        )
        SymbolicSurfacePolicyV1(pack.pack_id).require_admitted(
            canonical,
            runtime_symbols=canonical_symbols,
        )
        round_trip = adapter.canonical_serialize(canonical)
        if canonical != round_trip:
            return None
        if pack.backend.parse(canonical).root != pack.backend.parse(round_trip).root:
            return None
        trace = authority.production_trace(start_symbol, candidate.source)
        return canonical, trace, _cost(pack, candidate.source, trace, candidate)
    except Exception:  # noqa: BLE001 - a rejected candidate is not evidence
        return None


def generate_minimal_witness_basis(
    pack: DslPack, *, seed: int = 0
) -> MinimalWitnessBasisV1:
    """Select the exact minimum-cost admitted candidate for every alternative."""

    adapter = GrammarCapabilityAdapterV1(pack)
    authority = adapter.authority
    starts = adapter.start_symbols
    if (
        authority is None
        or authority.witness_candidates is None
        or isinstance(starts, UnsupportedCapabilityV1)
    ):
        raise UnexplainedAlternativeGap(
            f"{pack.pack_id}: witness candidate authority is unsupported"
        )
    candidates = tuple(authority.witness_candidates())
    admitted = {
        start: tuple(
            (candidate, result)
            for candidate in candidates
            if (
                result := _admit_candidate(pack, adapter, start, candidate)
            )
            is not None
        )
        for start in starts
    }
    unsupported_authority = dict(authority.unsupported_alternatives or {})
    witnesses: list[AlternativeWitnessV1] = []
    unsupported: list[UnsupportedAlternativeV1] = []
    for start in starts:
        for production in _reachable_productive(adapter, start):
            alternative_id = production_id(production)
            options = []
            for candidate, (canonical, trace, cost) in admitted[start]:
                paths = tuple(
                    occurrence.ast_path
                    for occurrence in trace
                    if occurrence.production == production
                )
                if paths:
                    options.append(
                        (
                            cost,
                            canonical,
                            candidate.source,
                            min(paths),
                            candidate,
                        )
                    )
            if options:
                cost, canonical, source, path, candidate = min(
                    options,
                    key=lambda option: (
                        *option[:4],
                        json.dumps(
                            [
                                (
                                    symbol.to_dict()
                                    if callable(getattr(symbol, "to_dict", None))
                                    else str(symbol)
                                )
                                for symbol in option[4].runtime_symbols
                            ],
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        ),
                    ),
                )
                witnesses.append(
                    AlternativeWitnessV1(
                        alternative_id=alternative_id,
                        start_symbol=start,
                        production=production,
                        source=source,
                        canonical_source=canonical,
                        focus_ast_path=path,
                        cost=cost,
                        runtime_symbols=candidate.runtime_symbols,
                    )
                )
            elif alternative_id in unsupported_authority:
                unsupported.append(
                    UnsupportedAlternativeV1(
                        alternative_id=alternative_id,
                        start_symbol=start,
                        production=production,
                        reason=unsupported_authority[alternative_id],
                    )
                )
            else:
                raise UnexplainedAlternativeGap(
                    f"{pack.pack_id}:{start}: no witness or typed unsupported "
                    f"reason for {alternative_id}"
                )
    return MinimalWitnessBasisV1(
        pack_id=pack.pack_id,
        authority_fingerprint=adapter.authority_fingerprints["combined"],
        seed=seed,
        witnesses=tuple(witnesses),
        unsupported=tuple(unsupported),
    )


__all__ = [
    "AlternativeWitnessV1",
    "MinimalWitnessBasisV1",
    "UnsupportedAlternativeV1",
    "UnexplainedAlternativeGap",
    "WitnessCostV1",
    "generate_minimal_witness_basis",
]
