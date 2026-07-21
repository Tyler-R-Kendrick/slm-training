"""Persistent typed lexical environments with surface aliases kept out of model state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping


class SymbolNamespace(str, Enum):
    """Disjoint symbol namespaces used by OpenUI and compiler consumers."""

    CONTENT = "content"
    BINDER = "binder"
    STATE = "state"
    QUERY = "query"
    ACTION = "action"
    MUTATION = "mutation"
    COMPILER_LOCAL = "compiler_local"


class ShadowingPolicy(str, Enum):
    """Whether a declaration may hide a visible alias in the same namespace."""

    FORBID = "forbid"
    ALLOW = "allow"


class ForwardReferencePolicy(str, Enum):
    """Whether an explicitly predeclared symbol may resolve before declaration."""

    FORBID = "forbid"
    ALLOW_PREDECLARED = "allow_predeclared"


@dataclass(frozen=True, order=True)
class StableSymbolId:
    """Opaque request-local identity allocated by namespace ordinal."""

    namespace: SymbolNamespace
    ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, SymbolNamespace):
            raise TypeError("namespace must be a SymbolNamespace")
        if self.ordinal < 0:
            raise ValueError("symbol ordinal must be non-negative")

    @property
    def canonical(self) -> str:
        return f"{self.namespace.value}:{self.ordinal:04d}"

    def to_dict(self) -> dict[str, Any]:
        return {"namespace": self.namespace.value, "ordinal": self.ordinal}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StableSymbolId:
        return cls(
            namespace=SymbolNamespace(str(data["namespace"])),
            ordinal=int(data["ordinal"]),
        )


@dataclass(frozen=True)
class ScopeSymbol:
    """Model-safe symbol facts; caller and template spellings are absent."""

    symbol_id: StableSymbolId
    declaration_frame_id: int
    semantic_type: str | None = None
    semantic_role: str | None = None

    def __post_init__(self) -> None:
        if self.declaration_frame_id < 0:
            raise ValueError("declaration_frame_id must be non-negative")
        for name in ("semantic_type", "semantic_role"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} must be non-empty when provided")

    @property
    def namespace(self) -> SymbolNamespace:
        return self.symbol_id.namespace

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "symbol_id": self.symbol_id.to_dict(),
            "declaration_frame_id": self.declaration_frame_id,
        }
        if self.semantic_type is not None:
            data["semantic_type"] = self.semantic_type
        if self.semantic_role is not None:
            data["semantic_role"] = self.semantic_role
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScopeSymbol:
        return cls(
            symbol_id=StableSymbolId.from_dict(data["symbol_id"]),
            declaration_frame_id=int(data["declaration_frame_id"]),
            semantic_type=_optional_text(data.get("semantic_type")),
            semantic_role=_optional_text(data.get("semantic_role")),
        )


@dataclass(frozen=True)
class SurfaceAlias:
    """Authority-side spelling for a stable symbol in one lexical frame."""

    alias: str
    symbol_id: StableSymbolId
    frame_id: int

    def __post_init__(self) -> None:
        if not self.alias.strip():
            raise ValueError("surface alias must be non-empty")
        if self.frame_id < 0:
            raise ValueError("alias frame_id must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "symbol_id": self.symbol_id.to_dict(),
            "frame_id": self.frame_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SurfaceAlias:
        return cls(
            alias=str(data["alias"]),
            symbol_id=StableSymbolId.from_dict(data["symbol_id"]),
            frame_id=int(data["frame_id"]),
        )


@dataclass(frozen=True)
class SurfaceAliasMap:
    """Persistent alias transport kept separate from model-visible scope facts."""

    entries: tuple[SurfaceAlias, ...] = ()

    def bind(self, entry: SurfaceAlias) -> SurfaceAliasMap:
        if any(item.symbol_id == entry.symbol_id for item in self.entries):
            raise ValueError(f"symbol {entry.symbol_id.canonical} already has an alias")
        if any(
            item.alias == entry.alias
            and item.frame_id == entry.frame_id
            and item.symbol_id.namespace == entry.symbol_id.namespace
            for item in self.entries
        ):
            raise ValueError(
                f"duplicate alias {entry.alias!r} in frame {entry.frame_id} "
                f"and namespace {entry.symbol_id.namespace.value}"
            )
        return SurfaceAliasMap((*self.entries, entry))

    def in_frame(
        self, alias: str, namespace: SymbolNamespace, frame_id: int
    ) -> SurfaceAlias | None:
        return next(
            (
                item
                for item in reversed(self.entries)
                if item.alias == alias
                and item.symbol_id.namespace is namespace
                and item.frame_id == frame_id
            ),
            None,
        )

    def for_symbol(self, symbol_id: StableSymbolId) -> SurfaceAlias | None:
        return next(
            (item for item in self.entries if item.symbol_id == symbol_id), None
        )

    def to_dict(self) -> dict[str, Any]:
        entries = sorted(
            self.entries,
            key=lambda item: (
                item.frame_id,
                item.symbol_id.namespace.value,
                item.symbol_id.ordinal,
                item.alias,
            ),
        )
        return {"entries": [entry.to_dict() for entry in entries]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SurfaceAliasMap:
        result = cls()
        for raw in data.get("entries", ()):
            result = result.bind(SurfaceAlias.from_dict(raw))
        return result

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class ScopeFrame:
    """One immutable lexical frame linked to its parent."""

    frame_id: int
    scope_id: str
    parent: ScopeFrame | None = None
    declarations: tuple[ScopeSymbol, ...] = ()

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if not self.scope_id.strip():
            raise ValueError("scope_id must be non-empty")
        if any(
            symbol.declaration_frame_id != self.frame_id for symbol in self.declarations
        ):
            raise ValueError("frame declarations must name their owning frame")
        ids = [symbol.symbol_id for symbol in self.declarations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate symbol declaration in frame")

    def chain(self) -> tuple[ScopeFrame, ...]:
        frames: list[ScopeFrame] = []
        current: ScopeFrame | None = self
        while current is not None:
            frames.append(current)
            current = current.parent
        return tuple(reversed(frames))


@dataclass(frozen=True)
class ScopeEnv:
    """Persistent typed environment whose fingerprint excludes surface aliases."""

    current: ScopeFrame = ScopeFrame(frame_id=0, scope_id="root")
    next_frame_id: int = 1
    next_ordinals: tuple[tuple[SymbolNamespace, int], ...] = ()
    predeclared: tuple[ScopeSymbol, ...] = ()

    def __post_init__(self) -> None:
        if self.next_frame_id <= max(frame.frame_id for frame in self.current.chain()):
            raise ValueError("next_frame_id must exceed every active frame id")
        ordinal_map = dict(self.next_ordinals)
        if len(ordinal_map) != len(self.next_ordinals):
            raise ValueError("duplicate namespace in next_ordinals")
        if any(not isinstance(namespace, SymbolNamespace) for namespace in ordinal_map):
            raise TypeError("next_ordinals keys must be SymbolNamespace values")
        if any(value < 0 for value in ordinal_map.values()):
            raise ValueError("next symbol ordinals must be non-negative")
        ids = [symbol.symbol_id for symbol in self.predeclared]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate predeclared symbol")
        active_symbols = [
            symbol for frame in self.current.chain() for symbol in frame.declarations
        ]
        active_frame_ids = {frame.frame_id for frame in self.current.chain()}
        if any(
            symbol.declaration_frame_id not in active_frame_ids
            for symbol in self.predeclared
        ):
            raise ValueError("predeclared symbol belongs to an inactive frame")
        all_ids = [symbol.symbol_id for symbol in (*active_symbols, *self.predeclared)]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("duplicate stable symbol id in environment")
        for symbol_id in all_ids:
            if ordinal_map.get(symbol_id.namespace, 0) <= symbol_id.ordinal:
                raise ValueError(
                    "next symbol ordinal must exceed every allocated symbol id"
                )

    def enter_scope(self, scope_id: str) -> ScopeEnv:
        frame = ScopeFrame(
            frame_id=self.next_frame_id,
            scope_id=scope_id,
            parent=self.current,
        )
        return replace(self, current=frame, next_frame_id=self.next_frame_id + 1)

    def exit_scope(self) -> ScopeEnv:
        if self.current.parent is None:
            raise ValueError("cannot exit the root scope")
        exited_frame_id = self.current.frame_id
        return replace(
            self,
            current=self.current.parent,
            predeclared=tuple(
                symbol
                for symbol in self.predeclared
                if symbol.declaration_frame_id != exited_frame_id
            ),
        )

    def validate_aliases(self, aliases: SurfaceAliasMap) -> None:
        """Fail closed unless every alias names active environment authority."""
        active_frames = {frame.frame_id for frame in self.current.chain()}
        symbols = {
            symbol.symbol_id: symbol
            for frame in self.current.chain()
            for symbol in frame.declarations
        }
        symbols.update({symbol.symbol_id: symbol for symbol in self.predeclared})
        for alias in aliases.entries:
            symbol = symbols.get(alias.symbol_id)
            if symbol is None:
                raise ValueError(f"alias {alias.alias!r} references an unknown symbol")
            if alias.frame_id not in active_frames:
                raise ValueError(f"alias {alias.alias!r} references an inactive frame")
            if alias.frame_id != symbol.declaration_frame_id:
                raise ValueError(f"alias {alias.alias!r} frame does not match symbol")

    def predeclare(
        self,
        namespace: SymbolNamespace,
        aliases: SurfaceAliasMap,
        *,
        alias: str | None,
        shadowing: ShadowingPolicy,
        semantic_type: str | None = None,
        semantic_role: str | None = None,
    ) -> tuple[ScopeEnv, SurfaceAliasMap, ScopeSymbol]:
        self._check_shadowing(namespace, aliases, alias, shadowing)
        env, symbol = self._allocate_symbol(
            namespace,
            semantic_type=semantic_type,
            semantic_role=semantic_role,
        )
        env = replace(env, predeclared=(*env.predeclared, symbol))
        return env, _bind_alias(aliases, alias, symbol), symbol

    def declare(
        self,
        namespace: SymbolNamespace,
        aliases: SurfaceAliasMap,
        *,
        alias: str | None,
        shadowing: ShadowingPolicy,
        semantic_type: str | None = None,
        semantic_role: str | None = None,
        predeclared_id: StableSymbolId | None = None,
    ) -> tuple[ScopeEnv, SurfaceAliasMap, ScopeSymbol]:
        if not isinstance(shadowing, ShadowingPolicy):
            raise TypeError("shadowing must be an explicit ShadowingPolicy")
        if predeclared_id is None:
            self._check_shadowing(namespace, aliases, alias, shadowing)
            env, symbol = self._allocate_symbol(
                namespace,
                semantic_type=semantic_type,
                semantic_role=semantic_role,
            )
            aliases = _bind_alias(aliases, alias, symbol)
        else:
            symbol = next(
                (item for item in self.predeclared if item.symbol_id == predeclared_id),
                None,
            )
            if symbol is None:
                raise ValueError(
                    f"unknown predeclared symbol {predeclared_id.canonical}"
                )
            if symbol.namespace is not namespace:
                raise ValueError("predeclared symbol namespace mismatch")
            if symbol.declaration_frame_id != self.current.frame_id:
                raise ValueError("predeclared symbol belongs to a different frame")
            if semantic_type is not None and semantic_type != symbol.semantic_type:
                raise ValueError("predeclared symbol semantic_type mismatch")
            if semantic_role is not None and semantic_role != symbol.semantic_role:
                raise ValueError("predeclared symbol semantic_role mismatch")
            bound = aliases.for_symbol(predeclared_id)
            if alias is not None and (bound is None or bound.alias != alias):
                raise ValueError("predeclared symbol alias mismatch")
            env = replace(
                self,
                predeclared=tuple(
                    item
                    for item in self.predeclared
                    if item.symbol_id != predeclared_id
                ),
            )
        frame = replace(env.current, declarations=(*env.current.declarations, symbol))
        return replace(env, current=frame), aliases, symbol

    def resolve(
        self,
        alias: str,
        namespace: SymbolNamespace,
        aliases: SurfaceAliasMap,
        *,
        forward_references: ForwardReferencePolicy,
    ) -> ScopeSymbol:
        if not isinstance(forward_references, ForwardReferencePolicy):
            raise TypeError(
                "forward_references must be an explicit ForwardReferencePolicy"
            )
        for frame in reversed(self.current.chain()):
            entry = aliases.in_frame(alias, namespace, frame.frame_id)
            if entry is None:
                continue
            declared = next(
                (
                    symbol
                    for symbol in frame.declarations
                    if symbol.symbol_id == entry.symbol_id
                ),
                None,
            )
            if declared is not None:
                return declared
            reserved = next(
                (
                    symbol
                    for symbol in self.predeclared
                    if symbol.symbol_id == entry.symbol_id
                ),
                None,
            )
            if (
                reserved is not None
                and forward_references is ForwardReferencePolicy.ALLOW_PREDECLARED
            ):
                return reserved
            raise LookupError(
                f"forward reference to {alias!r} is not allowed in "
                f"namespace {namespace.value}"
            )
        raise LookupError(f"unknown {namespace.value} alias {alias!r}")

    def visible(
        self, namespace: SymbolNamespace, aliases: SurfaceAliasMap
    ) -> tuple[ScopeSymbol, ...]:
        visible: list[ScopeSymbol] = []
        hidden_aliases: set[str] = set()
        for frame in reversed(self.current.chain()):
            for symbol in frame.declarations:
                if symbol.namespace is not namespace:
                    continue
                entry = aliases.for_symbol(symbol.symbol_id)
                if entry is not None and entry.alias in hidden_aliases:
                    continue
                visible.append(symbol)
                if entry is not None:
                    hidden_aliases.add(entry.alias)
        return tuple(visible)

    def _check_shadowing(
        self,
        namespace: SymbolNamespace,
        aliases: SurfaceAliasMap,
        alias: str | None,
        policy: ShadowingPolicy,
    ) -> None:
        if not isinstance(policy, ShadowingPolicy):
            raise TypeError("shadowing must be an explicit ShadowingPolicy")
        if alias is None:
            return
        for frame in reversed(self.current.chain()):
            if aliases.in_frame(alias, namespace, frame.frame_id) is None:
                continue
            if (
                frame.frame_id == self.current.frame_id
                or policy is ShadowingPolicy.FORBID
            ):
                raise ValueError(
                    f"alias {alias!r} already declared in namespace {namespace.value}"
                )
            return

    def _allocate_symbol(
        self,
        namespace: SymbolNamespace,
        *,
        semantic_type: str | None,
        semantic_role: str | None,
    ) -> tuple[ScopeEnv, ScopeSymbol]:
        if not isinstance(namespace, SymbolNamespace):
            raise TypeError("namespace must be a SymbolNamespace")
        ordinals = dict(self.next_ordinals)
        ordinal = ordinals.get(namespace, 0)
        ordinals[namespace] = ordinal + 1
        symbol = ScopeSymbol(
            symbol_id=StableSymbolId(namespace, ordinal),
            declaration_frame_id=self.current.frame_id,
            semantic_type=semantic_type,
            semantic_role=semantic_role,
        )
        return (
            replace(
                self,
                next_ordinals=tuple(
                    sorted(ordinals.items(), key=lambda item: item[0].value)
                ),
            ),
            symbol,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": [
                {
                    "frame_id": frame.frame_id,
                    "scope_id": frame.scope_id,
                    "declarations": [symbol.to_dict() for symbol in frame.declarations],
                }
                for frame in self.current.chain()
            ],
            "next_frame_id": self.next_frame_id,
            "next_ordinals": {
                namespace.value: ordinal
                for namespace, ordinal in sorted(
                    self.next_ordinals, key=lambda item: item[0].value
                )
            },
            "predeclared": [
                symbol.to_dict()
                for symbol in sorted(
                    self.predeclared,
                    key=lambda item: (
                        item.symbol_id.namespace.value,
                        item.symbol_id.ordinal,
                    ),
                )
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScopeEnv:
        parent: ScopeFrame | None = None
        for raw in data.get("frames", ()):
            parent = ScopeFrame(
                frame_id=int(raw["frame_id"]),
                scope_id=str(raw["scope_id"]),
                parent=parent,
                declarations=tuple(
                    ScopeSymbol.from_dict(item) for item in raw.get("declarations", ())
                ),
            )
        if parent is None:
            raise ValueError("serialized ScopeEnv must contain at least one frame")
        return cls(
            current=parent,
            next_frame_id=int(data["next_frame_id"]),
            next_ordinals=tuple(
                sorted(
                    (
                        (SymbolNamespace(str(namespace)), int(ordinal))
                        for namespace, ordinal in dict(
                            data.get("next_ordinals", {})
                        ).items()
                    ),
                    key=lambda item: item[0].value,
                )
            ),
            predeclared=tuple(
                ScopeSymbol.from_dict(item) for item in data.get("predeclared", ())
            ),
        )

    @property
    def fingerprint(self) -> str:
        data = self.to_dict()
        for frame in data["frames"]:
            frame["scope_id"] = f"scope:{frame['frame_id']}"
        return _fingerprint(data)


def _bind_alias(
    aliases: SurfaceAliasMap, alias: str | None, symbol: ScopeSymbol
) -> SurfaceAliasMap:
    if alias is None:
        return aliases
    return aliases.bind(
        SurfaceAlias(
            alias=alias,
            symbol_id=symbol.symbol_id,
            frame_id=symbol.declaration_frame_id,
        )
    )


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _fingerprint(data: Mapping[str, Any]) -> str:
    payload = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "ForwardReferencePolicy",
    "ScopeEnv",
    "ScopeFrame",
    "ScopeSymbol",
    "ShadowingPolicy",
    "StableSymbolId",
    "SurfaceAlias",
    "SurfaceAliasMap",
    "SymbolNamespace",
]
