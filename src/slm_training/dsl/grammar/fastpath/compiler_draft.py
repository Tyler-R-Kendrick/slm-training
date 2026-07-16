"""Compiler-drafted semantic completions for constrained TwoTower decode."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Literal

from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.force_emit import force_next_token_id
from slm_training.dsl.grammar.fastpath.token_map import allowed_id_set

Coverage = Literal["complete", "partial", "none"]
_NAME_BEFORE_PAREN = re.compile(r"([A-Z][A-Za-z0-9_]*)\s*$")


@dataclass(frozen=True)
class CompletionPath:
    """One compiler-valid semantic action plus its maximal forced suffix."""

    token_ids: tuple[int, ...]
    kind: str


@dataclass(frozen=True)
class CompletionForest:
    """All known next actions for a prefix and their coverage guarantee."""

    paths: tuple[CompletionPath, ...]
    coverage: Coverage
    terminals: tuple[str, ...] = ()

    @property
    def candidate_ids(self) -> tuple[int, ...]:
        return tuple(path.token_ids[0] for path in self.paths if path.token_ids)


def _token_piece(tokenizer: Any, token_id: int) -> str:
    raw = tokenizer.id_to_token.get(int(token_id), "")
    if raw == "NL":
        return "\n"
    decoded = tokenizer.decode([int(token_id)])
    if decoded or raw.startswith(("<BIND_", "<SYM_", "<STATE_")):
        return decoded
    return raw


def _semantic_kind(tokenizer: Any, token_id: int) -> str:
    kind_of = getattr(tokenizer, "kind_of", None)
    if callable(kind_of):
        try:
            return str(kind_of(int(token_id)).value)
        except Exception:  # noqa: BLE001
            pass
    piece = _token_piece(tokenizer, token_id)
    if piece[:1].isupper() and piece.isidentifier():
        return "component"
    if piece[:1].islower() and piece.isidentifier():
        return "binder"
    if piece.startswith(":") or piece.startswith("<SYM_"):
        return "symbol"
    return "structural"


def _known_terminal_coverage(tokenizer: Any, terminals: frozenset[str]) -> bool:
    """Whether token mapping exhausts the model vocabulary for these terminals."""
    broad = {
        "NAME",
        "COMPONENT",
        "STATE_NAME",
        "BUILTIN",
        "STRING",
        "NUMBER",
        "BOOL",
        "NULL",
    }
    # Lexical/BPE candidates are only a subset of semantic names. They remain
    # valid drafts, but only lexer-native symbol/action ids are exhaustive.
    if terminals & broad and not callable(getattr(tokenizer, "kind_of", None)):
        return False
    known = {
        "$END",
        "COMMENT",
        "WS_INLINE",
        "_NL",
        "NL",
        "NAME",
        "COMPONENT",
        "STATE_NAME",
        "BUILTIN",
        "STRING",
        "NUMBER",
        "BOOL",
        "NULL",
        "EQUAL",
        "LPAR",
        "RPAR",
        "LSQB",
        "RSQB",
        "LBRACE",
        "RBRACE",
        "COMMA",
        "DOT",
        "COLON",
        "QMARK",
        "PLUS",
        "MINUS",
        "STAR",
        "SLASH",
        "PERCENT",
        "BANG",
        "MORETHAN",
        "LESSTHAN",
        "__ANON_0",
        "__ANON_1",
        "__ANON_2",
        "__ANON_3",
        "__ANON_4",
        "__ANON_5",
    }
    vocab = getattr(tokenizer, "token_to_id", {})
    return all(term in known or term in vocab for term in terminals)


@lru_cache(maxsize=1)
def _official_schema() -> dict[str, Any] | None:
    try:
        from slm_training.dsl import lang_core

        if lang_core.bridge_available():
            return lang_core.library_schema()
    except Exception:  # noqa: BLE001
        pass
    return None


def _active_call(prefix_text: str) -> tuple[str, int] | None:
    """Return innermost component and positional property index."""
    stack: list[int] = []
    quoted = False
    escaped = False
    for i, char in enumerate(prefix_text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quoted:
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
            continue
        if quoted:
            continue
        if char == "(":
            stack.append(i)
        elif char == ")" and stack:
            stack.pop()
    if not stack:
        return None
    open_at = stack[-1]
    match = _NAME_BEFORE_PAREN.search(prefix_text[:open_at])
    if match is None:
        return None
    tail = prefix_text[open_at + 1 :]
    depth = 0
    quoted = False
    commas = 0
    for char in tail:
        if char == '"':
            quoted = not quoted
        elif not quoted and char in "([{":
            depth += 1
        elif not quoted and char in ")]}" and depth:
            depth -= 1
        elif not quoted and char == "," and depth == 0:
            commas += 1
    return match.group(1), commas


def _schema_enum_ids(
    tokenizer: Any, prefix_text: str, schema: dict[str, Any]
) -> set[int] | None:
    active = _active_call(prefix_text)
    if active is None:
        return None
    component, index = active
    definition = (schema.get("$defs") or {}).get(component) or {}
    properties = definition.get("properties") or {}
    names = list(properties)
    if index >= len(names):
        return set()
    values = (properties.get(names[index]) or {}).get("enum")
    if not values:
        return None
    ids = set()
    for value in values:
        for key in (value, f"STR:{value}"):
            token_id = tokenizer.token_to_id.get(key)
            if token_id is not None:
                ids.add(int(token_id))
                break
    return ids


def build_completion_forest(
    tokenizer: Any,
    prefix_ids: list[int],
    *,
    state: Any | None = None,
    slot_contract: list[str] | None = None,
    max_path_tokens: int = 8,
) -> CompletionForest:
    """Enumerate every mapped, globally extendable action at ``prefix_ids``.

    Lark supplies parser reachability, the lexer-native tokenizer supplies the
    compiler-derived component/binder/symbol spaces, and the optional slot
    contract restricts active placeholder symbols. Each branch is extended
    through its maximal deterministic grammar suffix.
    """
    engine = getattr(state, "engine", None) if state is not None else None
    if not isinstance(engine, OpenUIIncrementalEngine):
        engine = OpenUIIncrementalEngine()
    if state is not None:
        prefix_text = state.sync_ids(tokenizer, prefix_ids)
    else:
        prefix_text = tokenizer.decode(prefix_ids)
    if not engine.set_prefix(prefix_text) and prefix_text.strip():
        return CompletionForest((), "none")

    terminals = engine.next_terminals()
    candidates = allowed_id_set(tokenizer, terminals) or set()
    if "$END" in terminals:
        candidates.add(int(tokenizer.eos_id))
    needs_schema = bool(terminals & {"COMPONENT", "STRING"})
    schema = _official_schema() if needs_schema else None
    if schema is not None and "COMPONENT" in terminals:
        component_names = set(schema.get("properties") or {})
        candidates = {
            token_id
            for token_id in candidates
            if _semantic_kind(tokenizer, token_id) != "component"
            or _token_piece(tokenizer, token_id) in component_names
        }
    enum_ids = _schema_enum_ids(tokenizer, prefix_text, schema) if schema else None
    if enum_ids is not None:
        candidates = set(enum_ids)

    if slot_contract and "STRING" in terminals:
        try:
            from slm_training.models.grammar import contract_allowed_token_ids

            contract_ids = contract_allowed_token_ids(
                tokenizer, prefix_ids, slot_contract
            )
        except Exception:  # noqa: BLE001
            contract_ids = None
        if contract_ids is not None:
            kind_ids = getattr(tokenizer, "kind_ids", None)
            if callable(kind_ids):
                symbol_space = set(kind_ids("sym"))
                candidates = (candidates - symbol_space) | set(contract_ids)
            else:
                overlap = candidates & contract_ids
                candidates = overlap if overlap else set(contract_ids)

    inventory_complete = not (needs_schema and schema is None)
    kind_of = getattr(tokenizer, "kind_of", None)
    if callable(kind_of):
        try:
            from slm_training.models.dsl_tokenizer import TokenKind

            bind_ids = set(tokenizer.kind_ids(TokenKind.BIND))
            state_ids = set(tokenizer.kind_ids(TokenKind.STATE))
            builtin_ids = set(tokenizer.kind_ids(TokenKind.BUILTIN))
            sym_ids = set(tokenizer.kind_ids(TokenKind.SYM))
            visible_binds = [tid for tid in prefix_ids if tid in bind_ids]
            last = prefix_ids[-1] if prefix_ids else None
            at_statement_start = not prefix_ids or tokenizer.id_to_token.get(last) == "NL"
            if at_statement_start:
                next_slot = min(len(set(visible_binds)), max(0, tokenizer.bind_slots - 1))
                candidates -= bind_ids
                candidates.add(int(tokenizer.bind_id(next_slot)))
            elif visible_binds:
                candidates = (candidates - bind_ids) | (candidates & set(visible_binds))
            else:
                candidates -= bind_ids
            # The selected 0.2.x layout contract excludes state/effect actions.
            candidates -= state_ids | builtin_ids
            if not slot_contract and candidates & sym_ids:
                candidates -= sym_ids
                inventory_complete = False
        except Exception:  # noqa: BLE001
            inventory_complete = False

    specials = {
        int(tokenizer.pad_id),
        int(tokenizer.mask_id),
        int(tokenizer.bos_id),
        int(tokenizer.unk_id),
    }
    paths: list[CompletionPath] = []
    max_path_tokens = max(1, int(max_path_tokens))
    for candidate in sorted(candidates - specials):
        if candidate == int(tokenizer.eos_id):
            paths.append(CompletionPath((candidate,), "eos"))
            continue
        piece = _token_piece(tokenizer, candidate)
        branch = OpenUIIncrementalEngine(engine.grammar_path)
        if not branch.set_prefix(prefix_text):
            continue
        admitted = branch.probe_chunk(piece)
        if admitted is None:
            admitted = branch.set_prefix(prefix_text + piece)
        elif admitted:
            admitted = branch.advance(piece)
        # InteractiveParser accepted the edge and exposes at least one follow
        # terminal, which is the exact CFG reachability guarantee we need.
        if not admitted or not branch.next_terminals():
            continue
        drafted = [int(candidate)]
        branch_text = prefix_text + piece
        while len(drafted) < max_path_tokens:
            forced = force_next_token_id(branch, tokenizer, branch_text)
            if forced is None or forced in specials:
                break
            drafted.append(int(forced))
            branch_text += _token_piece(tokenizer, forced)
        paths.append(
            CompletionPath(tuple(drafted), _semantic_kind(tokenizer, candidate))
        )

    if not paths:
        coverage: Coverage = "partial" if terminals else "none"
    elif inventory_complete and _known_terminal_coverage(tokenizer, terminals):
        coverage = "complete"
    else:
        coverage = "partial"
    return CompletionForest(
        tuple(paths), coverage, tuple(sorted(str(term) for term in terminals))
    )


__all__ = [
    "CompletionForest",
    "CompletionPath",
    "Coverage",
    "build_completion_forest",
]
