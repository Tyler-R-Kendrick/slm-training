"""Map grammar terminals / forced strings onto tokenizer ids.

Supports both the legacy compositional ``OpenUITokenizer`` (heuristic vocab
scan) and the V5 lexer-native ``DSLNativeTokenizer`` (exact kind metadata).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from slm_training.models.tokenizer import OpenUITokenizer

_DSL_ALLOWED_CACHE: dict[tuple[int, int, frozenset[str]], frozenset[int]] = {}
_DSL_DIRECT_FACTS_CACHE: dict[
    tuple[str, tuple[tuple[str, str, str], ...]], "DSLDirectTokenFacts"
] = {}
_DSL_DIRECT_FACTS_CACHE_CAP = 16
_NUMBER_COMPLETE = re.compile(
    r"^-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$"
)
_NUMBER_PREFIX = re.compile(
    r"^(?:-?|"
    r"-?\d+(?:\.\d*)?|"
    r"-?\.\d*|"
    r"-?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d*)$"
)


@dataclass(frozen=True, slots=True)
class DSLDirectTokenFacts:
    """Immutable verified token facts for one frozen tokenizer layout."""

    layout_key: str
    punct: Mapping[int, str]
    bool: Mapping[int, str]
    null: Mapping[int, str]
    kind_terminals: Mapping[Any, str]
    str_lit_ids: Mapping[int, str]
    lit_str: int | None
    lit_num: int
    lit_end: int
    skip_ids: frozenset[int]

    def __getitem__(self, name: str) -> Any:
        """Compatibility for the existing direct-feed call sites."""
        return getattr(self, name)


def _is_dsl_native(tokenizer: Any) -> bool:
    try:
        from slm_training.models.dsl_tokenizer import is_dsl_native_tokenizer

        return is_dsl_native_tokenizer(tokenizer)
    except Exception:  # noqa: BLE001
        return False


def string_to_token_ids(tokenizer: OpenUITokenizer, text: str) -> list[int]:
    """Encode a forced lexeme; prefer exact vocab hit then tokenizer.encode."""
    if text in tokenizer.token_to_id:
        return [tokenizer.token_to_id[text]]
    # Newline maps to NL for lexer-native tokenizers.
    if text in {"\n", "\r\n"} and "NL" in tokenizer.token_to_id:
        return [tokenizer.token_to_id["NL"]]
    # Multi-char forced strings (rare) — encode without BOS/EOS if possible.
    ids = tokenizer.encode(text, add_special=False)
    return ids


def decode_prefix(tokenizer: OpenUITokenizer, token_ids: list[int]) -> str:
    """Decode an incremental grammar prefix without dropping terminal newlines."""
    if _is_dsl_native(tokenizer):
        return tokenizer.decode(token_ids, preserve_trailing_newline=True)
    return tokenizer.decode(token_ids)


def token_surface_piece(tokenizer: OpenUITokenizer, token_id: int) -> str:
    """Decode one token into the source fragment consumed by grammar state."""
    tid = int(token_id)
    if tid in {
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.mask_id,
    }:
        return ""
    raw = str(tokenizer.id_to_token.get(tid, ""))
    if raw == "NL":
        return "\n"
    if raw == "LIT_STR":
        return '"'
    if raw in {"LIT_NUM", "LIT_END"}:
        return ""
    if raw.startswith("B:"):
        try:
            return chr(int(raw[2:], 16))
        except ValueError:
            return raw
    kind_of = getattr(tokenizer, "kind_of", None)
    kind = getattr(kind_of(tid), "value", "") if callable(kind_of) else ""
    if kind in {"sym", "bind", "state", "lit", "macro"} or not raw:
        decoded = tokenizer.decode([tid])
        if decoded:
            return decoded
    return raw


def allowed_id_set(
    tokenizer: OpenUITokenizer,
    terminals: frozenset[str],
    *,
    active_dynamic_ids: set[int] | None = None,
    use_cache: bool = False,
) -> set[int] | None:
    """
    Expand accepts() terminal names to tokenizer ids.

    Returns None when the accept set is empty/unknown (caller should fall back).
    """
    if not terminals:
        return None
    if _is_dsl_native(tokenizer):
        fingerprint = hash(tuple(sorted(tokenizer.token_to_id.items())))
        key = (fingerprint, int(getattr(tokenizer, "version", 0)), terminals)
        cached = _DSL_ALLOWED_CACHE.get(key) if use_cache else None
        if cached is None:
            result = _allowed_id_set_dsl(tokenizer, terminals)
            if result is not None and use_cache:
                _DSL_ALLOWED_CACHE[key] = frozenset(result)
        else:
            result = set(cached)
        if result is not None and active_dynamic_ids is not None:
            from slm_training.models.dsl_tokenizer import TokenKind

            dynamic = tokenizer.kind_ids(TokenKind.SYM) | tokenizer.kind_ids(
                TokenKind.STATE
            )
            result = (result - dynamic) | (result & active_dynamic_ids)
        return result
    return _allowed_id_set_compositional(tokenizer, terminals)


def apply_literal_frame(
    tokenizer: OpenUITokenizer,
    prefix_ids: list[int],
    candidates: set[int] | None,
) -> set[int] | None:
    """Restrict lexer-native framed literals to bytes until ``LIT_END``."""
    if not _is_dsl_native(tokenizer):
        return candidates

    from slm_training.models.dsl_tokenizer import TokenKind

    openers = {
        int(token_id): name
        for name in ("LIT_STR", "LIT_NUM")
        if (token_id := tokenizer.token_to_id.get(name)) is not None
    }
    closer = tokenizer.token_to_id.get("LIT_END")
    if not openers or closer is None:
        return candidates

    byte_ids = set(tokenizer.kind_ids(TokenKind.BYTE))
    frame: str | None = None
    body = ""
    for token_id in prefix_ids:
        current = int(token_id)
        if current in openers and frame is None:
            frame = openers[current]
            body = ""
        elif current == int(closer) and frame is not None:
            frame = None
            body = ""
        elif frame is not None and current in byte_ids:
            raw = str(tokenizer.id_to_token.get(current, ""))
            if raw.startswith("B:"):
                body += chr(int(raw[2:], 16))

    if frame == "LIT_NUM":
        numeric_bytes = {
            token_id
            for token_id in byte_ids
            if (raw := str(tokenizer.id_to_token.get(token_id, ""))).startswith("B:")
            and _NUMBER_PREFIX.fullmatch(body + chr(int(raw[2:], 16)))
        }
        return numeric_bytes | (
            {int(closer)} if _NUMBER_COMPLETE.fullmatch(body) else set()
        )
    if frame is not None:
        return byte_ids | ({int(closer)} if body else set())
    if candidates is None:
        return None
    return set(candidates) - byte_ids - {int(closer)}


def _dsl_layout_key(tokenizer: Any) -> str:
    """Stable identity for the checkpoint-bound token-id layout."""
    payload = {
        "version": int(getattr(tokenizer, "version", 0)),
        "token_to_id": sorted(
            (str(token), int(token_id))
            for token, token_id in tokenizer.token_to_id.items()
        ),
        "id_to_kind": sorted(
            (int(token_id), str(kind))
            for token_id, kind in tokenizer.id_to_kind.items()
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dsl_direct_terminal_map(
    tokenizer: Any,
    grammar_terminals: Any,
) -> DSLDirectTokenFacts | None:
    """Verified DSL-native token-id → Lark-terminal mapping for direct feeds.

    Facts are cached process-wide by stable tokenizer-layout digest and live
    grammar-terminal signature. Request-local symbol values and macro
    expansions are deliberately absent. Anything unverifiable returns
    ``None`` so callers stay on the canonical text path.
    """
    if not _is_dsl_native(tokenizer):
        return None

    from slm_training.models.dsl_tokenizer import TokenKind

    layout_key = _dsl_layout_key(tokenizer)
    terminal_signature = tuple(
        sorted(
            (
                str(t.name),
                str(t.pattern.type),
                str(t.pattern.value),
            )
            for t in grammar_terminals
        )
    )
    cache_key = (layout_key, terminal_signature)
    cached = _DSL_DIRECT_FACTS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    by_name = {t.name: t for t in grammar_terminals}
    required = {
        "_NL",
        "NAME",
        "COMPONENT",
        "BUILTIN",
        "STATE_NAME",
        "STRING",
        "NUMBER",
        "BOOL",
        "NULL",
    }
    if not required <= set(by_name):
        return None

    # Literal (string-pattern) terminals: map the exact literal text to its
    # verified terminal name (covers EQUAL…RBRACE plus anonymous __ANON_*).
    literal_terminals = {
        t.pattern.value: t.name
        for t in grammar_terminals
        if t.pattern.type == "str" and t.pattern.value
    }
    punct: dict[int, str] = {}
    for token_id in tokenizer.kind_ids(TokenKind.STRUCT):
        text = str(tokenizer.id_to_token.get(int(token_id), ""))
        if text == "NL":
            punct[int(token_id)] = "_NL"
            continue
        name = literal_terminals.get(text)
        if name is None:
            # Unverifiable punctuation row — fail closed.
            return None
        punct[int(token_id)] = name

    t2id = tokenizer.token_to_id
    bool_ids = {
        int(t2id[b]): "BOOL" for b in ("true", "false") if b in t2id
    }
    if len(bool_ids) != 2 or "null" not in t2id:
        return None
    lit_num = t2id.get("LIT_NUM")
    lit_end = t2id.get("LIT_END")
    if lit_num is None or lit_end is None:
        return None

    kind_terminals = {
        TokenKind.COMPONENT: "COMPONENT",
        TokenKind.BUILTIN: "BUILTIN",
        TokenKind.BIND: "NAME",
        TokenKind.STATE: "STATE_NAME",
        TokenKind.SYM: "STRING",
    }
    str_lit_ids = {
        int(token_id): "STRING"
        for token_id in tokenizer.kind_ids(TokenKind.LIT)
        if str(tokenizer.id_to_token.get(int(token_id), "")).startswith("STR:")
    }
    skip_ids = {
        int(token_id)
        for token_id in (
            tokenizer.bos_id,
            tokenizer.eos_id,
            tokenizer.pad_id,
            tokenizer.mask_id,
        )
        if token_id is not None
    }
    facts = DSLDirectTokenFacts(
        layout_key=layout_key,
        punct=MappingProxyType(punct),
        bool=MappingProxyType(bool_ids),
        null=MappingProxyType({int(t2id["null"]): "NULL"}),
        kind_terminals=MappingProxyType(kind_terminals),
        str_lit_ids=MappingProxyType(str_lit_ids),
        lit_str=(
            int(value) if (value := t2id.get("LIT_STR")) is not None else None
        ),
        lit_num=int(lit_num),
        lit_end=int(lit_end),
        skip_ids=frozenset(skip_ids),
    )
    if len(_DSL_DIRECT_FACTS_CACHE) >= _DSL_DIRECT_FACTS_CACHE_CAP:
        _DSL_DIRECT_FACTS_CACHE.clear()
    _DSL_DIRECT_FACTS_CACHE[cache_key] = facts
    return facts


def terminal_equivalence_classes(
    tokenizer: OpenUITokenizer,
    terminal_sets: list[frozenset[str]],
) -> dict[frozenset[int], list[frozenset[str]]]:
    """Group parser states that induce the same token candidate set."""
    groups: dict[frozenset[int], list[frozenset[str]]] = {}
    for terminals in terminal_sets:
        ids = allowed_id_set(tokenizer, terminals, use_cache=True)
        if ids is not None:
            groups.setdefault(frozenset(ids), []).append(terminals)
    return groups


def _allowed_id_set_dsl(tokenizer: Any, terminals: frozenset[str]) -> set[int] | None:
    from slm_training.models.dsl_tokenizer import TokenKind

    ignore = {"$END", "COMMENT"}
    punctuation = {
        "EQUAL": "=",
        "LPAR": "(",
        "RPAR": ")",
        "LSQB": "[",
        "RSQB": "]",
        "LBRACE": "{",
        "RBRACE": "}",
        "COMMA": ",",
        "DOT": ".",
        "COLON": ":",
        "QMARK": "?",
        "PLUS": "+",
        "MINUS": "-",
        "STAR": "*",
        "SLASH": "/",
        "PERCENT": "%",
        "BANG": "!",
        "MORETHAN": ">",
        "LESSTHAN": "<",
        "__ANON_0": "||",
        "__ANON_1": "&&",
        "__ANON_2": "==",
        "__ANON_3": "!=",
        "__ANON_4": ">=",
        "__ANON_5": "<=",
    }
    ids: set[int] = set()
    broad = False
    for term in terminals:
        if term in ignore:
            continue
        mapped = punctuation.get(term, term)
        if mapped in tokenizer.token_to_id and mapped in punctuation.values():
            ids.add(tokenizer.token_to_id[mapped])
        elif term in {"_NL", "NL"}:
            ids.add(tokenizer.token_to_id["NL"])
        elif term == "WS_INLINE":
            # Whitespace is not modeled in lexer-native output.
            continue
        elif term == "COMPONENT":
            broad = True
            ids |= tokenizer.kind_ids(TokenKind.COMPONENT)
        elif term == "NAME":
            broad = True
            ids |= tokenizer.kind_ids(TokenKind.BIND)
        elif term == "STATE_NAME":
            broad = True
            ids |= tokenizer.kind_ids(TokenKind.STATE)
        elif term == "BUILTIN":
            broad = True
            ids |= tokenizer.kind_ids(TokenKind.BUILTIN)
        elif term == "STRING":
            broad = True
            ids |= tokenizer.kind_ids(TokenKind.SYM)
            ids |= {
                token_id
                for token_id in tokenizer.kind_ids(TokenKind.LIT)
                if str(tokenizer.id_to_token.get(token_id, "")).startswith("STR:")
            }
            # Fixed string symbols and placeholders are valid STRING starts.
            # Booleans, null, LIT_NUM, and LIT_END share the broad LIT kind but
            # belong to other grammar terminals and must not leak into STRING.
            lit_str = tokenizer.token_to_id.get("LIT_STR")
            if lit_str is not None:
                ids.add(lit_str)
        elif term == "NUMBER":
            broad = True
            lit_num = tokenizer.token_to_id.get("LIT_NUM")
            if lit_num is not None:
                ids.add(lit_num)
        elif term == "BOOL":
            for b in ("true", "false"):
                if b in tokenizer.token_to_id:
                    ids.add(tokenizer.token_to_id[b])
        elif term == "NULL":
            if "null" in tokenizer.token_to_id:
                ids.add(tokenizer.token_to_id["null"])
        else:
            # Literal terminal name may already be in vocab.
            if term in tokenizer.token_to_id:
                ids.add(tokenizer.token_to_id[term])
    if not ids and broad:
        return None
    return ids or None


def _allowed_id_set_compositional(
    tokenizer: OpenUITokenizer,
    terminals: frozenset[str],
) -> set[int] | None:
    ignore = {"$END", "COMMENT"}
    forced_map = {
        "EQUAL": "=",
        "LPAR": "(",
        "RPAR": ")",
        "LSQB": "[",
        "RSQB": "]",
        "COMMA": ",",
        "NAME": None,  # any name — expand to lowercase identifiers in vocab
        "COMPONENT": None,
        "STRING": None,
        "NUMBER": None,
        "BOOL": None,
        "_NL": "\n",
        "WS_INLINE": " ",
    }
    ids: set[int] = set()
    broad = False
    for term in terminals:
        if term in ignore:
            continue
        mapped = forced_map.get(term, term)
        if mapped is None:
            # Broad content terminals: expand to matching vocab ids when possible.
            broad = True
            if term == "COMPONENT":
                for tok, tid in tokenizer.token_to_id.items():
                    if tok[:1].isupper() and tok.isidentifier():
                        ids.add(tid)
            elif term == "NAME":
                for tok, tid in tokenizer.token_to_id.items():
                    if tok[:1].islower() and tok.isidentifier():
                        ids.add(tid)
            elif term == "STRING":
                for tok, tid in tokenizer.token_to_id.items():
                    if tok.startswith('"') or tok.startswith(":"):
                        ids.add(tid)
            continue
        for tid in string_to_token_ids(tokenizer, mapped):
            ids.add(tid)
    if not ids and broad:
        return None
    return ids or None
