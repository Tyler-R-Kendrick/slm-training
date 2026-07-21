"""Map grammar terminals / forced strings onto tokenizer ids.

Supports both the legacy compositional ``OpenUITokenizer`` (heuristic vocab
scan) and the V5 lexer-native ``DSLNativeTokenizer`` (exact kind metadata).
"""

from __future__ import annotations

from typing import Any

from slm_training.models.tokenizer import OpenUITokenizer

_DSL_ALLOWED_CACHE: dict[tuple[int, int, frozenset[str]], frozenset[int]] = {}


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
    """Restrict lexer-native string frames to their legal token channel."""
    if not _is_dsl_native(tokenizer):
        return candidates

    from slm_training.models.dsl_tokenizer import TokenKind

    opener = tokenizer.token_to_id.get("LIT_STR")
    closer = tokenizer.token_to_id.get("LIT_END")
    if opener is None or closer is None:
        return candidates

    inside = False
    for token_id in prefix_ids:
        if int(token_id) == int(opener) and not inside:
            inside = True
        elif int(token_id) == int(closer) and inside:
            inside = False

    byte_ids = set(tokenizer.kind_ids(TokenKind.BYTE))
    if inside:
        return byte_ids | {int(closer)}
    if candidates is None:
        return None
    return set(candidates) - byte_ids - {int(closer)}


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
