"""Map grammar terminals / forced strings onto tokenizer ids.

Supports both the legacy compositional ``OpenUITokenizer`` (heuristic vocab
scan) and the V5 lexer-native ``DSLNativeTokenizer`` (exact kind metadata).
"""

from __future__ import annotations

from typing import Any

from slm_training.models.tokenizer import OpenUITokenizer


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


def allowed_id_set(
    tokenizer: OpenUITokenizer,
    terminals: frozenset[str],
) -> set[int] | None:
    """
    Expand accepts() terminal names to tokenizer ids.

    Returns None when the accept set is empty/unknown (caller should fall back).
    """
    if not terminals:
        return None
    if _is_dsl_native(tokenizer):
        return _allowed_id_set_dsl(tokenizer, terminals)
    return _allowed_id_set_compositional(tokenizer, terminals)


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
            ids |= tokenizer.kind_ids(TokenKind.LIT)
            # Fixed strings + LIT_STR opener + bool/null are LIT; also allow
            # byte channel only after LIT_STR (caller handles framing).
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
