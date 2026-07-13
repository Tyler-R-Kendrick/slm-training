"""Map grammar terminals / forced strings onto OpenUITokenizer ids."""

from __future__ import annotations

from slm_training.models.tokenizer import OpenUITokenizer


def string_to_token_ids(tokenizer: OpenUITokenizer, text: str) -> list[int]:
    """Encode a forced lexeme; prefer exact vocab hit then tokenizer.encode."""
    if text in tokenizer.token_to_id:
        return [tokenizer.token_to_id[text]]
    # Multi-char forced strings (rare) — encode without BOS/EOS if possible.
    ids = tokenizer.encode(text)
    # Drop bos/eos wrappers when present as sole wrappers.
    if ids and ids[0] == tokenizer.bos_id:
        ids = ids[1:]
    if ids and ids[-1] == tokenizer.eos_id:
        ids = ids[:-1]
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
    forced_map = {
        "EQUAL": "=",
        "LPAR": "(",
        "RPAR": ")",
        "LSQB": "[",
        "RSQB": "]",
        "COMMA": ",",
        "NAME": None,  # any name — too broad; leave unconstrained
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
        mapped = forced_map.get(term, term)
        if mapped is None:
            broad = True
            continue
        for tid in string_to_token_ids(tokenizer, mapped):
            ids.add(tid)
        # Also allow PascalCase components already in vocab when COMPONENT accepted.
        if term == "COMPONENT":
            for tok, tid in tokenizer.token_to_id.items():
                if tok[:1].isupper() and tok.isidentifier():
                    ids.add(tid)
            broad = True
        if term == "NAME":
            for tok, tid in tokenizer.token_to_id.items():
                if tok[:1].islower() and tok.isidentifier():
                    ids.add(tid)
            broad = True
        if term == "STRING":
            for tok, tid in tokenizer.token_to_id.items():
                if tok.startswith('"') or tok.startswith(":"):
                    ids.add(tid)
            broad = True
    if not ids and broad:
        return None
    return ids or None
