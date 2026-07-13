"""Slot-contract template fill helpers for OpenUI decode (E20).

Given a placeholder inventory, build a certified-minimal OpenUI skeleton so
decode can bind inventory slots first, then optionally diffuse structure.
"""

from __future__ import annotations

import re

_BINDER_RE = re.compile(r"[^A-Za-z0-9_]+")


def _binder_name(placeholder: str, index: int) -> str:
    body = placeholder[1:] if placeholder.startswith(":") else placeholder
    parts = [p for p in body.split(".") if p]
    if not parts:
        return f"item_{index}"
    raw = "_".join(parts[-2:] if len(parts) > 1 else parts)
    cleaned = _BINDER_RE.sub("_", raw).strip("_").lower() or f"item_{index}"
    if cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return cleaned[:48]


def normalize_placeholders(placeholders: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in placeholders or []:
        ph = raw if raw.startswith(":") else f":{raw}"
        if ph in seen:
            continue
        seen.add(ph)
        out.append(ph)
    return out


def build_slot_contract_template(placeholders: list[str] | None) -> str:
    """
    Deterministic valid OpenUI program binding every inventory slot.

    Layout is a single column Stack of TextContent nodes — enough for parse +
    fidelity when the model cannot yet compose richer trees.
    """
    slots = normalize_placeholders(placeholders)
    if not slots:
        return 'root = Stack([item], "column")\nitem = TextContent(":item")\n'

    binders: list[str] = []
    used: set[str] = set()
    lines: list[str] = []
    for i, ph in enumerate(slots):
        base = _binder_name(ph, i)
        name = base
        n = 2
        while name in used or name == "root":
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        binders.append(name)
        lines.append(f'{name} = TextContent("{ph}")')

    children = ", ".join(binders)
    header = f'root = Stack([{children}], "column")'
    return header + "\n" + "\n".join(lines) + "\n"


def template_mask_positions(
    token_ids: list[int],
    tokenizer,
    *,
    mask_placeholders: bool = True,
    mask_binders: bool = True,
) -> list[int]:
    """
    Positions to remask on a template canvas so MaskGIT can refine them.

    Keeps structural punctuation / keywords so the program stays near-valid.
    """
    structural = {
        "=",
        "(",
        ")",
        "[",
        "]",
        ",",
        '"',
        ":",
        ".",
        "root",
        "Stack",
        "TextContent",
        "Card",
        "column",
        "row",
    }
    special = {
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.mask_id,
    }
    mask_at: list[int] = []
    for i, tid in enumerate(token_ids):
        if tid in special:
            continue
        tok = tokenizer.id_to_token.get(int(tid), "")
        if tok in structural:
            continue
        if tok.startswith(":"):
            if mask_placeholders:
                mask_at.append(i)
            continue
        # Binder identifiers / residual segments.
        if mask_binders and tok.isidentifier():
            mask_at.append(i)
    return mask_at
