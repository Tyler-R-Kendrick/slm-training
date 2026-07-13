"""Slot-contract template fill helpers for OpenUI decode (E20).

Given a placeholder inventory, build a certified-minimal OpenUI skeleton so
decode can bind inventory slots first, then optionally diffuse structure.
"""

from __future__ import annotations

import re

_BINDER_RE = re.compile(r"[^A-Za-z0-9_]+")

_BUTTON_HINTS = (
    "button",
    "btn",
    "cta",
    "action",
    "submit",
    "primary",
    "label",
    "confirm",
    "continue",
    "create",
    "save",
    "next",
)
_INPUT_HINTS = ("input", "email", "search", "field", "name", "password")
_IMAGE_HINTS = ("image", "avatar", "photo", "thumb", "icon", "banner")
_CALLOUT_HINTS = ("callout", "alert", "warning", "info", "notice")


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


def _slot_kind(placeholder: str) -> str:
    body = placeholder.lower()
    if any(h in body for h in _BUTTON_HINTS):
        return "button"
    if any(h in body for h in _INPUT_HINTS):
        return "input"
    if any(h in body for h in _IMAGE_HINTS):
        return "image"
    if any(h in body for h in _CALLOUT_HINTS):
        return "callout"
    return "text"


def build_slot_contract_template(placeholders: list[str] | None) -> str:
    """
    Deterministic valid OpenUI program binding every inventory slot.

    Uses placeholder-name heuristics for component types and wraps text-heavy
    groups in a Card so meaningful-parse component recall stays above floor.
    """
    slots = normalize_placeholders(placeholders)
    if not slots:
        return (
            'root = Stack([card], "column")\n'
            'title = TextContent(":item")\n'
            "card = Card([title])\n"
        )

    binders: list[str] = []
    kinds: list[str] = []
    used: set[str] = {"root", "card", "hero", "note"}
    leaf_lines: list[str] = []
    consumed: set[int] = set()

    def _fresh(base: str) -> str:
        name = base
        n = 2
        while name in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        return name

    # Pair callout title/body into a single Callout("info", title, body).
    for i, ph in enumerate(slots):
        if i in consumed:
            continue
        kind = _slot_kind(ph)
        if kind == "callout" and "title" in ph.lower():
            body_idx = None
            for j in range(i + 1, len(slots)):
                if j in consumed:
                    continue
                other = slots[j]
                if _slot_kind(other) == "callout" and "body" in other.lower():
                    body_idx = j
                    break
                if _slot_kind(other) == "text" and any(
                    x in other.lower() for x in ("body", "description", "desc")
                ):
                    body_idx = j
                    break
            if body_idx is not None:
                name = _fresh(_binder_name(ph, i))
                leaf_lines.append(
                    f'{name} = Callout("info", "{ph}", "{slots[body_idx]}")'
                )
                binders.append(name)
                kinds.append("callout")
                consumed.add(i)
                consumed.add(body_idx)
                continue

        name = _fresh(_binder_name(ph, i))
        binders.append(name)
        kinds.append(kind)
        leaf_lines.append(_emit_leaf(name, ph, kind))
        consumed.add(i)

    textish = [b for b, k in zip(binders, kinds) if k == "text"]
    other = [b for b, k in zip(binders, kinds) if k != "text"]

    lines: list[str] = []
    root_children: list[str] = []
    if len(textish) >= 2:
        # Card wrapper improves Card+TextContent recall vs flat TextContent stack.
        card_kids = ", ".join(textish)
        lines.append(f"card = Card([{card_kids}])")
        root_children.append("card")
        root_children.extend(other)
    else:
        root_children = list(binders)

    children = ", ".join(root_children)
    lines.insert(0, f'root = Stack([{children}], "column")')
    lines.extend(leaf_lines)
    return "\n".join(lines) + "\n"


def _emit_leaf(binder: str, ph: str, kind: str) -> str:
    if kind == "button":
        return f'{binder} = Button("{ph}")'
    if kind == "input":
        # name + placeholder props — keep name as a stable literal.
        return f'{binder} = Input("text", "{ph}")'
    if kind == "image":
        return f'{binder} = ImageBlock("{ph}", "{ph}")'
    if kind == "callout":
        # Single-slot callout: reuse placeholder for title + description.
        return f'{binder} = Callout("info", "{ph}", "{ph}")'
    return f'{binder} = TextContent("{ph}")'


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
    try:
        from slm_training.models.dsl_tokenizer import TokenKind, is_dsl_native_tokenizer

        if is_dsl_native_tokenizer(tokenizer):
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
                kind = tokenizer.kind_of(int(tid))
                if kind == TokenKind.SYM and mask_placeholders:
                    mask_at.append(i)
                elif kind == TokenKind.BIND and mask_binders:
                    mask_at.append(i)
            return mask_at
    except Exception:  # noqa: BLE001
        pass

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
        "Button",
        "Input",
        "ImageBlock",
        "Callout",
        "column",
        "row",
    }
    special = {
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.mask_id,
    }
    mask_at = []
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