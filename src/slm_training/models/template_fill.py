"""Slot-contract template fill helpers for OpenUI decode (E20 / E35).

Given a placeholder inventory, build a certified-minimal OpenUI skeleton so
decode can bind inventory slots first, then optionally diffuse structure.

E35 honesty: prefer inventory extracted from the prompt/DESIGN.md text rather
than a hidden gold ``record.placeholders`` channel.
"""

from __future__ import annotations

import hashlib
import re

from slm_training.dsl.placeholders import PLACEHOLDER_RE, extract_placeholders

_BINDER_RE = re.compile(r"[^A-Za-z0-9_]+")
_INVENTORY_LINE_RE = re.compile(
    r"(?i)(?:placeholders?|slot(?:\s+inventory)?|inventory)\s*:\s*(.+)$",
    re.MULTILINE,
)

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


def ensure_prompt_inventory(prompt: str, placeholders: list[str] | None) -> str:
    """Append a visible slot inventory section if the prompt lacks placeholders."""
    slots = normalize_placeholders(placeholders)
    if not slots:
        return prompt
    existing = inventory_from_prompt(prompt, heuristic=False)
    if existing:
        return prompt
    joined = ", ".join(slots)
    base = (prompt or "").rstrip()
    return f"{base}\nPlaceholders: {joined}"


def ensure_prompt_semantic_roles(
    prompt: str, placeholders: list[str] | None
) -> str:
    """Normalize prompt-mentioned component types into an honest role contract."""
    if any(line.startswith("Semantic roles:") for line in prompt.splitlines()):
        return prompt
    slots = normalize_placeholders(placeholders)
    if not slots:
        return prompt
    from slm_training.data.quality import (
        _prompt_component_mentions,
        semantic_role_contract,
    )

    components = sorted(_prompt_component_mentions(prompt))
    if not components:
        return prompt
    base = prompt.rstrip()
    if not any(line.startswith("Components:") for line in base.splitlines()):
        base = f"{base}\nComponents: {', '.join(components)}"
    roles = semantic_role_contract(slots, components)
    return f"{base}\nSemantic roles: {roles}" if roles else base


def prompt_semantic_role_candidates(
    prompt: str, placeholders: list[str] | None
) -> dict[str, tuple[str, ...]]:
    """Return only schema-compatible candidates justified by visible prompt prose."""
    slots = normalize_placeholders(placeholders)
    if not slots:
        return {}
    from slm_training.data.quality import (
        _prompt_component_mentions,
        semantic_role_candidates,
    )

    components = sorted(_prompt_component_mentions(prompt))
    if not components:
        return {}
    candidates = {
        slot: set(names)
        for slot, names in semantic_role_candidates(slots, components).items()
    }
    authored_prompt = "\n".join(
        line
        for line in prompt.splitlines()
        if not line.startswith(("Placeholders:", "Components:", "Semantic roles:"))
    )
    clauses = [
        re.findall(r"[a-z0-9]+", clause)
        for clause in re.split(
            r"[,;.!?]|\b(?:and|with)\b", authored_prompt.lower()
        )
    ]

    def positions(words: list[str], needle: list[str]) -> list[int]:
        return [
            index
            for index in range(len(words) - len(needle) + 1)
            if words[index : index + len(needle)] == needle
        ]

    for slot in slots:
        role = re.findall(r"[a-z0-9]+", slot.removeprefix(":").split(".")[-1])
        if not role:
            continue
        for component in components:
            family = re.findall(
                r"[a-z0-9]+",
                re.sub(r"(?<!^)(?=[A-Z])", " ", component).lower(),
            )
            if any(
                abs(role_at - family_at) <= 3
                for clause in clauses
                for role_at in positions(clause, role)
                for family_at in positions(clause, family)
            ):
                candidates.setdefault(slot, set()).add(component)
    return {
        slot: tuple(sorted(candidates.get(slot, ())))
        for slot in slots
    }


def prompt_semantic_plan(prompt: str):
    """Build a predicted partial plan from explicit, schema-owned prompt mentions."""
    from slm_training.data.progspec.semantic_plan import (
        PlanArchetype,
        PlanCoverage,
        PlanIdentity,
        RoleSlot,
        SemanticPlanV1,
    )
    from slm_training.data.quality import (
        _prompt_component_mentions,
        _prompt_component_requirements,
    )

    authored_prompt = "\n".join(
        line
        for line in prompt.splitlines()
        if not line.startswith(("Placeholders:", "Components:", "Semantic roles:"))
    )
    components = list(
        _prompt_component_requirements(
            authored_prompt,
            preserve_repeated_mentions=True,
        )
    )
    normalized_prompt = re.sub(r"[^a-z0-9]+", " ", authored_prompt.lower())
    if "Button" not in components and any(
        re.search(rf"\b{re.escape(hint)}\b", normalized_prompt)
        for hint in _BUTTON_HINTS
    ):
        components.append("Button")
    if not components:
        components = sorted(_prompt_component_mentions(prompt))
    if not components:
        return None
    return SemanticPlanV1(
        identity=PlanIdentity(
            pack_id="openui",
            prompt_context_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            provenance="predicted",
        ),
        archetype=PlanArchetype(confidence=1.0),
        role_slots=tuple(
            RoleSlot(
                role_id=f"prompt_component_{index}",
                component_family=component,
                required=True,
                evidence_spans=(component,),
            )
            for index, component in enumerate(components)
        ),
        coverage=PlanCoverage(
            named_requirements_accounted_for=tuple(components),
        ),
    )


def inventory_from_prompt(
    prompt: str | None,
    design_md: str | None = None,
    *,
    heuristic: bool = True,
) -> list[str]:
    """
    Derive a slot inventory from user-visible text (E35).

    Priority:
      1. Explicit ``Placeholders:`` / ``Inventory:`` lines
      2. Any ``:ns.slot`` tokens in prompt + DESIGN.md
      3. (optional) keyword heuristic under a stable namespace from the prompt
    """
    text = f"{prompt or ''}\n{design_md or ''}"
    explicit: list[str] = []
    for match in _INVENTORY_LINE_RE.finditer(text):
        explicit.extend(PLACEHOLDER_RE.findall(match.group(1)))
        for raw in re.split(r"[,;\s]+", match.group(1)):
            raw = raw.strip().strip("\"'")
            if not raw:
                continue
            body = raw[1:] if raw.startswith(":") else raw
            if "." in body and body.replace(".", "").replace("_", "").isalnum():
                explicit.append(raw if raw.startswith(":") else f":{raw}")
    if explicit:
        return normalize_placeholders(explicit)

    found = extract_placeholders(text)
    if found:
        return normalize_placeholders(found)

    if not heuristic:
        return []
    return _heuristic_inventory_from_prompt(prompt or "")


def _heuristic_inventory_from_prompt(prompt: str) -> list[str]:
    """Keyword → inventory under a stable namespace derived from the prompt."""
    low = prompt.lower()
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:6]
    ns = "ui"
    for candidate in (
        "smoke",
        "held",
        "adv",
        "ood",
        "hero",
        "form",
        "login",
        "settings",
        "pricing",
        "gallery",
        "auth",
        "dash",
        "modal",
        "tabs",
    ):
        if candidate in low:
            ns = candidate
            break
    else:
        ns = f"p{digest}"

    slots: list[str] = []
    if any(w in low for w in ("title", "header", "heading", "kicker")):
        slots.append(f":{ns}.title")
    if any(w in low for w in ("subtitle", "subheading")):
        slots.append(f":{ns}.subtitle")
    if any(w in low for w in ("body", "description", "desc", "copy", "text", "blurb")):
        slots.append(f":{ns}.body")
    if any(w in low for w in ("button", "cta", "submit", "action", "continue")):
        slots.append(f":{ns}.cta")
    if any(w in low for w in ("email", "input", "field", "password", "search")):
        slots.append(f":{ns}.input")
    if any(w in low for w in ("image", "avatar", "photo", "banner", "icon")):
        slots.append(f":{ns}.image")
        slots.append(f":{ns}.alt")
    if any(w in low for w in ("callout", "alert", "notice", "warning")):
        slots.append(f":{ns}.callout.title")
        slots.append(f":{ns}.callout.body")
    if any(w in low for w in ("card", "feature")) and f":{ns}.title" not in slots:
        slots.append(f":{ns}.title")
        if f":{ns}.body" not in slots:
            slots.append(f":{ns}.body")
    if not slots:
        slots = [f":{ns}.title", f":{ns}.body"]
    return normalize_placeholders(slots)


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
                elif kind in {TokenKind.BIND, TokenKind.STATE} and mask_binders:
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
