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


def ensure_prompt_semantic_roles(prompt: str, placeholders: list[str] | None) -> str:
    """Reject the removed marker-semantic prompt channel."""
    raise ValueError("template markers are opaque; semantic role contracts are prohibited")


def prompt_semantic_role_candidates(
    prompt: str,
    placeholders: list[str] | None,
    *,
    include_schema_candidates: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Reject the removed marker-name candidate channel."""
    raise ValueError("template markers are opaque; semantic role candidates are prohibited")


def typed_semantic_role_candidates(
    prompt: str,
    placeholders: list[str] | None,
    runtime_symbols: list[object] | None,
    *,
    include_schema_candidates: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Reject the removed typed marker-semantic channel."""
    raise ValueError("template markers are opaque; typed semantic roles are prohibited")


def typed_semantic_role_properties(
    placeholders: list[str] | None,
    runtime_symbols: list[object] | None,
) -> dict[str, tuple[str, ...]]:
    """Reject the removed typed marker-property channel."""
    raise ValueError("template markers are opaque; typed semantic roles are prohibited")


def prompt_semantic_plan(prompt: str):
    """Build a predicted partial plan from explicit, schema-owned prompt mentions."""
    from slm_training.data.progspec.semantic_plan import (
        PlanArchetype,
        PlanCoverage,
        PlanIdentity,
        PlanTopology,
        RoleSlot,
        SemanticPlanV1,
    )
    from slm_training.data.quality import (
        _official_component_names,
        _prompt_component_mentions,
        _prompt_component_requirements,
    )

    authored_prompt = "\n".join(
        line
        for line in prompt.splitlines()
        if not line.startswith(
            ("Placeholders:", "Components:", "Semantic roles:", "Roles:")
        )
    )
    authored_prompt = re.sub(
        r"\(\s*(?:semantic\s+)?roles?\s*:[^)]*\)",
        "",
        authored_prompt,
        flags=re.IGNORECASE,
    )
    components = list(
        _prompt_component_requirements(
            authored_prompt,
            preserve_repeated_mentions=True,
        )
    )
    normalized_prompt = re.sub(r"[^a-z0-9]+", " ", authored_prompt.lower())
    official_components = _official_component_names()
    for component in sorted(official_components):
        if not component.endswith("Group"):
            continue
        base = component.removesuffix("Group")
        phrase = re.sub(r"(?<!^)(?=[A-Z])", " ", base).lower()
        if (
            base not in official_components
            and component not in components
            and re.search(rf"\b{re.escape(phrase)}s?\b", normalized_prompt)
        ):
            components.append(component)
    if "Button" not in components and any(
        re.search(rf"\b{re.escape(hint)}\b", normalized_prompt)
        for hint in _BUTTON_HINTS
    ):
        components.append("Button")
    if not components:
        components = sorted(_prompt_component_mentions(prompt))
    if not components:
        return None
    role_slots = tuple(
        RoleSlot(
            role_id=f"prompt_component_{index}",
            component_family=component,
            required=True,
            evidence_spans=(component,),
        )
        for index, component in enumerate(components)
    )
    topology = _prompt_outer_group_topology(authored_prompt, role_slots)
    return SemanticPlanV1(
        identity=PlanIdentity(
            pack_id="openui",
            prompt_context_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            provenance="predicted",
        ),
        archetype=PlanArchetype(confidence=1.0),
        role_slots=role_slots,
        topology=PlanTopology(
            parent_relation_candidates=(topology,) if topology is not None else None,
            sibling_order_groups=(
                tuple(topology["sibling_role_ids"]),
            )
            if topology is not None
            else None,
        ),
        coverage=PlanCoverage(
            named_requirements_accounted_for=tuple(components),
        ),
    )


def _prompt_outer_group_topology(
    prompt: str,
    role_slots: tuple[object, ...],
) -> dict[str, object] | None:
    """Extract an explicit group-inside-outer-component relation.

    This deliberately abstains unless the prompt names both a multi-component
    group and an outer component. It never infers topology from component counts
    alone.
    """
    from slm_training.data.quality import _prompt_component_requirements

    normalized = re.sub(r"[^a-z0-9]+", " ", prompt.lower()).strip()
    relation = re.search(
        r"\b(?:group|them|these)\s+(?:inside|within)\s+"
        r"(?:an?\s+|the\s+)?outer\s+(.+)$",
        normalized,
    )
    if relation is None:
        return None
    outer = _prompt_component_requirements(
        relation.group(1), preserve_repeated_mentions=True
    )
    inner_prompt = normalized[: relation.start()]
    inner = list(
        _prompt_component_requirements(
            inner_prompt,
            preserve_repeated_mentions=True,
        )
    )
    if len(outer) != 1 or len(inner) < 2:
        return None
    outer_family = outer[0]
    matching_outer_slots = [
        str(getattr(slot, "role_id"))
        for slot in role_slots
        if getattr(slot, "component_family", None) == outer_family
    ]
    if not matching_outer_slots:
        return None

    sibling_families = inner
    if " around " in inner_prompt:
        left_prompt, right_prompt = inner_prompt.split(" around ", 1)
        right_prompt = right_prompt.split(" then ", 1)[0]
        left = list(
            _prompt_component_requirements(
                left_prompt,
                preserve_repeated_mentions=True,
            )
        )
        right = list(
            _prompt_component_requirements(
                right_prompt,
                preserve_repeated_mentions=True,
            )
        )
        if len(left) >= 2 and right:
            sibling_families = [left[0], *right, *left[1:]]

    remaining_role_ids: dict[str, list[str]] = {}
    for slot in role_slots:
        family = str(getattr(slot, "component_family", "") or "")
        remaining_role_ids.setdefault(family, []).append(str(getattr(slot, "role_id")))
    remaining_role_ids[outer_family].remove(matching_outer_slots[-1])
    sibling_role_ids: list[str] = []
    for family in sibling_families:
        candidates = remaining_role_ids.get(family, [])
        if not candidates:
            return None
        sibling_role_ids.append(candidates.pop(0))
    return {
        "relation": "outer_group",
        "parent_role_id": matching_outer_slots[-1],
        "parent_family": outer_family,
        "group_family": "Stack",
        "sibling_role_ids": sibling_role_ids,
        "sibling_families": sibling_families,
        "direction": "column",
        "evidence": relation.group(0),
    }


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


def build_slot_contract_template(placeholders: list[str] | None) -> str:
    """
    Deterministic valid OpenUI program binding every inventory slot.

    Marker spelling is opaque: component choice and binder identity depend only
    on grammar and ordinal position, never user-defined names.
    """
    slots = normalize_placeholders(placeholders)
    if not slots:
        return "root = Separator()\n"

    binders = [f"slot_{index}" for index in range(len(slots))]
    children = ", ".join(binders)
    lines = [f'root = Stack([{children}], "column")']
    lines.extend(
        f'{binder} = TextContent("{slot}")'
        for binder, slot in zip(binders, slots, strict=True)
    )
    return "\n".join(lines) + "\n"


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
