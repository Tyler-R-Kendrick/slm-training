"""Convert RICO semantic screens into placeholder OpenUI ExampleRecords."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from slm_training.data.contract import generated_enum_literal, generated_value_literal
from slm_training.data.rico.labels import MAPPABLE_LABELS
from slm_training.dsl.schema import ExampleRecord

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


@dataclass
class RicoElement:
    component_label: str
    klass: str | None = None
    resource_id: str | None = None
    clickable: bool = False
    bounds: list[int] | None = None
    icon_class: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RicoElement:
        bounds = data.get("bounds")
        if bounds is not None and not isinstance(bounds, list):
            bounds = list(bounds)
        return cls(
            component_label=str(data.get("component_label") or ""),
            klass=data.get("klass"),
            resource_id=data.get("resource_id"),
            clickable=bool(data.get("clickable")),
            bounds=bounds,
            icon_class=data.get("icon_class"),
        )


def _slug(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    if ":id/" in value:
        value = value.split(":id/", 1)[1]
    value = value.split(".")[-1]
    slug = _SLUG_RE.sub("_", value).strip("_").lower()
    return slug or fallback


def _sort_key(el: RicoElement) -> tuple[int, int]:
    if not el.bounds or len(el.bounds) < 4:
        return (0, 0)
    x1, y1, x2, y2 = el.bounds[:4]
    return (int(y1), int(x1))


def _infer_direction(elements: list[RicoElement]) -> str:
    """Return official Stack direction: column | row."""
    if len(elements) < 2:
        return "column"
    xs: list[float] = []
    ys: list[float] = []
    for el in elements:
        if el.bounds and len(el.bounds) >= 4:
            x1, y1, x2, y2 = el.bounds[:4]
            xs.append((x1 + x2) / 2)
            ys.append((y1 + y2) / 2)
    if len(xs) < 2:
        return "column"
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    return "row" if dx > dy * 1.2 else "column"


def _role_for_label(label: str) -> str:
    if label in {"Text Button", "Button Bar"}:
        return "button"
    if label in {"Radio Button"}:
        return "radio"
    if label in {"Checkbox"}:
        return "checkbox"
    if label in {"On/Off Switch"}:
        return "switch"
    if label in {"Multi-Tab", "Bottom Navigation"}:
        return "nav"
    if label in {"Card", "List Item", "Modal"}:
        return "card"
    if label in {"Input"}:
        return "input"
    if label in {"Slider"}:
        return "slider"
    if label in {"Date Picker"}:
        return "datepicker"
    if label in {"Image", "Icon", "Background Image"}:
        return "image"
    if label in {"Text", "Toolbar"}:
        return "text"
    return "widget"


def screen_to_openui(
    elements: list[RicoElement],
    *,
    max_children: int = 6,
    namespace: str | None = None,
) -> tuple[str, list[str], dict[str, Any]]:
    """Map semantic elements to a placeholder OpenUI program (openuiLibrary)."""
    usable = [el for el in elements if el.component_label in MAPPABLE_LABELS]
    usable.sort(key=_sort_key)
    seen: set[str] = set()
    filtered: list[RicoElement] = []
    for el in usable:
        key = f"{el.component_label}:{el.resource_id}:{tuple(el.bounds or [])}"
        if key in seen:
            continue
        seen.add(key)
        filtered.append(el)
    filtered = filtered[:max_children]
    if not filtered:
        raise ValueError("no mappable RICO elements")

    direction = _infer_direction(filtered)
    child_ids: list[str] = []
    lines: list[str] = []
    placeholders: list[str] = []
    roles: list[str] = []

    for idx, el in enumerate(filtered):
        role = _role_for_label(el.component_label)
        roles.append(role)
        base = _slug(el.resource_id, f"{role}{idx}")
        if base[0].isdigit():
            base = f"el_{base}"
        name = base
        n = 1
        while name in child_ids:
            name = f"{base}_{n}"
            n += 1
        child_ids.append(name)

        if role in {"button", "nav"}:
            ph = f":{name}.label"
            placeholders.append(ph)
            lines.append(f'{name} = Button("{ph}")')
        elif role == "card":
            title = f":{name}.title"
            body = f":{name}.body"
            title_id = f"{name}_title"
            body_id = f"{name}_body"
            placeholders.extend([title, body])
            lines.append(f'{title_id} = TextContent("{title}")')
            lines.append(f'{body_id} = TextContent("{body}")')
            lines.append(f"{name} = Card([{title_id}, {body_id}])")
        elif role == "input":
            ph = f":{name}.placeholder"
            placeholders.append(ph)
            lines.append(f'{name} = Input("{name}", "{ph}")')
        elif role == "slider":
            ph = f":{name}.label"
            placeholders.append(ph)
            variant = generated_enum_literal("Slider", "variant")
            default_value = generated_value_literal("Slider", "defaultValue", "50")
            lines.append(
                f'{name} = Slider("{name}", {variant}, 0, 100, 1, '
                f'{default_value}, "{ph}")'
            )
        elif role == "datepicker":
            lines.append(f'{name} = DatePicker("{name}")')
        elif role == "checkbox":
            ph = f":{name}.label"
            desc = f":{name}.description"
            placeholders.extend([ph, desc])
            lines.append(f'{name} = CheckBoxItem("{ph}", "{desc}", "{name}")')
        elif role == "radio":
            ph = f":{name}.label"
            desc = f":{name}.description"
            placeholders.extend([ph, desc])
            lines.append(f'{name} = RadioItem("{ph}", "{desc}", "{name}")')
        elif role == "switch":
            ph = f":{name}.label"
            desc = f":{name}.description"
            placeholders.extend([ph, desc])
            lines.append(f'{name} = SwitchItem("{ph}", "{desc}", "{name}")')
        elif role == "image":
            alt = f":{name}.alt"
            # ImageBlock src is an asset ref; keep placeholder-shaped path for policy.
            src = f":assets.{name}"
            placeholders.extend([src, alt])
            lines.append(f'{name} = ImageBlock("{src}", "{alt}")')
        else:
            ph = f":{name}.text"
            placeholders.append(ph)
            lines.append(f'{name} = TextContent("{ph}")')

    # Structure-only scaffold: direction is layout; omit style gap tokens.
    root = f'root = Stack([{", ".join(child_ids)}], "{direction}")'
    openui = "\n".join([root, *lines])
    meta = {
        "direction": direction,
        "roles": roles,
        "component_labels": [el.component_label for el in filtered],
        "n_children": len(child_ids),
        "namespace": namespace,
        "library": "openuiLibrary",
    }
    return openui, placeholders, meta


def _prompt_for_screen(screen: dict[str, Any], meta: dict[str, Any]) -> str:
    labels = meta.get("component_labels") or []
    direction = meta.get("direction") or "column"
    roles = meta.get("roles") or []
    parts: list[str] = []
    n_text = sum(1 for x in labels if x in {"Text", "Toolbar"})
    n_btn = sum(
        1
        for x in labels
        if x
        in {
            "Text Button",
            "Button Bar",
            "Multi-Tab",
            "Bottom Navigation",
        }
    )
    n_card = sum(1 for x in labels if x in {"Card", "List Item", "Modal"})
    n_form = sum(
        1
        for x in labels
        if x in {"Input", "Checkbox", "Radio Button", "On/Off Switch", "Slider", "Date Picker"}
    )
    n_img = sum(1 for x in labels if x in {"Image", "Icon", "Background Image"})
    if n_card:
        parts.append(f"{n_card} card{'s' if n_card != 1 else ''}")
    if n_text:
        parts.append(f"{n_text} text block{'s' if n_text != 1 else ''}")
    if n_btn:
        parts.append(f"{n_btn} button{'s' if n_btn != 1 else ''}")
    if n_form:
        parts.append(f"{n_form} form control{'s' if n_form != 1 else ''}")
    if n_img:
        parts.append(f"{n_img} image{'s' if n_img != 1 else ''}")
    structure = ", ".join(parts) or "UI widgets"
    # Deterministic role hint for the model (sorted unique roles).
    role_hint = ", ".join(sorted(set(roles)))
    idx = screen.get("screen_index")
    split_src = screen.get("split_src") or "train"
    return (
        f"Build a {direction} mobile OpenUI layout with {structure} "
        f"(roles: {role_hint}) using placeholders only "
        f"(RICO {split_src} screen {idx})."
    )


def screen_to_record(
    screen: dict[str, Any],
    *,
    split: str = "train",
    suite: str | None = None,
    max_children: int = 6,
    id_prefix: str = "rico",
) -> ExampleRecord:
    elements = [RicoElement.from_dict(e) for e in (screen.get("elements") or [])]
    screen_index = screen.get("screen_index", 0)
    split_src = screen.get("split_src") or "train"
    namespace = f"{split_src}{screen_index}"
    openui, placeholders, meta = screen_to_openui(
        elements, max_children=max_children, namespace=namespace
    )
    record_id = f"{id_prefix}_{split_src}_{screen_index}"
    prompt = _prompt_for_screen(screen, meta)
    record_meta = {
        **meta,
        "rico_split": split_src,
        "rico_screen_index": screen_index,
        "root_klass": screen.get("root_klass"),
        "source_dataset": "rico",
    }
    if suite:
        record_meta["suite"] = suite
    return ExampleRecord(
        id=record_id,
        prompt=prompt,
        openui=openui,
        placeholders=placeholders,
        split=split,
        source="rico",
        meta=record_meta,
    )
