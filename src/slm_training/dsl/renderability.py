"""Standalone-root renderability rules for the pinned OpenUI library."""

from __future__ import annotations

from typing import Any


# These nodes declare data or items consumed by another component.  They are
# valid OpenUI expressions but render no visible standalone root.
STRUCTURAL_ROOT_CONTAINERS: dict[str, str] = {
    "Col": "Table",
    "Series": "a chart",
    "Slice": "PieChart",
    "ScatterSeries": "ScatterChart",
    "Point": "ScatterSeries",
    "SelectItem": "Select",
    "CheckBoxItem": "CheckBoxGroup",
    "RadioItem": "RadioGroup",
    "SwitchItem": "SwitchGroup",
    "TabItem": "Tabs",
    "AccordionItem": "Accordion",
    "StepsItem": "Steps",
}


def root_type(program: Any) -> str:
    """Return the component name of a validated document root, if present."""
    root = getattr(program, "root", None)
    return str(root.get("typeName") or "") if isinstance(root, dict) else ""


def structural_root_container(name: str) -> str | None:
    """Return the visible parent required by a structural-only root."""
    return STRUCTURAL_ROOT_CONTAINERS.get(name)


__all__ = ["STRUCTURAL_ROOT_CONTAINERS", "root_type", "structural_root_container"]
