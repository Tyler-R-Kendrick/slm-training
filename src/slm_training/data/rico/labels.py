"""RICO semantic component labels (Liu et al. / HF shunk031/Rico)."""

from __future__ import annotations

# Official semantic annotation label ids from the Rico semantic annotations release.
COMPONENT_LABELS: dict[int, str] = {
    0: "Text",
    1: "Image",
    2: "Icon",
    3: "Text Button",
    4: "List Item",
    5: "Input",
    6: "Background Image",
    7: "Card",
    8: "Web View",
    9: "Radio Button",
    10: "Drawer",
    11: "Checkbox",
    12: "Advertisement",
    13: "Modal",
    14: "Pager Indicator",
    15: "Slider",
    16: "On/Off Switch",
    17: "Button Bar",
    18: "Toolbar",
    19: "Number Stepper",
    20: "Multi-Tab",
    21: "Date Picker",
    22: "Map View",
    23: "Video",
    24: "Bottom Navigation",
}

# Labels we can express with the OpenUI training subset (Stack/Card/Text/Button).
MAPPABLE_LABELS: frozenset[str] = frozenset(
    {
        "Text",
        "Text Button",
        "List Item",
        "Input",
        "Card",
        "Radio Button",
        "Checkbox",
        "Modal",
        "On/Off Switch",
        "Button Bar",
        "Toolbar",
        "Multi-Tab",
        "Bottom Navigation",
    }
)
