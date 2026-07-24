"""OpenUI structural / preferred token priors for constrained decode."""

from __future__ import annotations

# Logical token-id namespaces are deliberately separate from codec-local
# embedding rows.  Native, choice, and legacy tokenizers all retain their
# historical 0-based ids for checkpoint compatibility; consumers that combine
# token families must use these disjoint logical ranges instead.
TOKEN_ID_NAMESPACE_REGISTRY_VERSION = 1
TOKEN_ID_NAMESPACE_RANGES = {
    "control": range(0x0000, 0x1000),
    "openui": range(0x1000, 0x4000),
    "abstract_plan": range(0x4000, 0x7000),
    "model_native": range(0x7000, 0x10000),
}


def logical_token_id(namespace: str, local_id: int) -> int:
    """Map a local token id into its versioned, collision-free namespace."""
    token_range = TOKEN_ID_NAMESPACE_RANGES.get(namespace)
    if token_range is None:
        raise ValueError(f"unknown token-id namespace: {namespace!r}")
    if not 0 <= local_id < len(token_range):
        raise ValueError(f"token id {local_id} is outside {namespace!r} capacity")
    return token_range.start + local_id

STRUCTURAL_TOKENS = frozenset(
    {
        "root",
        "Stack",
        "Card",
        "CardHeader",
        "TextContent",
        "Button",
        "Buttons",
        "Input",
        "Form",
        "FormControl",
        "Label",
        "TextArea",
        "Select",
        "SelectItem",
        "CheckBoxGroup",
        "CheckBoxItem",
        "RadioGroup",
        "RadioItem",
        "SwitchGroup",
        "SwitchItem",
        "Slider",
        "DatePicker",
        "Image",
        "ImageBlock",
        "ImageGallery",
        "Modal",
        "Tabs",
        "TabItem",
        "Callout",
        "TextCallout",
        "Separator",
        "Table",
        "Col",
        "column",
        "row",
        "none",
        "xs",
        "s",
        "m",
        "l",
        "xl",
        "2xl",
        "primary",
        "secondary",
        "tertiary",
        "small",
        "default",
        "large",
        "small-heavy",
        "large-heavy",
        "=",
        "(",
        ")",
        "[",
        "]",
        ",",
        "\n",
        " ",
        '"',
        "null",
        "true",
        "false",
        "Query",
        "Mutation",
        "Action",
        "@Run",
        "@Set",
        "@Reset",
        "@ToAssistant",
        "@OpenUrl",
        "@Count",
        "@First",
        "@Last",
        "@Sum",
        "@Avg",
        "@Min",
        "@Max",
        "@Sort",
        "@Filter",
        "@Round",
        "@Abs",
        "@Floor",
        "@Ceil",
        "@Each",
        "$",
        "@",
        "{",
        "}",
        ":",
        ".",
        "?",
        "+",
        "-",
        "*",
        "/",
        "%",
        "!",
        "==",
        "!=",
        ">",
        "<",
        ">=",
        "<=",
        "&&",
        "||",
    }
)

PREFERRED_COMPONENT_NAMES = frozenset(
    {
        "Stack",
        "Card",
        "TextContent",
        "Button",
        "Input",
        "Form",
        "ImageBlock",
        "Modal",
        "Tabs",
        "Slider",
        "CheckBoxItem",
        "RadioItem",
        "SwitchItem",
        "DatePicker",
    }
)
