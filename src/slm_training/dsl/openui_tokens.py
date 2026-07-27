"""OpenUI structural / preferred token priors for constrained decode."""

from __future__ import annotations

# Logical token-id namespaces are deliberately separate from codec-local
# embedding rows.  Native, choice, and legacy tokenizers all retain their
# historical 0-based ids for checkpoint compatibility; consumers that combine
# token families must use these disjoint logical ranges instead.
TOKEN_ID_NAMESPACE_REGISTRY_VERSION = 2
TOKEN_ID_NAMESPACE_RANGES = {
    "control": range(0x0000, 0x1000),
    "openui": range(0x1000, 0x4000),
    "abstract_plan": range(0x4000, 0x7000),
    "model_native": range(0x7000, 0x10000),
    # Decode invariant I13 (docs/design/decode-invariants.md): the reserved
    # compute-ops vocabulary, known and shared by the encoder and the decoder.
    # Grammar symbols layer on top of it in their own namespaces above; natural
    # language sits above those and is strictly optional. Placed past the
    # existing codec ranges so no stored embedding row moves.
    "ops": range(0x10000, 0x11000),
}


def logical_token_id(namespace: str, local_id: int) -> int:
    """Map a local token id into its versioned, collision-free namespace."""
    token_range = TOKEN_ID_NAMESPACE_RANGES.get(namespace)
    if token_range is None:
        raise ValueError(f"unknown token-id namespace: {namespace!r}")
    if not 0 <= local_id < len(token_range):
        raise ValueError(f"token id {local_id} is outside {namespace!r} capacity")
    return token_range.start + local_id


# AP-016 (SLM-302): reserved, default-off abstract-plan latent codebook.
# Local id 0/1 are the begin/end delimiters; ids ``ABSTRACT_PLAN_SLOT_LOCAL_OFFSET``
# .. +M-1 are codebook slots. Nothing here enables generation on its own — a
# tokenizer only emits these ids when explicitly built with
# ``abstract_plan_slots > 0`` (see ``models.dsl_tokenizer``/``models.choice_tokenizer``).
# See ``slm_training.dsl.abstract_plan.AbstractPlanV1`` for the versioned contract.
ABSTRACT_PLAN_CODEBOOK_VERSION = 1
ABSTRACT_PLAN_BEGIN = "<beginabstract>"
ABSTRACT_PLAN_END = "<endabstract>"
ABSTRACT_PLAN_BEGIN_LOCAL_ID = 0
ABSTRACT_PLAN_END_LOCAL_ID = 1
ABSTRACT_PLAN_SLOT_LOCAL_OFFSET = 2
ABSTRACT_PLAN_CODEBOOK_SIZE = 64  # M: default slot count when enabled
MAX_ABSTRACT_PLAN_SLOTS = 128  # m_max: hard cap on slot_count
DEFAULT_ABSTRACT_PLAN_ROUNDS = 3  # T: default recurrence depth


def abstract_plan_slot_token(index: int) -> str:
    """Deterministic, intentionally non-interpretable text for codebook slot ``index``."""
    if index < 0:
        raise ValueError("abstract-plan slot index must be non-negative")
    return f"<ABS_{index}>"


def abstract_plan_logical_id(local_id: int) -> int:
    """Map a begin/end/slot local id into the collision-free ``abstract_plan`` namespace."""
    return logical_token_id("abstract_plan", local_id)


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
