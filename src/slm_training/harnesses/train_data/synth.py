"""Prompt / layout synthesis plugins for the training-data harness."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from slm_training.dsl.lang_core import parse
from slm_training.dsl.schema import ExampleRecord

_ROOT_STACK_RE = re.compile(
    r'^root\s*=\s*Stack\(\[(?P<children>[^\]]*)\](?P<rest>(?:,\s*"[^"]*")*)\)\s*$',
    re.M,
)
_PLACEHOLDER_RE = re.compile(r'":([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z0-9_.]+)"')


class PromptSynthesizer(Protocol):
    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        """Return zero or more additional records derived from ``record``."""


class NoopSynthesizer:
    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        return []


class TemplateSynthesizer:
    """Deterministic prompt paraphrases (same OpenUI target)."""

    TEMPLATES = (
        "Please generate UI for: {prompt}",
        "OpenUI layout request: {prompt}",
        "Design a screen that does the following — {prompt}",
    )

    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        out: list[ExampleRecord] = []
        for i, template in enumerate(self.TEMPLATES):
            out.append(
                ExampleRecord(
                    id=f"{record.id}_syn_{i}",
                    prompt=template.format(prompt=record.prompt),
                    openui=record.openui,
                    placeholders=list(record.placeholders),
                    split=record.split,
                    source=f"{record.source}+template",
                    meta={
                        **record.meta,
                        "synth": "template",
                        "parent_id": record.id,
                        "synth_index": i,
                    },
                    design_md=record.design_md,
                )
            )
        return out


class ComponentPromptSynthesizer:
    """Describe a target's component inventory and content concepts in prose."""

    @staticmethod
    def _quantity(name: str, count: int) -> str:
        name = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
        if count == 1:
            return f"one {name}"
        return f"{count} {name} components"

    @staticmethod
    def _join(items: list[str]) -> str:
        if len(items) < 2:
            return "".join(items)
        if len(items) == 2:
            return " and ".join(items)
        return f"{', '.join(items[:-1])}, and {items[-1]}"

    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        from slm_training.data.quality import component_counts

        counts = component_counts(record.openui)
        if not counts:
            return []
        inventory = [
            self._quantity(name, count) for name, count in counts.items()
        ]
        concepts = list(
            dict.fromkeys(
                placeholder.rsplit(".", 1)[-1].lstrip(":").replace("_", " ")
                for placeholder in record.placeholders
                if placeholder.strip(":")
            )
        )
        prompt = f"Build an OpenUI layout with {self._join(inventory)}."
        if concepts:
            prompt += f" Include content slots for {self._join(concepts)}."
        return [
            ExampleRecord(
                id=f"{record.id}_component_prompt",
                prompt=prompt,
                openui=record.openui,
                placeholders=list(record.placeholders),
                split=record.split,
                source=f"{record.source}+component_prompt",
                meta={
                    **record.meta,
                    "synth": "component_prompt",
                    "task": "generation",
                    "parent_id": record.id,
                    "component_inventory": dict(counts),
                    "content_concepts": concepts,
                },
                design_md=record.design_md,
            )
        ]


class SemanticSlotSynthesizer:
    """Rename generation slots with diverse schema-role semantics."""

    _ALIASES = {
        "text": ("title", "body", "heading", "caption", "value"),
        "label": ("label", "action", "submit", "confirm", "create"),
        "placeholder": ("email", "name", "search", "query", "value"),
        "title": ("title", "heading", "name"),
        "body": ("body", "description", "details"),
        "description": ("description", "help", "details"),
        "trigger": ("tab", "section", "option"),
        "src": ("image", "photo", "thumbnail"),
        "alt": ("alt", "image_description"),
    }

    @classmethod
    def _slot_owners(cls, source: str) -> dict[str, tuple[str, str]]:
        owners: dict[str, tuple[str, str]] = {}

        def walk(value: object, component: str | None = None, prop: str = "") -> None:
            if isinstance(value, dict):
                owner = (
                    str(value["typeName"])
                    if value.get("type") == "element" and value.get("typeName")
                    else component
                )
                props = value.get("props")
                if isinstance(props, dict):
                    for name, child in props.items():
                        walk(child, owner, str(name))
                for name, child in value.items():
                    if name != "props":
                        walk(child, owner, prop)
            elif isinstance(value, list):
                for child in value:
                    walk(child, component, prop)
            elif (
                isinstance(value, str)
                and value.startswith(":")
                and component is not None
            ):
                owners.setdefault(value, (component, prop))

        walk(parse(source).root)
        return owners

    @classmethod
    def _replacement(
        cls,
        *,
        record_id: str,
        slot: str,
        component: str,
        prop: str,
        occurrence: int,
    ) -> str:
        aliases = cls._ALIASES.get(prop.lower(), (prop.lower() or "content",))
        digest = hashlib.sha256(f"{record_id}:{slot}".encode()).digest()
        role = aliases[int.from_bytes(digest[:4], "big") % len(aliases)]
        namespace = re.sub(r"(?<!^)(?=[A-Z])", "_", component).lower()
        suffix = "" if occurrence == 1 else f"_{occurrence}"
        return f":{namespace}.{role}{suffix}"

    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        if str(record.meta.get("task") or "") != "generation":
            return []
        owners = self._slot_owners(record.openui)
        if not owners:
            return []
        occurrences: defaultdict[tuple[str, str], int] = defaultdict(int)
        replacements: dict[str, str] = {}
        for slot in record.placeholders:
            owner = owners.get(slot)
            if owner is None:
                continue
            occurrences[owner] += 1
            replacements[slot] = self._replacement(
                record_id=record.id,
                slot=slot,
                component=owner[0],
                prop=owner[1],
                occurrence=occurrences[owner],
            )
        if not replacements or len(set(replacements.values())) != len(replacements):
            return []

        def replace(text: str) -> str:
            for old in sorted(replacements, key=len, reverse=True):
                text = text.replace(old, replacements[old])
            return text

        return [
            ExampleRecord(
                id=f"{record.id}_semantic_slots",
                prompt=replace(record.prompt),
                openui=replace(record.openui),
                placeholders=[replacements.get(slot, slot) for slot in record.placeholders],
                split=record.split,
                source=f"{record.source}+semantic_slots",
                meta={
                    **record.meta,
                    "synth": "semantic_slots",
                    "parent_id": record.id,
                    "slot_role_map": replacements,
                },
                design_md=record.design_md,
            )
        ]


class LayoutAugmentSynthesizer:
    """
    Deterministic structural augmentations.

    Produces at most two variants:
    1. Flip Stack direction column<->row when a single root Stack exists
    2. Append a secondary CTA button sibling when the root has fewer than 4 children
    """

    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        out: list[ExampleRecord] = []
        flipped = self._flip_direction(record)
        if flipped:
            out.append(flipped)
        with_cta = self._append_cta(record)
        if with_cta:
            out.append(with_cta)
        return out

    def _flip_direction(self, record: ExampleRecord) -> ExampleRecord | None:
        match = _ROOT_STACK_RE.search(record.openui)
        if not match:
            return None
        rest = match.group("rest") or ""
        if '"column"' in rest:
            new_rest = rest.replace('"column"', '"row"', 1)
            direction = "row"
        elif '"row"' in rest:
            new_rest = rest.replace('"row"', '"column"', 1)
            direction = "column"
        elif rest.strip() == "":
            # Bare Stack([...]) — treat as column default, make explicit row.
            new_rest = ', "row", "m"'
            direction = "row"
        else:
            return None
        children = match.group("children")
        new_root = f"root = Stack([{children}]{new_rest})"
        openui = _ROOT_STACK_RE.sub(new_root, record.openui, count=1)
        if openui == record.openui:
            return None
        return ExampleRecord(
            id=f"{record.id}_aug_dir",
            prompt=f"{record.prompt} Prefer a {direction} Stack layout.",
            openui=openui,
            placeholders=list(record.placeholders),
            split=record.split,
            source=f"{record.source}+aug",
            meta={
                **record.meta,
                "synth": "layout_augment",
                "aug": "flip_direction",
                "parent_id": record.id,
            },
            design_md=record.design_md,
        )

    def _append_cta(self, record: ExampleRecord) -> ExampleRecord | None:
        match = _ROOT_STACK_RE.search(record.openui)
        if not match:
            return None
        children = [c.strip() for c in match.group("children").split(",") if c.strip()]
        if len(children) >= 4 or "cta_aug" in children:
            return None
        if any("Button(" in line for line in record.openui.splitlines()):
            # Already has a button — skip to avoid redundant CTAs.
            return None
        rest = match.group("rest") or ', "column", "m"'
        new_children = ", ".join([*children, "cta_aug"])
        new_root = f"root = Stack([{new_children}]{rest})"
        openui = _ROOT_STACK_RE.sub(new_root, record.openui, count=1)
        openui = openui.rstrip() + '\ncta_aug = Button(":cta_aug.label")'
        placeholders = list(record.placeholders)
        if ":cta_aug.label" not in placeholders:
            placeholders.append(":cta_aug.label")
        return ExampleRecord(
            id=f"{record.id}_aug_cta",
            prompt=f"{record.prompt} Include a clear call-to-action button.",
            openui=openui,
            placeholders=placeholders,
            split=record.split,
            source=f"{record.source}+aug",
            meta={
                **record.meta,
                "synth": "layout_augment",
                "aug": "append_cta",
                "parent_id": record.id,
            },
            design_md=record.design_md,
        )


class CardHierarchySynthesizer:
    """Wrap each root Stack section in a Card, with matching prompt evidence."""

    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        match = _ROOT_STACK_RE.search(record.openui)
        if not match or "Card(" in record.openui:
            return []
        children = [child.strip() for child in match.group("children").split(",")]
        if (
            not 1 <= len(children) <= 6
            or not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", child) for child in children)
            or any(f"card_aug_{index}" in record.openui for index in range(len(children)))
        ):
            return []
        cards = [f"card_aug_{index}" for index in range(len(children))]
        rest = match.group("rest") or ', "column", "m"'
        root = f"root = Stack([{', '.join(cards)}]{rest})"
        openui = _ROOT_STACK_RE.sub(root, record.openui, count=1).rstrip()
        openui += "\n" + "\n".join(
            f"{card} = Card([{child}])" for card, child in zip(cards, children)
        )
        return [
            ExampleRecord(
                id=f"{record.id}_card_sections",
                prompt=(
                    f"{record.prompt} Present each of the {len(cards)} primary "
                    f"section{'s' if len(cards) != 1 else ''} in its own Card."
                ),
                openui=openui,
                placeholders=list(record.placeholders),
                split=record.split,
                source=f"{record.source}+card_hierarchy",
                meta={
                    **record.meta,
                    "synth": "layout_augment",
                    "aug": "card_hierarchy",
                    "parent_id": record.id,
                    "card_count": len(cards),
                },
                design_md=record.design_md,
            )
        ]


class NamespaceAugmentSynthesizer:
    """Re-prefix placeholders with a deterministic namespace (:acme.hero.title)."""

    PREFIXES = ("acme", "beta", "gamma", "delta", "omega", "nova")

    def _pick_prefix(self, record_id: str) -> str:
        digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
        return self.PREFIXES[int(digest[:8], 16) % len(self.PREFIXES)]

    def _reprefix(self, token: str, prefix: str) -> str:
        if not token.startswith(":"):
            return token
        body = token[1:]
        if "." not in body:
            return f":{prefix}.{body}"
        return f":{prefix}.{body}"

    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        prefix = self._pick_prefix(record.id)
        openui = record.openui

        def _sub(match: re.Match[str]) -> str:
            first = match.group(1)
            rest = match.group(2)
            return f'":{prefix}.{first}.{rest}"'

        new_openui = _PLACEHOLDER_RE.sub(_sub, openui)
        if new_openui == openui:
            return []
        new_placeholders = [
            self._reprefix(ph, prefix) for ph in (record.placeholders or [])
        ]
        return [
            ExampleRecord(
                id=f"{record.id}_ns_{prefix}",
                prompt=record.prompt,
                openui=new_openui,
                placeholders=new_placeholders,
                split=record.split,
                source=f"{record.source}+namespace",
                meta={
                    **record.meta,
                    "synth": "namespace_augment",
                    "namespace_prefix": prefix,
                    "parent_id": record.id,
                },
                design_md=record.design_md,
            )
        ]


class FrozenArtifactSynthesizer:
    """Emit paraphrase + abstraction-ladder rows from committed frontier artifacts.

    Reads ``src/slm_training/resources/frontier/<gold_id>.<hash8>.json`` (agent-skill output), binds
    it to the gold by content hash *and* structural fingerprint (faithfulness), and
    drops silently on a mismatch (a changed gold → regenerate). Rows keep the gold's
    placeholder skeleton as the target; only the prompt varies. No model call — the
    build stays deterministic and re-validates every row downstream. Edit / vision
    sections of the bundle are consumed by later stages (P4 / P7).
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = root

    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        if record.split != "train":
            return []
        from slm_training.data.frontier import gold_content_hash, load_artifact
        from slm_training.data.house_style import resolve_target
        from slm_training.data.ladder import GroundingError, build_rung
        from slm_training.data.leakage import fingerprint_openui_structure

        gold_hash = gold_content_hash(record.openui, record.prompt)
        artifact = load_artifact(record.id, gold_hash, root=self._root)
        if artifact is None:
            return []
        # Faithfulness: the described skeleton must be structurally the gold.
        if fingerprint_openui_structure(artifact.skeleton_openui) != (
            fingerprint_openui_structure(record.openui)
        ):
            return []
        out: list[ExampleRecord] = []
        for i, paraphrase in enumerate(artifact.paraphrases):
            if paraphrase.strip():
                out.append(
                    self._derive(
                        record,
                        rid=f"{record.id}_fd_{i}",
                        prompt=paraphrase,
                        family="frontier_described",
                        synth="frontier_described",
                    )
                )
        for level, text in sorted(artifact.ladder.items()):
            if text.strip():
                try:
                    rung = build_rung(level, text, record.openui)
                    resolution = resolve_target(text, (record.openui,), rung.level)
                except (GroundingError, ValueError):
                    continue
                rung_meta = rung.to_meta()
                if rung.target_determinacy.value == "house_style":
                    rung_meta["house_style_resolution"] = resolution.to_meta()
                out.append(
                    self._derive(
                        record,
                        rid=f"{record.id}_ladder_{level}",
                        prompt=text,
                        family=rung.family,
                        synth=rung.family,
                        openui=resolution.target,
                        extra_meta=rung_meta,
                    )
                )
        return out

    @staticmethod
    def _derive(
        record: ExampleRecord,
        *,
        rid: str,
        prompt: str,
        family: str,
        synth: str,
        openui: str | None = None,
        extra_meta: dict | None = None,
    ) -> ExampleRecord:
        return ExampleRecord(
            id=rid,
            prompt=prompt,
            openui=openui or record.openui,
            placeholders=list(record.placeholders),
            split=record.split,
            source=f"{record.source}+{family}",
            meta={
                **record.meta,
                "synth": synth,
                "task": "generation",
                "parent_id": record.id,
                **(extra_meta or {}),
            },
            design_md=record.design_md,
        )


class QualitySynthesizer:
    """Compose template paraphrases + layout augments (deterministic, ordered)."""

    def __init__(self) -> None:
        self._template = TemplateSynthesizer()
        self._layout = LayoutAugmentSynthesizer()

    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        # Layout first (new OpenUI), then light prompt paraphrases of the *original* only.
        return [*self._layout.expand(record), *self._template.expand(record)]


def get_synthesizer(name: str) -> PromptSynthesizer:
    if name in {"none", "noop", "off"}:
        return NoopSynthesizer()
    if name in {"template", "templates"}:
        return TemplateSynthesizer()
    if name in {"component", "component_prompt", "inventory"}:
        return ComponentPromptSynthesizer()
    if name in {"semantic_slot", "semantic_slots", "slot_roles"}:
        return SemanticSlotSynthesizer()
    if name in {"layout", "layout_augment", "aug"}:
        return LayoutAugmentSynthesizer()
    if name in {"card", "card_hierarchy", "card_sections"}:
        return CardHierarchySynthesizer()
    if name in {"namespace", "namespace_augment", "ns"}:
        return NamespaceAugmentSynthesizer()
    if name in {"quality", "full", "default"}:
        return QualitySynthesizer()
    if name in {"frontier", "frontier_artifact", "frontier_described"}:
        return FrozenArtifactSynthesizer()
    raise ValueError(f"unknown synthesizer {name!r}")
