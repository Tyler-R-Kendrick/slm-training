"""Capability-aware prompt rendering from SemanticFrameV1.

Offline, deterministic renderers turn frame facts into ordinary simplified-NL
prompts. Production surfaces stay free of product/DSL triggers and never copy
the target. Provenance stays off the prompt text.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace
from typing import Any

from slm_training.dsl.semantic_frame import SemanticFactV1, SemanticFrameV1
from slm_training.harness_core.lineage.records import canonical_json, content_sha
from slm_training.harnesses.synthesis_plan import (
    PromptSurface,
    PromptSurfacePolicy,
    RendererFamily,
)
from slm_training.harnesses.train_data.frame_paraphrases import (
    FrameLeakDetectorV1,
    render_fact_phrase,
)

SCHEMA_VERSION = "semantic_prompts/v1"
PROVIDER_ID = "offline_fixture_renderer"
RENDERER_VERSION = "v1"

CUE_INTERVENTIONS: tuple[str, ...] = (
    "remove_technical",
    "add_harmless_wrapper",
    "substitute_format_noun",
    "vary_mood",
    "change_punctuation",
    "change_politeness",
)

_BINDER_RE = re.compile(r"^([A-Za-z_][\w]*)\s*=", re.MULTILINE)
_MARKER_RE = re.compile(r"(?<![\w])[:$][A-Za-z_][\w]*(?:\.[\w]+)*")
_HASH_RE = re.compile(r"\b[0-9a-f]{32,}\b", re.IGNORECASE)
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TECH_RE = re.compile(
    r"\b(?:components?|widgets?|controls?|schemas?|grammar|tokens?|nodes?)\b",
    re.IGNORECASE,
)
_COMMON_BINDERS = frozenset({"root", "title", "label", "header", "body", "text"})
_FAMILY_IDS = frozenset(item.value for item in RendererFamily)
_PROVENANCE_KEYS = (
    "ast_path",
    "schema_field",
    "canonical_source",
    "schema_fingerprint",
    "frame_fingerprint",
    "prompt_digest",
    "response_digest",
    "cache_key",
    "category",
    "provider_id",
    "renderer_family",
    "renderer_version",
    "invariance_group_id",
    "required_fact_ids",
    "house_style_fact_ids",
    "forbidden_fact_ids",
)
_NOUN_SWAPS: tuple[tuple[str, str], ...] = (
    ("call to action", "button"),
    ("heading", "title"),
    ("title", "heading"),
    ("button", "action"),
    ("action", "button"),
    ("copy", "note"),
)
_USER_NOUNS: dict[str, tuple[str, ...]] = {
    "TextContent": ("heading", "title", "copy"),
    "Button": ("button", "call to action", "action"),
    "Form": ("form",),
    "Input": ("field",),
    "Image": ("image",),
    "ImageBlock": ("image",),
    "Tabs": ("tabs",),
    "Card": ("grouped card",),
    "Callout": ("notice",),
    "Separator": ("divider",),
    "Modal": ("dialog",),
    "Table": ("table",),
}
_TASK_USE: dict[str, str] = {
    "heading": "read a short title",
    "title": "read a short title",
    "copy": "read a short note",
    "button": "tap a button",
    "call to action": "tap a call to action",
    "action": "tap an action",
    "form": "fill a form",
    "field": "type in a field",
    "image": "see an image",
    "tabs": "switch tabs",
    "grouped card": "scan a grouped card",
    "notice": "read a notice",
    "divider": "see a divider",
    "dialog": "confirm in a dialog",
    "table": "scan a table",
}
_SURFACE_FAMILIES: dict[PromptSurface, frozenset[RendererFamily]] = {
    PromptSurface.SIMPLIFIED_NL: frozenset(
        {
            RendererFamily.CONCISE_INTENT,
            RendererFamily.TASK_CONTEXT,
            RendererFamily.COMPOSITIONAL_REQUIREMENT,
            RendererFamily.PARAPHRASE,
        }
    ),
    PromptSurface.GRAMMAR_SCHEMA: frozenset({RendererFamily.GRAMMAR_SCHEMA}),
    PromptSurface.TRANSFORMATION_NL: frozenset({RendererFamily.TRANSFORMATION}),
}


@dataclass(frozen=True)
class PromptCandidate:
    prompt_text: str
    renderer_family: str
    renderer_version: str
    surface: str
    required_fact_ids: tuple[str, ...]
    house_style_fact_ids: tuple[str, ...]
    forbidden_fact_ids: tuple[str, ...]
    cue_intervention: str | None
    invariance_group_id: str
    provider_id: str
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_text": self.prompt_text,
            "renderer_family": self.renderer_family,
            "renderer_version": self.renderer_version,
            "surface": self.surface,
            "required_fact_ids": list(self.required_fact_ids),
            "house_style_fact_ids": list(self.house_style_fact_ids),
            "forbidden_fact_ids": list(self.forbidden_fact_ids),
            "cue_intervention": self.cue_intervention,
            "invariance_group_id": self.invariance_group_id,
            "provider_id": self.provider_id,
            "provenance": dict(self.provenance),
        }

    @property
    def sha(self) -> str:
        return content_sha(self.to_dict())


@dataclass(frozen=True)
class PromptRenderResult:
    candidates: tuple[PromptCandidate, ...]
    rejections: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [item.to_dict() for item in self.candidates],
            "rejections": [dict(item) for item in self.rejections],
        }


def _fact_id(fact: SemanticFactV1) -> str:
    path = ".".join(str(part) for part in fact.ast_path) or "root"
    value = "null" if fact.value is None else str(fact.value)
    return f"{fact.predicate}@{path}={value}"


def _split_camel(name: str) -> str:
    return _CAMEL_RE.sub(" ", name).lower()


def _word(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE) is not None


def _component_types(frame: SemanticFrameV1) -> tuple[str, ...]:
    return tuple(node.type_name for node in frame.nodes if node.kind == "element")


def _direction(frame: SemanticFrameV1) -> str | None:
    for fact in frame.facts:
        if fact.predicate == "prop_value" and fact.ast_path[-1:] == ("direction",):
            return str(fact.value)
    return None


def _binders(frame: SemanticFrameV1) -> tuple[str, ...]:
    return tuple(_BINDER_RE.findall(frame.canonical_source))


def _an(noun: str) -> str:
    article = "an" if noun[:1].lower() in "aeiou" else "a"
    return f"{article} {noun}"


def _user_nouns(
    types: tuple[str, ...], variant: int, *, canonical: bool = False
) -> tuple[str, ...]:
    nouns: list[str] = []
    kid_index = 0
    for type_name in types:
        if type_name == "Stack":
            continue
        options = _USER_NOUNS.get(type_name)
        if options is None:
            nouns.append("section")
        else:
            offset = 0 if canonical else variant + kid_index
            nouns.append(options[offset % len(options)])
        kid_index += 1
    return tuple(nouns)


def _layout_clause(direction: str | None, nouns: tuple[str, ...], variant: int) -> str:
    if len(nouns) < 2:
        return f"with {nouns[0]}" if nouns else "ready to use"
    left, right = nouns[0], nouns[1]
    if direction == "row":
        options = (
            f"place the {left} and {right} side by side",
            f"keep the {left} next to the {right}",
            f"put the {left} beside the {right}",
        )
    else:
        options = (
            f"put the {left} above the {right}",
            f"keep the {right} under the {left}",
            f"sit the {left} over the {right}",
        )
    return options[variant % len(options)]


def _render_family(
    family: RendererFamily,
    frame: SemanticFrameV1,
    *,
    variant: int,
) -> str:
    types = _component_types(frame)
    nouns = _user_nouns(
        types, variant, canonical=family is RendererFamily.CONCISE_INTENT
    )
    direction = _direction(frame)
    primary = nouns[0] if nouns else "content"
    secondary = nouns[1] if len(nouns) > 1 else None
    layout = _layout_clause(direction, nouns, variant)
    if family is RendererFamily.CONCISE_INTENT:
        if secondary and direction == "row":
            return f"Need {primary} with {secondary} beside it."
        if secondary:
            return f"Need {primary} with {secondary} underneath."
        return f"Need {primary}."
    if family is RendererFamily.TASK_CONTEXT:
        uses = [_TASK_USE.get(noun, f"use {noun}") for noun in nouns[:2] or (primary,)]
        joined = " and ".join(uses)
        scene = ("checkout", "welcome", "sign-in")[variant % 3]
        return f"I need a {scene} so customers can {joined}."
    if family is RendererFamily.COMPOSITIONAL_REQUIREMENT:
        clauses = [_an(noun) for noun in nouns[:3]] or [_an(primary)]
        if secondary:
            tail = "beside it" if direction == "row" else "below it"
            return f"Show {clauses[0]}, then {clauses[1]} {tail}."
        return f"Show {clauses[0]}."
    if family is RendererFamily.PARAPHRASE:
        if secondary and direction == "row":
            return f"Put {primary} beside {secondary}."
        if secondary:
            return f"Put {primary} above {secondary} people can tap."
        return f"Put {primary} people can use."
    if family is RendererFamily.GRAMMAR_SCHEMA:
        phrases = [
            render_fact_phrase(fact)
            for fact in frame.facts_by_category("required")
            if fact.predicate != "content_placeholder_id"
        ]
        return "OpenUI grammar schema: " + "; ".join(phrases) + "."
    if family is RendererFamily.TRANSFORMATION:
        return f"Change this so you {layout}."
    raise ValueError(f"unsupported renderer family: {family.value}")


def _inventory_listed(text: str, types: tuple[str, ...]) -> bool:
    if not types:
        return False
    lowered = text.lower()
    hits = 0
    for type_name in types:
        names = {type_name.lower(), _split_camel(type_name)}
        if any(_word(lowered, name) for name in names if name):
            hits += 1
    return hits == len(set(types))


def _serialized_leak(text: str, frame: SemanticFrameV1) -> str | None:
    source = frame.canonical_source
    for line in source.splitlines():
        fragment = line.strip()
        if len(fragment) >= 8 and fragment in text:
            return fragment
    for type_name in _component_types(frame):
        if f"{type_name}(" in text:
            return f"{type_name}("
    return None


def _reject_reasons(
    text: str,
    *,
    frame: SemanticFrameV1,
    policy: PromptSurfacePolicy,
    detector: FrameLeakDetectorV1,
) -> tuple[str, ...]:
    if policy.surface is PromptSurface.GRAMMAR_SCHEMA:
        return ()
    reasons: list[str] = []
    ok, leak_reasons = detector.check(text)
    if not ok:
        reasons.extend(leak_reasons)
    lowered = text.lower()
    for cue in policy.hard_forbidden_cues:
        if cue.lower() in lowered:
            reasons.append(f"hard_forbidden_cue:{cue}")
    if re.search(r"root\s*=", text):
        reasons.append("binder_code:root")
    for binder in _binders(frame):
        if re.search(rf"\b{re.escape(binder)}\s*=", text):
            reasons.append(f"binder_code:{binder}")
        elif binder.lower() not in _COMMON_BINDERS and _word(text, binder):
            reasons.append(f"binder_name:{binder}")
    if _MARKER_RE.search(text):
        reasons.append("marker_surface")
    if _HASH_RE.search(text):
        reasons.append("hash")
    for family_id in _FAMILY_IDS:
        if family_id in text:
            reasons.append(f"family_id:{family_id}")
    for key in _PROVENANCE_KEYS:
        if key in text:
            reasons.append(f"provenance_key:{key}")
    fragment = _serialized_leak(text, frame)
    if fragment is not None:
        reasons.append(f"target_fragment:{fragment}")
    if _inventory_listed(text, _component_types(frame)):
        reasons.append("complete_inventory")
    # de-dupe while keeping order
    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    return tuple(ordered)


def _invariance_group_id(
    root_identity: str,
    required_fact_ids: tuple[str, ...],
    surface: str,
) -> str:
    return content_sha(
        {
            "root_identity": root_identity,
            "required_fact_ids": list(required_fact_ids),
            "surface": surface,
        }
    )


def _candidate(
    text: str,
    *,
    family: RendererFamily,
    policy: PromptSurfacePolicy,
    required_ids: tuple[str, ...],
    house_ids: tuple[str, ...],
    forbidden_ids: tuple[str, ...],
    required_fact_phrases: tuple[str, ...],
    target_fact_phrases: tuple[str, ...],
    forbidden_fact_phrases: tuple[str, ...],
    root_identity: str,
    seed: int,
    intervention: str | None = None,
) -> PromptCandidate:
    surface = policy.surface.value
    return PromptCandidate(
        prompt_text=text,
        renderer_family=family.value,
        renderer_version=RENDERER_VERSION,
        surface=surface,
        required_fact_ids=required_ids,
        house_style_fact_ids=house_ids,
        forbidden_fact_ids=forbidden_ids,
        cue_intervention=intervention,
        invariance_group_id=_invariance_group_id(root_identity, required_ids, surface),
        provider_id=PROVIDER_ID,
        provenance={
            "schema_version": SCHEMA_VERSION,
            "root_identity": root_identity,
            "seed": seed,
            "surface": surface,
            "renderer_family": family.value,
            "renderer_version": RENDERER_VERSION,
            "provider_id": PROVIDER_ID,
            "required_fact_phrases": list(required_fact_phrases),
            "target_fact_phrases": list(target_fact_phrases),
            "forbidden_fact_phrases": list(forbidden_fact_phrases),
            "canonical": canonical_json(
                {"root_identity": root_identity, "surface": surface}
            ),
        },
    )


def _has_generic(text: str, cues: tuple[str, ...]) -> bool:
    return any(_word(text, cue) for cue in cues)


def _rejection(
    reason: str,
    *,
    family: RendererFamily | str,
    detail: str,
    prompt_text: str = "",
) -> dict[str, Any]:
    name = family.value if isinstance(family, RendererFamily) else family
    payload = {"reason": reason, "detail": detail, "renderer_family": name}
    if prompt_text:
        payload["prompt_text"] = prompt_text
    return payload


def render_prompt_candidates(
    frame: SemanticFrameV1,
    policy: PromptSurfacePolicy,
    *,
    root_identity: str,
    seed: int = 0,
) -> PromptRenderResult:
    required_facts = frame.facts_by_category("required")
    house_facts = frame.facts_by_category("optional")
    forbidden_facts = frame.facts_by_category("forbidden")
    required_ids = tuple(_fact_id(fact) for fact in required_facts)
    house_ids = tuple(_fact_id(fact) for fact in house_facts)
    forbidden_ids = tuple(_fact_id(fact) for fact in forbidden_facts)
    target_fact_phrases = tuple(render_fact_phrase(fact) for fact in required_facts)
    forbidden_fact_phrases = tuple(render_fact_phrase(fact) for fact in forbidden_facts)
    detector = FrameLeakDetectorV1.from_frame(frame)
    allowed = [
        family
        for family in policy.renderer_families
        if family in _SURFACE_FAMILIES.get(policy.surface, frozenset())
    ]
    rejections: list[dict[str, Any]] = []
    for family in policy.renderer_families:
        if family not in allowed:
            rejections.append(
                _rejection(
                    "family_surface_mismatch",
                    family=family,
                    detail=f"{family.value} is not legal on {policy.surface.value}",
                )
            )
    if not allowed:
        return PromptRenderResult((), tuple(rejections))

    rng = random.Random(seed)
    start = rng.randrange(len(allowed))
    order = allowed[start:] + allowed[:start]
    admitted: list[PromptCandidate] = []
    attempts = max(policy.prompts_per_root * len(order) * 3, policy.prompts_per_root)
    for attempt in range(attempts):
        if len(admitted) >= policy.prompts_per_root:
            break
        family = order[attempt % len(order)]
        text = _render_family(family, frame, variant=attempt // len(order) + seed)
        reasons = _reject_reasons(text, frame=frame, policy=policy, detector=detector)
        if reasons:
            rejections.append(
                _rejection(
                    reasons[0],
                    family=family,
                    detail=",".join(reasons),
                    prompt_text=text,
                )
            )
            continue
        admitted.append(
            _candidate(
                text,
                family=family,
                policy=policy,
                required_ids=required_ids,
                house_ids=house_ids,
                forbidden_ids=forbidden_ids,
                required_fact_phrases=tuple(
                    phrase
                    for phrase in target_fact_phrases
                    if phrase.casefold() in text.casefold()
                ),
                target_fact_phrases=target_fact_phrases,
                forbidden_fact_phrases=forbidden_fact_phrases,
                root_identity=root_identity,
                seed=seed,
            )
        )

    need_plain = (
        policy.prompts_per_root >= 2
        and len(policy.renderer_families) >= 4
        and policy.balanced_generic_cues
        and admitted
        and not any(
            not _has_generic(item.prompt_text, policy.balanced_generic_cues)
            for item in admitted
        )
    )
    if need_plain:
        replacement = None
        for family in order:
            text = _render_family(family, frame, variant=97 + seed)
            if _has_generic(text, policy.balanced_generic_cues):
                continue
            reasons = _reject_reasons(
                text, frame=frame, policy=policy, detector=detector
            )
            if reasons:
                continue
            replacement = _candidate(
                text,
                family=family,
                policy=policy,
                required_ids=required_ids,
                house_ids=house_ids,
                forbidden_ids=forbidden_ids,
                required_fact_phrases=tuple(
                    phrase
                    for phrase in target_fact_phrases
                    if phrase.casefold() in text.casefold()
                ),
                target_fact_phrases=target_fact_phrases,
                forbidden_fact_phrases=forbidden_fact_phrases,
                root_identity=root_identity,
                seed=seed,
            )
            break
        if replacement is not None:
            admitted[-1] = replacement
        else:
            rejections.append(
                _rejection(
                    "missing_no_generic_cue",
                    family=order[0],
                    detail="no admitted prompt omitted balanced generic cues",
                )
            )

    return PromptRenderResult(tuple(admitted), tuple(rejections))


def _swap_noun(text: str) -> str:
    lowered = text.lower()
    for source, dest in _NOUN_SWAPS:
        match = re.search(rf"\b{re.escape(source)}\b", lowered)
        if match is None:
            continue
        start, end = match.span()
        return text[:start] + dest + text[end:]
    return text


def _vary_mood(text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith("?"):
        return stripped[:-1] + "."
    body = stripped[:-1] if stripped.endswith((".", "!")) else stripped
    prefixes = (
        ("Need ", "Can we have "),
        ("I need ", "Can I have "),
        ("Show ", "Can we show "),
        ("Put ", "Can we put "),
        ("Change this so ", "Can we change this so "),
        ("Please ", "Could you "),
    )
    for source, dest in prefixes:
        if body.startswith(source):
            return dest + body[len(source) :] + "?"
    return f"Can we have {body[0].lower() + body[1:]}?"


def _change_punctuation(text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith("."):
        return stripped[:-1] + "!"
    if stripped.endswith("!"):
        return stripped[:-1] + "."
    if stripped.endswith("?"):
        return stripped[:-1] + "."
    return stripped + "."


def _change_politeness(text: str) -> str:
    if text.lower().startswith("please "):
        rest = text[7:]
        return rest[:1].upper() + rest[1:]
    return "Please " + text[:1].lower() + text[1:]


def _remove_technical(text: str) -> str:
    cleaned = _TECH_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
    return cleaned or text


def apply_cue_intervention(
    candidate: PromptCandidate,
    intervention: str,
    *,
    frame: SemanticFrameV1,
) -> PromptCandidate:
    """Same-target invariance. Label stays on ``invariance_group_id``."""
    if intervention not in CUE_INTERVENTIONS:
        raise ValueError(f"unsupported cue intervention: {intervention}")
    text = candidate.prompt_text
    if intervention == "remove_technical":
        text = _remove_technical(text)
    elif intervention == "add_harmless_wrapper":
        text = "For a small shop, " + text[:1].lower() + text[1:]
    elif intervention == "substitute_format_noun":
        text = _swap_noun(text)
    elif intervention == "vary_mood":
        text = _vary_mood(text)
    elif intervention == "change_punctuation":
        text = _change_punctuation(text)
    else:
        text = _change_politeness(text)

    surface = PromptSurface(candidate.surface)
    if surface is not PromptSurface.GRAMMAR_SCHEMA:
        policy = PromptSurfacePolicy(
            surface=surface,
            prompts_per_root=1,
            renderer_families=(RendererFamily(candidate.renderer_family),),
            hard_forbidden_cues=(
                "openui",
                "dsl",
                "ast",
                "component inventory",
                "declarations",
                "reference graph",
                "placeholder",
            ),
            balanced_generic_cues=(),
        )
        reasons = _reject_reasons(
            text,
            frame=frame,
            policy=policy,
            detector=FrameLeakDetectorV1.from_frame(frame),
        )
        if reasons:
            raise ValueError(f"intervention leaked target: {reasons[0]}")

    provenance = dict(candidate.provenance)
    provenance["cue_intervention"] = intervention
    return replace(
        candidate,
        prompt_text=text,
        cue_intervention=intervention,
        provenance=provenance,
    )
