"""FrameParaphraseSetV1: style-planned, leak-checked prompt-only paraphrases.

DSH2-04 (SLM-365): generate multiple natural-language prompt surfaces for one
verified :class:`~slm_training.dsl.semantic_frame.SemanticFrameV1` (DSH2-02)
through the provenance-complete provider contract of DSH2-03
(:mod:`slm_training.dsl.paraphrase_provenance`) -- without leaking target DSL
syntax, raw marker surfaces, answer identity, or provenance fields into the
prompts.

**Facts only, never the answer.** Every prompt is constructed from the
frame's declared facts (entities/roles/relations/order/cardinality/closed
values), rendered to plain-language fact phrases by this module. The
canonical answer text (``SemanticFrameV1.canonical_source``) is used only to
*fingerprint* the answer a prompt links to and to *build the leak detector*;
its surface never enters a prompt payload. Targets stay pure canonical
symbol-only AST surfaces.

**Declared detector.** :class:`FrameLeakDetectorV1` is built from the frame
and rejects any prompt containing: component production forms (``Stack(``,
``Card(``, ``TextContent(`` and their exact-case bare tokens), raw marker
surfaces (``:ident``/``$ident`` codec forms, including the answer's exact
placeholder values), answer digests, and provenance field names.

**Grounding floor / stop rule.** :func:`enforce_grounding_floor` accepts a
candidate row only when it covers every required fact, states no forbidden
fact, and passes the leak detector. A style whose output cannot meet that
floor is dropped with a recorded reason rather than published noisy -- the
style set shrinks; grounding is never weakened to buy diversity.

**Diversity + dedup.** Accepted rows carry lexical (token-multiset) and
structural (style-template identity) diversity metrics per frame/style/
provider. :func:`dedupe_paraphrase_rows` drops exact duplicates,
near-duplicates (normalized-token Jaccard >= 0.9), and repeated
style-template outputs, within one frame's set and across the answer families
combined by :func:`generate_answer_family_paraphrases`. Rejected rows are
always counted with reasons -- nothing is dropped silently.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from slm_training.dsl.paraphrase_provenance import (
    PARAPHRASE_SYSTEM_PROMPT_V1,
    ParaphraseProvider,
    ParaphraseRequestV1,
    ParaphraseResponseV1,
    SamplingParamsV1,
    redact_text,
    validate_raw_response,
)
from slm_training.dsl.semantic_frame import SemanticFactV1, SemanticFrameV1
from slm_training.harness_core.lineage.records import canonical_json, content_sha

SCHEMA_VERSION = "frame_paraphrases/v1"

#: Plan-declared style set. If a style cannot meet the grounding floor for a
#: frame it is dropped with a recorded reason (stop rule) -- the plan is never
#: satisfied by weakening grounding or leaking target syntax.
STYLE_PLAN: tuple[str, ...] = ("concise", "descriptive", "imperative", "compositional")

#: Normalized-token Jaccard at or above which two same-style prompts are
#: near-duplicates.
NEAR_DUPLICATE_JACCARD = 0.9

_MARKER_RE = re.compile(r"(?<![\w])[:$][A-Za-z_][\w]*(?:\.[\w]+)*")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

_PROVENANCE_FIELDS: tuple[str, ...] = (
    "ast_path",
    "schema_field",
    "canonical_source",
    "schema_fingerprint",
    "frame_fingerprint",
    "prompt_digest",
    "response_digest",
    "cache_key",
    "category",
)

_OPERATOR_WORDS: Mapping[str, str] = {
    "||": "or",
    "&&": "and",
    "==": "equals",
    "!=": "does not equal",
    ">=": "is at least",
    "<=": "is at most",
    ">": "is greater than",
    "<": "is less than",
    "+": "plus",
    "-": "minus",
    "*": "times",
    "/": "divided by",
    "%": "modulo",
    "!": "not",
}


class FrameParaphraseError(ValueError):
    """A frame paraphrase set could not be constructed honestly."""


# --------------------------------------------------------------------------
# Fact-phrase rendering (frame facts only, never answer text)
# --------------------------------------------------------------------------


def _split_camel(name: str) -> str:
    return _CAMEL_BOUNDARY_RE.sub(" ", name).lower()


def _prop_of(fact: SemanticFactV1) -> str:
    for part in reversed(fact.ast_path):
        if isinstance(part, str) and part not in {"props", "entries", "args"}:
            return part.replace("_", " ")
    return "value"


def _value_words(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "unset"
    return str(value)


def render_fact_phrase(fact: SemanticFactV1) -> str:
    """Render one frame fact as a plain-language phrase.

    Only declared fact content is rendered: predicate, role/prop name, and
    value. Marker surfaces (``:ident`` placeholder values) are deliberately
    described by role, never reproduced, and component type names are split
    into lowercase words so the canonical production form never appears.
    """
    predicate = fact.predicate
    value = fact.value
    if predicate == "component_type":
        return f"a {_split_camel(str(value))} component"
    if predicate == "content_placeholder_id":
        return f"placeholder text for the {_prop_of(fact)} field"
    if predicate == "content_literal_value":
        return (
            f'the text "{value}"'
            if isinstance(value, str)
            else (f"the {_prop_of(fact)} text is {_value_words(value)}")
        )
    if predicate in {"prop_value", "closed_value", "list_item_value"}:
        return f"the {_prop_of(fact)} is {_value_words(value)}"
    if predicate == "cardinality":
        noun = "item" if value == 1 else "items"
        return f"exactly {value} {noun} in the {_prop_of(fact)} slot"
    if predicate == "state_ref_name":
        return f'reads the state "{value}"'
    if predicate == "runtime_ref_name":
        return f'uses the runtime reference "{value}"'
    if predicate == "unresolved_reference":
        return f'references "{value}"'
    if predicate == "inline_effect_kind":
        return f"an inline {str(value).lower()} effect"
    if predicate == "object_key":
        return f'a field named "{value}"'
    if predicate == "object_value":
        return f"the field value {_value_words(value)}"
    if predicate.endswith("_op") and str(value) in _OPERATOR_WORDS:
        return f"the operator {_OPERATOR_WORDS[str(value)]}"
    return f"{predicate.replace('_', ' ')} {_value_words(value)}"


def _render_forbidden_phrase(fact: SemanticFactV1) -> str:
    """Render the assertion a forbidden fact excludes (must stay absent)."""
    if fact.schema_field.startswith("grammar.") and fact.schema_field.endswith(
        "_operators"
    ):
        word = _OPERATOR_WORDS.get(str(fact.value), str(fact.value))
        return f"the operator {word}"
    return f"the {_prop_of(fact)} is {_value_words(fact.value)}"


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


# --------------------------------------------------------------------------
# Declared leak detector
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameLeakDetectorV1:
    """Declared leak detector for prompts generated from one frame.

    Rejects prompts containing target DSL production forms (``Card(`` etc.
    and their exact-case bare tokens), raw marker surfaces (``:ident`` /
    ``$ident`` codec forms and the answer's exact placeholder values), answer
    digests, and provenance field names.
    """

    component_names: tuple[str, ...]
    marker_surfaces: tuple[str, ...]
    answer_digests: tuple[str, ...]
    provenance_fields: tuple[str, ...] = _PROVENANCE_FIELDS

    @classmethod
    def from_frame(cls, frame: SemanticFrameV1) -> "FrameLeakDetectorV1":
        component_names = sorted(
            {n.type_name for n in frame.nodes if n.kind == "element"}
        )
        markers = sorted(
            {
                str(f.value)
                for f in frame.facts
                if f.predicate == "content_placeholder_id" and isinstance(f.value, str)
            }
        )
        answer_sha = hashlib.sha256(frame.canonical_source.encode("utf-8")).hexdigest()
        frame_sha = content_sha(
            {
                "dsl": frame.dsl,
                "schema_fingerprint": frame.schema_fingerprint,
                "nodes": [asdict(n) for n in frame.nodes],
                "roles": [asdict(r) for r in frame.roles],
                "relations": [asdict(r) for r in frame.relations],
                "effects": [asdict(e) for e in frame.effects],
                "facts": [asdict(f) for f in frame.facts],
            }
        )
        return cls(
            component_names=tuple(component_names),
            marker_surfaces=tuple(markers),
            answer_digests=(answer_sha, frame_sha),
        )

    def check(self, text: str) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        for name in self.component_names:
            if f"{name}(" in text:
                reasons.append(f"dsl_syntax:{name}(")
            elif re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", text):
                reasons.append(f"dsl_component_token:{name}")
        for marker in _MARKER_RE.findall(text):
            reasons.append(f"marker_surface:{marker}")
        for marker in self.marker_surfaces:
            if marker in text and f"marker_surface:{marker}" not in reasons:
                reasons.append(f"marker_surface:{marker}")
        for digest in self.answer_digests:
            if digest in text:
                reasons.append("answer_digest")
        for field_name in self.provenance_fields:
            if field_name in text:
                reasons.append(f"provenance_field:{field_name}")
        return (not reasons, tuple(reasons))


# --------------------------------------------------------------------------
# Frame -> sendable style payload (facts only)
# --------------------------------------------------------------------------


def _structure_payload(frame: SemanticFrameV1) -> dict[str, Any]:
    """Project the frame to a nested fact-phrase tree (no answer text)."""
    node_paths = {n.ast_path: n.node_id for n in frame.nodes}
    phrases_by_node: dict[tuple[str | int, ...], list[str]] = {
        path: [] for path in node_paths
    }
    for fact in frame.facts:
        if fact.category != "required":
            continue
        best: tuple[str | int, ...] | None = None
        for path in node_paths:
            if fact.ast_path[: len(path)] == path and (
                best is None or len(path) > len(best)
            ):
                best = path
        key = best if best is not None else ()
        phrases_by_node.setdefault(key, []).append(render_fact_phrase(fact))

    children: dict[str, list[tuple[int, str]]] = {}
    for role in frame.roles:
        children.setdefault(role.parent_id, []).append((role.order or 0, role.child_id))

    id_to_path = {n.node_id: n.ast_path for n in frame.nodes}

    def build(path: tuple[str | int, ...]) -> dict[str, Any]:
        node_id = node_paths.get(path)
        kids = []
        if node_id is not None:
            for _, child_id in sorted(children.get(node_id, [])):
                child_path = id_to_path.get(child_id)
                if child_path in node_paths:
                    kids.append(build(child_path))
        return {"phrases": phrases_by_node.get(path, []), "children": kids}

    root_path = min(node_paths, key=len) if node_paths else ()
    tree = build(root_path)
    unattached = phrases_by_node.get((), []) if root_path != () else []
    return {"root": tree, "unattached": unattached}


# --------------------------------------------------------------------------
# Offline deterministic fixture provider (no network, no credentials)
# --------------------------------------------------------------------------


class FrameParaphraseFixtureProviderV1:
    """Fully offline, credential-free style renderer over fact phrases.

    Implements the DSH2-03 :class:`ParaphraseProvider` protocol. ``generate``
    reads the JSON style payload (style + fact-phrase tree, built from frame
    facts only) and renders one of four deterministic style templates. Same
    inputs always yield the same text; no network, no external state.
    """

    provider_id = "slm365-frame-fixture"
    model_id = "frame-style-renderer"
    model_revision = "v1"

    def secrets(self) -> tuple[str, ...]:
        return ()

    def generate(
        self,
        *,
        system_text: str,
        prompt_text: str,
        sampling: SamplingParamsV1,
        seed: int | None,
    ) -> str:
        payload = json.loads(prompt_text)
        style = payload["style"]
        structure = payload["structure"]
        root = structure["root"]
        phrases = self._flatten(root) + list(structure.get("unattached", []))
        if style == "concise":
            return "Layout: " + "; ".join(phrases) + "."
        if style == "descriptive":
            first, rest = phrases[0], phrases[1:]
            text = f"This layout has {first}"
            text += "".join(f"; it also includes {p}" for p in rest)
            return text + "."
        if style == "imperative":
            return "Build this layout: make sure " + "; make sure ".join(phrases) + "."
        if style == "compositional":
            return (
                self._compositional(root, structure)
                or "Layout: " + "; ".join(phrases) + "."
            )
        raise FrameParaphraseError(f"undeclared style {style!r} in style payload")

    def _flatten(self, node: dict[str, Any]) -> list[str]:
        phrases = list(node.get("phrases") or [])
        for child in node.get("children") or []:
            phrases.extend(self._flatten(child))
        return phrases

    def _compositional(self, root: dict[str, Any], structure: dict[str, Any]) -> str:
        parts: list[str] = []

        def walk(node: dict[str, Any], depth: int) -> None:
            phrases = node.get("phrases") or []
            if phrases:
                lead = (
                    "Start with"
                    if not parts
                    else ("Inside it," if depth == 1 else "Within that,")
                )
                parts.append(f"{lead} " + "; ".join(phrases) + ".")
            for child in node.get("children") or []:
                walk(child, depth + 1)

        walk(root, 0)
        for phrase in structure.get("unattached", []):
            parts.append(f"Also, {phrase}.")
        return " ".join(parts)


# --------------------------------------------------------------------------
# Typed rows
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameParaphraseRowV1:
    """One accepted prompt row, linked to its canonical answer by fingerprint."""

    style: str
    prompt_text: str
    prompt_sha256: str
    required_facts_covered: tuple[str, ...]
    forbidden_facts_absent: bool
    leak_check_pass: bool
    style_provider_provenance: Mapping[str, Any]
    template_signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "style": self.style,
            "prompt_text": self.prompt_text,
            "prompt_sha256": self.prompt_sha256,
            "required_facts_covered": list(self.required_facts_covered),
            "forbidden_facts_absent": self.forbidden_facts_absent,
            "leak_check_pass": self.leak_check_pass,
            "style_provider_provenance": dict(self.style_provider_provenance),
            "template_signature": self.template_signature,
        }


@dataclass(frozen=True)
class RejectedParaphraseRowV1:
    """A rejected prompt candidate, always counted with a reason."""

    style: str
    reason: str
    detail: str
    prompt_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "style": self.style,
            "reason": self.reason,
            "detail": self.detail,
            "prompt_sha256": self.prompt_sha256,
        }


@dataclass(frozen=True)
class StyleDiversityV1:
    style: str
    token_count: int
    distinct_token_count: int
    distinct_token_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameDiversityV1:
    per_style: tuple[StyleDiversityV1, ...]
    pairwise_token_jaccard: Mapping[str, float]
    lexical_diversity: float
    structural_diversity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_style": [s.to_dict() for s in self.per_style],
            "pairwise_token_jaccard": dict(self.pairwise_token_jaccard),
            "lexical_diversity": self.lexical_diversity,
            "structural_diversity": self.structural_diversity,
        }


@dataclass(frozen=True)
class FrameParaphraseSetV1:
    """The full accepted/rejected paraphrase set for one frame."""

    schema_version: str
    frame_fingerprint: str
    canonical_answer_fingerprint: str
    equivalence_fingerprints: tuple[str, ...]
    style_plan: tuple[str, ...]
    rows: tuple[FrameParaphraseRowV1, ...]
    rejected_rows: tuple[RejectedParaphraseRowV1, ...]
    rejection_counts: Mapping[str, int]
    per_style_counts: Mapping[str, Mapping[str, int]]
    diversity: FrameDiversityV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frame_fingerprint": self.frame_fingerprint,
            "canonical_answer_fingerprint": self.canonical_answer_fingerprint,
            "equivalence_fingerprints": list(self.equivalence_fingerprints),
            "style_plan": list(self.style_plan),
            "rows": [r.to_dict() for r in self.rows],
            "rejected_rows": [r.to_dict() for r in self.rejected_rows],
            "rejection_counts": dict(self.rejection_counts),
            "per_style_counts": {k: dict(v) for k, v in self.per_style_counts.items()},
            "diversity": self.diversity.to_dict(),
        }


@dataclass(frozen=True)
class AnswerFamilyParaphraseSetV1:
    """Cross-frame set: all accepted prompts link one canonical answer family."""

    schema_version: str
    canonical_answer_fingerprint: str
    equivalence_fingerprints: tuple[str, ...]
    frame_fingerprints: tuple[str, ...]
    rows: tuple[FrameParaphraseRowV1, ...]
    rejected_rows: tuple[RejectedParaphraseRowV1, ...]
    rejection_counts: Mapping[str, int]
    per_frame_sets: tuple[FrameParaphraseSetV1, ...]
    diversity: FrameDiversityV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "canonical_answer_fingerprint": self.canonical_answer_fingerprint,
            "equivalence_fingerprints": list(self.equivalence_fingerprints),
            "frame_fingerprints": list(self.frame_fingerprints),
            "rows": [r.to_dict() for r in self.rows],
            "rejected_rows": [r.to_dict() for r in self.rejected_rows],
            "rejection_counts": dict(self.rejection_counts),
            "per_frame_sets": [s.to_dict() for s in self.per_frame_sets],
            "diversity": self.diversity.to_dict(),
        }


# --------------------------------------------------------------------------
# Diversity
# --------------------------------------------------------------------------


def compute_diversity(rows: Sequence[FrameParaphraseRowV1]) -> FrameDiversityV1:
    """Lexical (token-multiset) + structural (template identity) diversity."""
    per_style: list[StyleDiversityV1] = []
    for row in rows:
        tokens = _TOKEN_RE.findall(row.prompt_text.lower())
        distinct = set(tokens)
        per_style.append(
            StyleDiversityV1(
                style=row.style,
                token_count=len(tokens),
                distinct_token_count=len(distinct),
                distinct_token_ratio=(len(distinct) / len(tokens)) if tokens else 0.0,
            )
        )
    pairwise: dict[str, float] = {}
    jaccards: list[float] = []
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            a, b = _tokens(left.prompt_text), _tokens(right.prompt_text)
            union = a | b
            j = (len(a & b) / len(union)) if union else 0.0
            pairwise[f"{left.style}|{right.style}"] = round(j, 6)
            jaccards.append(j)
    lexical = 1.0 - (sum(jaccards) / len(jaccards)) if jaccards else 0.0
    signatures = {r.template_signature for r in rows}
    structural = (len(signatures) / len(rows)) if rows else 0.0
    return FrameDiversityV1(
        per_style=tuple(per_style),
        pairwise_token_jaccard=pairwise,
        lexical_diversity=round(lexical, 6),
        structural_diversity=round(structural, 6),
    )


# --------------------------------------------------------------------------
# Dedup (exact / near-duplicate / repeated style template)
# --------------------------------------------------------------------------


def dedupe_paraphrase_rows(
    rows: Iterable[FrameParaphraseRowV1],
    *,
    near_jaccard: float = NEAR_DUPLICATE_JACCARD,
) -> tuple[tuple[FrameParaphraseRowV1, ...], tuple[RejectedParaphraseRowV1, ...]]:
    """Drop exact, near-duplicate, and repeated-style-template rows.

    Exact duplicates are detected across all rows; near-duplicates (token-set
    Jaccard >= ``near_jaccard``) and repeated style-template outputs are
    detected within the same style, so genuinely different styles of one
    frame are never deduplicated against each other. Works within one frame's
    rows and across combined answer-family rows. Nothing is dropped silently:
    every dropped row is returned with its reason.
    """
    kept: list[FrameParaphraseRowV1] = []
    rejected: list[RejectedParaphraseRowV1] = []
    for row in rows:
        reason: str | None = None
        normalized = _normalize(row.prompt_text)
        for other in kept:
            if normalized == _normalize(other.prompt_text):
                reason = "exact_duplicate"
                break
            if row.style != other.style:
                continue
            if row.template_signature == other.template_signature:
                reason = "repeated_style_template"
                break
            a, b = _tokens(row.prompt_text), _tokens(other.prompt_text)
            union = a | b
            if union and len(a & b) / len(union) >= near_jaccard:
                reason = "near_duplicate"
                break
        if reason is None:
            kept.append(row)
        else:
            rejected.append(
                RejectedParaphraseRowV1(
                    style=row.style,
                    reason=reason,
                    detail=f"duplicates an already-accepted {row.style!r} row",
                    prompt_sha256=row.prompt_sha256,
                )
            )
    return tuple(kept), tuple(rejected)


# --------------------------------------------------------------------------
# Grounding floor (stop rule)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    style: str
    text: str
    provenance: Mapping[str, Any]
    coverage_keys: tuple[str, ...]
    forbidden_phrases: tuple[str, ...]
    leak_detector: FrameLeakDetectorV1


def enforce_grounding_floor(
    candidates: Iterable[_Candidate],
) -> tuple[tuple[FrameParaphraseRowV1, ...], tuple[RejectedParaphraseRowV1, ...]]:
    """Accept only candidates covering all required facts with no leak.

    Stop rule: a style that cannot cover every required fact, avoid every
    forbidden fact, and pass the declared leak detector is dropped with a
    recorded reason -- never published as a noisy row, and grounding is never
    weakened to keep the style.
    """
    accepted: list[FrameParaphraseRowV1] = []
    rejected: list[RejectedParaphraseRowV1] = []
    for cand in candidates:
        text_norm = _normalize(cand.text)
        sha = hashlib.sha256(cand.text.encode("utf-8")).hexdigest()
        covered = tuple(k for k in cand.coverage_keys if _normalize(k) in text_norm)
        missing = tuple(k for k in cand.coverage_keys if _normalize(k) not in text_norm)
        forbidden_hit = tuple(
            p for p in cand.forbidden_phrases if _normalize(p) in text_norm
        )
        leak_ok, leak_reasons = cand.leak_detector.check(cand.text)
        if missing:
            rejected.append(
                RejectedParaphraseRowV1(
                    style=cand.style,
                    reason="missing_required_facts",
                    detail=f"required fact phrases not covered: {list(missing)!r}",
                    prompt_sha256=sha,
                )
            )
            continue
        if forbidden_hit:
            rejected.append(
                RejectedParaphraseRowV1(
                    style=cand.style,
                    reason="forbidden_fact_present",
                    detail=f"forbidden assertions present: {list(forbidden_hit)!r}",
                    prompt_sha256=sha,
                )
            )
            continue
        if not leak_ok:
            rejected.append(
                RejectedParaphraseRowV1(
                    style=cand.style,
                    reason="leak_detected",
                    detail=f"declared detector rejected: {list(leak_reasons)!r}",
                    prompt_sha256=sha,
                )
            )
            continue
        signature = content_sha(
            {"style": cand.style, "coverage": sorted(cand.coverage_keys)}
        )
        accepted.append(
            FrameParaphraseRowV1(
                style=cand.style,
                prompt_text=cand.text,
                prompt_sha256=sha,
                required_facts_covered=covered,
                forbidden_facts_absent=True,
                leak_check_pass=True,
                style_provider_provenance=cand.provenance,
                template_signature=signature,
            )
        )
    return tuple(accepted), tuple(rejected)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _canonical_answer_fingerprint(frame: SemanticFrameV1) -> str:
    return hashlib.sha256(frame.canonical_source.encode("utf-8")).hexdigest()


def _style_system_text(style: str) -> str:
    return (
        f"{PARAPHRASE_SYSTEM_PROMPT_V1} Render every declared fact phrase "
        f"verbatim in the '{style}' style; add connective prose only."
    )


def _request_style(
    provider: ParaphraseProvider,
    frame: SemanticFrameV1,
    *,
    style: str,
    structure: Mapping[str, Any],
    sampling: SamplingParamsV1,
    seed: int | None,
    now: Callable[[], str],
) -> tuple[str, Mapping[str, Any]]:
    """Run one style through the DSH2-03 contract; return (text, provenance)."""
    payload = {"style": style, "structure": structure}
    system_text = _style_system_text(style)
    prompt_text = canonical_json(payload)
    prompt_digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    system_digest = hashlib.sha256(system_text.encode("utf-8")).hexdigest()
    frame_fingerprint = content_sha(
        {
            "dsl": frame.dsl,
            "schema_fingerprint": frame.schema_fingerprint,
            "nodes": [asdict(n) for n in frame.nodes],
            "roles": [asdict(r) for r in frame.roles],
            "relations": [asdict(r) for r in frame.relations],
            "effects": [asdict(e) for e in frame.effects],
            "facts": [asdict(f) for f in frame.facts],
        }
    )
    request = ParaphraseRequestV1(
        request_id=content_sha({"style": style, "prompt_digest": prompt_digest})[:32],
        provider_id=provider.provider_id,
        model_id=provider.model_id,
        model_revision=provider.model_revision,
        schema_fingerprint=frame.schema_fingerprint,
        frame_fingerprint=frame_fingerprint,
        system_digest=system_digest,
        prompt_digest=prompt_digest,
        sampling=sampling,
        seed=seed,
        created_at=now(),
    )
    raw = provider.generate(
        system_text=system_text, prompt_text=prompt_text, sampling=sampling, seed=seed
    )
    secrets = tuple(provider.secrets())
    redacted = redact_text(raw, secrets)
    response = ParaphraseResponseV1(
        response_digest=hashlib.sha256(redacted.encode("utf-8")).hexdigest(),
        created_at=now(),
    )
    provenance = {
        "request": request.to_dict(),
        "response": response.to_dict(),
        "provider_id": provider.provider_id,
        "model_id": provider.model_id,
        "model_revision": provider.model_revision,
    }
    return redacted, provenance


def generate_frame_paraphrases(
    frame: SemanticFrameV1,
    *,
    provider: ParaphraseProvider,
    style_plan: Sequence[str] = STYLE_PLAN,
    leak_detector: FrameLeakDetectorV1 | None = None,
    equivalence_fingerprints: Sequence[str] = (),
    sampling: SamplingParamsV1 | None = None,
    seed: int | None = None,
    now: Callable[[], str] | None = None,
) -> FrameParaphraseSetV1:
    """Generate the plan-declared style paraphrase set for one frame.

    Every style is requested through the DSH2-03 provider contract with full
    per-style provenance, validated for well-formedness, then passed through
    :func:`enforce_grounding_floor` (required coverage, forbidden absence,
    declared leak detector) and within-set dedup. Accepted rows link to the
    canonical answer and equivalence set by fingerprint only.
    """
    clock = now or (lambda: datetime.now(timezone.utc).isoformat())
    sampling = sampling or SamplingParamsV1()
    detector = leak_detector or FrameLeakDetectorV1.from_frame(frame)
    structure = _structure_payload(frame)
    required_facts = frame.facts_by_category("required")
    coverage_keys = tuple(render_fact_phrase(f) for f in required_facts)
    forbidden_phrases = tuple(
        _render_forbidden_phrase(f) for f in frame.facts_by_category("forbidden")
    )

    candidates: list[_Candidate] = []
    rejected: list[RejectedParaphraseRowV1] = []
    for style in style_plan:
        text, provenance = _request_style(
            provider,
            frame,
            style=style,
            structure=structure,
            sampling=sampling,
            seed=seed,
            now=clock,
        )
        ok, reason = validate_raw_response(
            text,
            prompt_text=canonical_json({"style": style, "structure": structure}),
            secrets=tuple(provider.secrets()),
        )
        if not ok:
            rejected.append(
                RejectedParaphraseRowV1(
                    style=style,
                    reason="provider_response_invalid",
                    detail=str(reason),
                )
            )
            continue
        candidates.append(
            _Candidate(
                style=style,
                text=text,
                provenance=provenance,
                coverage_keys=coverage_keys,
                forbidden_phrases=forbidden_phrases,
                leak_detector=detector,
            )
        )

    accepted, floor_rejected = enforce_grounding_floor(candidates)
    rejected.extend(floor_rejected)
    kept, dedup_rejected = dedupe_paraphrase_rows(accepted)
    rejected.extend(dedup_rejected)

    rejection_counts: dict[str, int] = {}
    for row in rejected:
        rejection_counts[row.reason] = rejection_counts.get(row.reason, 0) + 1
    per_style: dict[str, dict[str, int]] = {}
    for style in style_plan:
        per_style[style] = {
            "accepted": sum(1 for r in kept if r.style == style),
            "rejected": sum(1 for r in rejected if r.style == style),
        }

    frame_fingerprint = content_sha(
        {
            "dsl": frame.dsl,
            "schema_fingerprint": frame.schema_fingerprint,
            "nodes": [asdict(n) for n in frame.nodes],
            "roles": [asdict(r) for r in frame.roles],
            "relations": [asdict(r) for r in frame.relations],
            "effects": [asdict(e) for e in frame.effects],
            "facts": [asdict(f) for f in frame.facts],
        }
    )
    return FrameParaphraseSetV1(
        schema_version=SCHEMA_VERSION,
        frame_fingerprint=frame_fingerprint,
        canonical_answer_fingerprint=_canonical_answer_fingerprint(frame),
        equivalence_fingerprints=tuple(equivalence_fingerprints),
        style_plan=tuple(style_plan),
        rows=kept,
        rejected_rows=tuple(rejected),
        rejection_counts=rejection_counts,
        per_style_counts=per_style,
        diversity=compute_diversity(kept),
    )


def generate_answer_family_paraphrases(
    frames: Sequence[SemanticFrameV1],
    *,
    provider: ParaphraseProvider,
    style_plan: Sequence[str] = STYLE_PLAN,
    sampling: SamplingParamsV1 | None = None,
    seed: int | None = None,
    now: Callable[[], str] | None = None,
) -> AnswerFamilyParaphraseSetV1:
    """Generate + dedup across an answer family of equivalent frames.

    Every frame is generated independently, then all rows are deduplicated
    together (exact / near / repeated-template across the family). All kept
    rows link to the same canonical answer fingerprint and the family's
    accepted equivalence set.
    """
    if not frames:
        raise FrameParaphraseError("answer family must contain at least one frame")
    canonical = _canonical_answer_fingerprint(frames[0])
    equivalence = tuple(_canonical_answer_fingerprint(f) for f in frames)
    sets = tuple(
        generate_frame_paraphrases(
            frame,
            provider=provider,
            style_plan=style_plan,
            equivalence_fingerprints=equivalence,
            sampling=sampling,
            seed=seed,
            now=now,
        )
        for frame in frames
    )
    all_rows = [row for s in sets for row in s.rows]
    kept, dedup_rejected = dedupe_paraphrase_rows(all_rows)
    rejected = [r for s in sets for r in s.rejected_rows] + list(dedup_rejected)
    rejection_counts: dict[str, int] = {}
    for row in rejected:
        rejection_counts[row.reason] = rejection_counts.get(row.reason, 0) + 1
    return AnswerFamilyParaphraseSetV1(
        schema_version=SCHEMA_VERSION,
        canonical_answer_fingerprint=canonical,
        equivalence_fingerprints=equivalence,
        frame_fingerprints=tuple(s.frame_fingerprint for s in sets),
        rows=kept,
        rejected_rows=tuple(rejected),
        rejection_counts=rejection_counts,
        per_frame_sets=sets,
        diversity=compute_diversity(kept),
    )


__all__ = [
    "NEAR_DUPLICATE_JACCARD",
    "SCHEMA_VERSION",
    "STYLE_PLAN",
    "AnswerFamilyParaphraseSetV1",
    "FrameDiversityV1",
    "FrameLeakDetectorV1",
    "FrameParaphraseError",
    "FrameParaphraseFixtureProviderV1",
    "FrameParaphraseRowV1",
    "FrameParaphraseSetV1",
    "RejectedParaphraseRowV1",
    "StyleDiversityV1",
    "compute_diversity",
    "dedupe_paraphrase_rows",
    "enforce_grounding_floor",
    "generate_answer_family_paraphrases",
    "generate_frame_paraphrases",
    "render_fact_phrase",
]
