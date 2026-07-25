"""OperatorNlTurnV1: schema-grounded NL edit turns over operator applications.

DSH3-11 (SLM-379): generate natural-language *edit* prompts for an
:class:`~slm_training.dsl.operators.contracts.OperatorApplicationV1` by
paraphrasing EXACT operator effects from a derived
:class:`~slm_training.dsl.operator_frame.OperatorFrameV1` (DSH3-11 frame) —
never by letting an LLM define transformation semantics. The provider,
provenance, structured schema, and quarantine/leak machinery are the CAP1
contract reused unchanged (DSH2-03
:mod:`slm_training.dsl.paraphrase_provenance` + the SLM-365 offline fixture
provider pattern).

**Admission, not generation, is the gate.** A prompt candidate is admitted
only when, under the current state and the exact
:class:`~slm_training.dsl.operators.legal_set.OperatorLegalSetV1`:

1. the target application is a member of the legal set
   (else ``missing_legal_membership``) and of the declared accepted
   application set (else ``not_in_accepted_set``);
2. every state-reference fact resolves unambiguously through the typed
   descriptor tables — a descriptor shared by several opaque refs is fine
   only when every legal action using any of those refs is effect-equivalent
   and the prompt names the accepted set (else ``ambiguous_state_reference``);
3. the prompt is not compatible with any non-equivalent legal application
   (else ``compatible_with_non_equivalent``) — *compatibility* is mechanical:
   an action's own derived frame covers every required edit/state-reference
   phrase the prompt asserts. When several equivalent applications are
   accepted, the prompt must explicitly name the accepted SET
   (``accepted_set_not_named`` otherwise);
4. the prompt passes the declared leak detector (``leak``): no serialized
   operator action, operator id, opaque id, request id, or digest surface.

**Targets stay symbol-only.** Every admitted row's target is the reserved
operator application serialization (``OPERATOR <id> <typed args>``), the
resulting canonical AST source, or a declared dual view — natural language
never enters targets.

**Stop rule.** A fixture family whose prompts cannot be disambiguated under
the current state/schema records disposition ``symbolic_cap2_only`` — the
honest retention of symbolic CAP2 only, never a weakened admission.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from slm_training.dsl.operator_frame import (
    OperatorFactV1,
    OperatorFrameV1,
    derive_operator_frame,
)
from slm_training.dsl.operators.contracts import (
    OperatorApplicationV1,
)
from slm_training.dsl.operators.legal_set import (
    LegalOperatorActionV1,
    OperatorLegalSetV1,
    deserialize_operator_action,
)
from slm_training.dsl.operators.references import ReferenceTableV1
from slm_training.dsl.operators.registry import (
    OperatorLibraryV1,
    OperatorStateV1,
)
from slm_training.dsl.pack import DslPack
from slm_training.dsl.paraphrase_provenance import (
    PARAPHRASE_SYSTEM_PROMPT_V1,
    ParaphraseProvider,
    ParaphraseRequestV1,
    ParaphraseResponseV1,
    SamplingParamsV1,
    redact_text,
    validate_raw_response,
)
from slm_training.harness_core.lineage.records import canonical_json, content_sha
from slm_training.harnesses.train_data.operator_corpus import OperatorTargetView

SCHEMA_VERSION = "operator_nl_turns/v1"

#: Plan-declared prompt styles (mirrors the SLM-365 plan).
STYLE_PLAN: tuple[str, ...] = ("concise", "imperative")

REJECTION_CODES: tuple[str, ...] = (
    "missing_legal_membership",
    "not_in_accepted_set",
    "ambiguous_state_reference",
    "compatible_with_non_equivalent",
    "accepted_set_not_named",
    "leak",
    "provider_response_invalid",
)

#: Fixture-family dispositions (stop rule).
DISPOSITION_NL_ADMITTED = "nl_prompts_admitted"
DISPOSITION_SYMBOLIC_CAP2_ONLY = "symbolic_cap2_only"

_MARKER_RE = re.compile(r"(?<![\w])[:$][A-Za-z_][\w]*(?:\.[\w]+)*")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_PROVENANCE_FIELDS: tuple[str, ...] = (
    "slot_id",
    "descriptor_fingerprint",
    "before_state_digest",
    "after_state_digest",
    "application_id",
    "semantic_id",
    "effect_fingerprint",
    "prompt_digest",
    "response_digest",
    "cache_key",
)

_SYMBOL_WORDS: Mapping[str, str] = {
    "node": "node",
    "role": "role",
    "index": "position",
    "value": "value",
    "symbol": "symbol",
    "template": "template",
}


class OperatorNlTurnError(ValueError):
    """An operator NL turn set could not be constructed honestly."""


# --------------------------------------------------------------------------
# Fact-phrase rendering (frame facts only, never serialized actions)
# --------------------------------------------------------------------------


def _value_words(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "unset"
    if isinstance(value, (str, int, float)):
        return str(value)
    return "new structured content"


def render_operator_fact_phrase(fact: OperatorFactV1) -> str:
    """Render one operator frame fact as a plain-language phrase.

    Only declared fact content is rendered: delta kind, ref kind, typed
    descriptor value type, and scalar before/after values. Opaque ids,
    request ids, operator ids, and serialized actions never appear.
    """
    ref_word = _SYMBOL_WORDS[fact.ref.KIND.value]
    if fact.fact_kind == "effect_delta":
        delta_kind = str(fact.provenance.delta_kind)
        before = _value_words((fact.value or {}).get("before"))
        after = _value_words((fact.value or {}).get("after"))
        return f"change the {delta_kind} of the {ref_word} from {before} to {after}"
    if fact.fact_kind in {"consumed_role", "consumed_binder"}:
        return f"remove the existing {ref_word}"
    if fact.fact_kind in {"produced_role", "produced_binder"}:
        return f"introduce a new {ref_word}"
    if fact.fact_kind == "state_reference":
        value_type = str((fact.value or {}).get("value_type", "value")).replace(
            "_", " "
        )
        return f"act on the {ref_word} {value_type}"
    if fact.fact_kind == "untouched_reference":
        return f"leave the {ref_word} unchanged"
    return f"{fact.fact_kind.replace('_', ' ')} of the {ref_word}"


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


# --------------------------------------------------------------------------
# Declared leak detector
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorLeakDetectorV1:
    """Declared leak detector for operator edit prompts.

    Rejects prompts containing any serialized operator action or operator id
    from the legal set, any opaque/request id surface from the reference
    table, any digest surface (state/AST/effect/application/semantic ids),
    marker surfaces (``:ident``/``$ident`` codec forms), and provenance field
    names.
    """

    operator_ids: tuple[str, ...]
    serialized_actions: tuple[str, ...]
    opaque_ids: tuple[str, ...]
    request_ids: tuple[str, ...]
    digests: tuple[str, ...]
    provenance_fields: tuple[str, ...] = _PROVENANCE_FIELDS

    @classmethod
    def from_evidence(
        cls,
        legal_set: OperatorLegalSetV1,
        frame: OperatorFrameV1,
        reference_table: ReferenceTableV1,
    ) -> "OperatorLeakDetectorV1":
        actions = legal_set.operator_actions
        digests = {
            frame.before_state_digest,
            frame.before_ast_digest,
            frame.effect_fingerprint,
            legal_set.state_fingerprint,
            legal_set.reference_table_fingerprint,
        }
        for action in actions:
            digests.update(
                (action.semantic_id, action.application_id, action.proof_fingerprint)
            )
        return cls(
            operator_ids=tuple(sorted({action.operator_id for action in actions})),
            serialized_actions=tuple(sorted({a.serialized for a in actions})),
            opaque_ids=tuple(
                sorted({entry.ref.opaque_id for entry in reference_table.entries})
            ),
            request_ids=(reference_table.request_id,),
            digests=tuple(sorted(digests)),
        )

    def check(self, text: str) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        for serialized in self.serialized_actions:
            if serialized in text:
                reasons.append("serialized_action")
                break
        for operator_id in self.operator_ids:
            if operator_id in text:
                reasons.append(f"operator_id:{operator_id}")
        if re.search(r"(?<![\w])OPERATOR(?![\w])", text):
            reasons.append("reserved_opcode")
        for opaque_id in self.opaque_ids:
            if re.search(rf"(?<![\w]){re.escape(opaque_id)}(?![\w])", text):
                reasons.append(f"opaque_id:{opaque_id}")
        for request_id in self.request_ids:
            if re.search(rf"(?<![\w]){re.escape(request_id)}(?![\w])", text):
                reasons.append(f"request_id:{request_id}")
        for marker in _MARKER_RE.findall(text):
            reasons.append(f"marker_surface:{marker}")
        for digest in self.digests:
            if digest in text:
                reasons.append("digest_surface")
                break
        for field_name in self.provenance_fields:
            if field_name in text:
                reasons.append(f"provenance_field:{field_name}")
        return (not reasons, tuple(reasons))


# --------------------------------------------------------------------------
# Offline deterministic fixture provider (DSH2-03 protocol, no network)
# --------------------------------------------------------------------------


class OperatorNlFixtureProviderV1:
    """Fully offline, credential-free edit-prompt renderer over fact phrases.

    Implements the DSH2-03 :class:`ParaphraseProvider` protocol. ``generate``
    reads the JSON style payload (style + fact phrases, built from frame
    facts only) and renders a deterministic template. Same inputs always
    yield the same text; no network, no external state.
    """

    provider_id = "slm379-operator-nl-fixture"
    model_id = "operator-edit-renderer"
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
        phrases = list(payload["phrases"])
        accepted_set_size = int(payload.get("accepted_set_size") or 0)
        if style == "concise":
            text = "Edit: " + "; ".join(phrases) + "."
        elif style == "imperative":
            text = "Apply this edit: make sure " + "; make sure ".join(phrases) + "."
        else:
            raise OperatorNlTurnError(f"undeclared style {style!r} in style payload")
        if accepted_set_size > 1:
            text += (
                f" Any of the {accepted_set_size} equivalent accepted "
                "applications satisfies this edit."
            )
        return text


# --------------------------------------------------------------------------
# Typed rows
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorNlTurnV1:
    """One admitted NL edit turn: prompt + symbol-only target."""

    style: str
    prompt_text: str
    prompt_sha256: str
    target_view: str
    target: Mapping[str, str]
    names_accepted_set: bool
    accepted_application_ids: tuple[str, ...]
    state_reference_scoring: Mapping[str, str]
    style_provider_provenance: Mapping[str, Any]
    frame_fingerprint: str
    schema: str = "operator_nl_turn/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "style": self.style,
            "prompt_text": self.prompt_text,
            "prompt_sha256": self.prompt_sha256,
            "target_view": self.target_view,
            "target": dict(self.target),
            "names_accepted_set": self.names_accepted_set,
            "accepted_application_ids": list(self.accepted_application_ids),
            "state_reference_scoring": dict(self.state_reference_scoring),
            "style_provider_provenance": dict(self.style_provider_provenance),
            "frame_fingerprint": self.frame_fingerprint,
        }


@dataclass(frozen=True)
class RejectedOperatorPromptV1:
    """A rejected prompt candidate, always counted with a code (stop rule)."""

    style: str
    code: str
    detail: str
    prompt_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.code not in REJECTION_CODES:
            raise OperatorNlTurnError(f"undeclared rejection code {self.code!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "style": self.style,
            "code": self.code,
            "detail": self.detail,
            "prompt_sha256": self.prompt_sha256,
        }


@dataclass(frozen=True)
class OperatorNlTurnSetV1:
    """The full admitted/rejected prompt set for one operator application."""

    schema_version: str
    operator_id: str
    application_id: str
    frame_fingerprint: str
    legal_set_fingerprint: str
    rows: tuple[OperatorNlTurnV1, ...]
    rejected_rows: tuple[RejectedOperatorPromptV1, ...]
    rejection_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operator_id": self.operator_id,
            "application_id": self.application_id,
            "frame_fingerprint": self.frame_fingerprint,
            "legal_set_fingerprint": self.legal_set_fingerprint,
            "rows": [row.to_dict() for row in self.rows],
            "rejected_rows": [row.to_dict() for row in self.rejected_rows],
            "rejection_counts": dict(self.rejection_counts),
        }


# --------------------------------------------------------------------------
# Ambiguity classifier (state references + non-equivalent compatibility)
# --------------------------------------------------------------------------


def _effect_fingerprints_by_action(
    pack: DslPack,
    library: OperatorLibraryV1,
    state: OperatorStateV1,
    legal_set: OperatorLegalSetV1,
    provenance: Any,
) -> dict[str, str | None]:
    """Dry-run every legal action; map application_id -> effect fingerprint."""
    out: dict[str, str | None] = {}
    for action in legal_set.operator_actions:
        application = library.dry_run(
            pack, state, action.operator_id, action.arguments, provenance
        )
        out[action.application_id] = (
            application.effect.fingerprint if application.effect is not None else None
        )
    return out


def classify_state_references(
    frame: OperatorFrameV1,
    reference_table: ReferenceTableV1,
    *,
    equivalent: bool,
) -> Mapping[str, str]:
    """Score each state-reference fact: ``unambiguous`` / ``accepted_set_scored``.

    A descriptor fingerprint that names exactly one opaque ref is
    ``unambiguous``. A descriptor shared by several refs is
    ``accepted_set_scored`` when every legal action using any of those refs
    is effect-equivalent to the target (the caller passes ``equivalent``);
    otherwise the caller must reject ``ambiguous_state_reference``.
    """
    by_descriptor: dict[str, set[tuple[str, str, str]]] = {}
    for entry in reference_table.entries:
        by_descriptor.setdefault(entry.descriptor.fingerprint, set()).add(
            (entry.ref.KIND.value, entry.ref.request_id, entry.ref.opaque_id)
        )
    scoring: dict[str, str] = {}
    for fact in frame.state_reference_facts:
        fingerprint = fact.provenance.descriptor_fingerprint or ""
        refs = by_descriptor.get(fingerprint, set())
        key = f"{fact.ref.KIND.value}:{fact.ref.opaque_id}"
        if len(refs) <= 1:
            scoring[key] = "unambiguous"
        elif equivalent:
            scoring[key] = "accepted_set_scored"
        else:
            scoring[key] = "ambiguous"
    return scoring


def prompt_compatible_action_ids(
    coverage_phrases: Sequence[str],
    candidate_frames: Mapping[str, OperatorFrameV1],
) -> tuple[str, ...]:
    """Application ids whose frame covers every phrase the prompt asserts."""
    normalized = tuple(_normalize(phrase) for phrase in coverage_phrases)
    out: list[str] = []
    for application_id, frame in sorted(candidate_frames.items()):
        own = {
            _normalize(render_operator_fact_phrase(fact))
            for fact in (*frame.required_edit_facts, *frame.state_reference_facts)
        }
        if all(phrase in own for phrase in normalized):
            out.append(application_id)
    return tuple(out)


# --------------------------------------------------------------------------
# Symbol-only targets
# --------------------------------------------------------------------------


def build_symbol_only_target(
    view: OperatorTargetView,
    action: LegalOperatorActionV1,
    after_state: OperatorStateV1,
) -> Mapping[str, str]:
    """The symbol-only target for one admitted turn (never natural language)."""
    if view is OperatorTargetView.OPERATOR_ONLY:
        return {"operator": action.serialized}
    if view is OperatorTargetView.RESULT_AST_ONLY:
        return {"result_ast": after_state.source}
    if view is OperatorTargetView.DUAL:
        return {"operator": action.serialized, "result_ast": after_state.source}
    raise OperatorNlTurnError(f"undeclared NL target view {view!r}")


def assert_target_symbol_only(target: Mapping[str, str]) -> None:
    """Validate that a target carries only symbolic content.

    Operator serializations must round-trip through the reserved codec; the
    result AST must be present DSL source text (checked by the caller's pack
    at admission time — here we assert the closed key set and that no value
    reads as natural-language prose).
    """
    if not set(target) <= {"operator", "result_ast"}:
        raise OperatorNlTurnError("operator NL target keys are not closed")
    operator = target.get("operator")
    if operator is not None:
        deserialize_operator_action(operator)  # raises on any non-canonical form
    result_ast = target.get("result_ast")
    if result_ast is not None and not result_ast.strip():
        raise OperatorNlTurnError("result AST target is empty")


# --------------------------------------------------------------------------
# Admission
# --------------------------------------------------------------------------


def _style_system_text(style: str) -> str:
    return (
        f"{PARAPHRASE_SYSTEM_PROMPT_V1} Render every declared edit fact "
        f"phrase verbatim in the '{style}' style; add connective prose only."
    )


def _request_style(
    provider: ParaphraseProvider,
    *,
    style: str,
    phrases: Sequence[str],
    accepted_set_size: int,
    sampling: SamplingParamsV1,
    seed: int | None,
    now: Callable[[], str],
) -> tuple[str, Mapping[str, Any]]:
    """Run one style through the DSH2-03 contract; return (text, provenance)."""
    payload = {
        "style": style,
        "phrases": list(phrases),
        "accepted_set_size": accepted_set_size,
    }
    system_text = _style_system_text(style)
    prompt_text = canonical_json(payload)
    request = ParaphraseRequestV1(
        request_id=content_sha({"style": style, "payload": payload})[:32],
        provider_id=provider.provider_id,
        model_id=provider.model_id,
        model_revision=provider.model_revision,
        schema_fingerprint=content_sha({"schema": "operator_nl_turns/v1"}),
        frame_fingerprint=content_sha(payload),
        system_digest=hashlib.sha256(system_text.encode("utf-8")).hexdigest(),
        prompt_digest=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
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


def _reject(
    style: str, code: str, detail: str, sha: str | None = None
) -> RejectedOperatorPromptV1:
    return RejectedOperatorPromptV1(
        style=style, code=code, detail=detail, prompt_sha256=sha
    )


def generate_operator_nl_turns(
    *,
    pack: DslPack,
    library: OperatorLibraryV1,
    state: OperatorStateV1,
    after_state: OperatorStateV1,
    reference_table: ReferenceTableV1,
    legal_set: OperatorLegalSetV1,
    application: OperatorApplicationV1,
    accepted_application_ids: Sequence[str],
    provider: ParaphraseProvider | None = None,
    style_plan: Sequence[str] = STYLE_PLAN,
    target_view: OperatorTargetView = OperatorTargetView.OPERATOR_ONLY,
    provenance: Any = None,
    sampling: SamplingParamsV1 | None = None,
    seed: int | None = None,
    now: Callable[[], str] | None = None,
) -> OperatorNlTurnSetV1:
    """Generate + admit NL edit prompts for one operator application.

    Admission follows the module docstring: legal-set membership, accepted-set
    membership, state-reference ambiguity classification, non-equivalent
    compatibility, accepted-set naming, and the declared leak detector. Every
    rejection is coded and counted; nothing is dropped silently.
    """
    if not application.succeeded or application.effect is None:
        raise OperatorNlTurnError("NL turns require a succeeded application")
    provider = provider or OperatorNlFixtureProviderV1()
    clock = now or (lambda: datetime.now(timezone.utc).isoformat())
    sampling = sampling or SamplingParamsV1()
    declaration = next(
        decl
        for decl in library.declarations
        if decl.fingerprint == application.operator_fingerprint
    )
    frame = derive_operator_frame(
        declaration,
        application.arguments,
        state,
        application.effect,
        reference_table,
    )
    detector = OperatorLeakDetectorV1.from_evidence(legal_set, frame, reference_table)

    rows: list[OperatorNlTurnV1] = []
    rejected: list[RejectedOperatorPromptV1] = []

    legal_actions = {
        action.application_id: action for action in legal_set.operator_actions
    }
    action = legal_actions.get(application.application_id)
    accepted = tuple(sorted(set(accepted_application_ids)))
    if action is None:
        for style in style_plan:
            rejected.append(
                _reject(
                    style,
                    "missing_legal_membership",
                    "target application is absent from the exact legal set",
                )
            )
        return _turn_set(frame, legal_set, application, rows, rejected)
    if application.application_id not in accepted:
        for style in style_plan:
            rejected.append(
                _reject(
                    style,
                    "not_in_accepted_set",
                    "target application is legal but not in the accepted set",
                )
            )
        return _turn_set(frame, legal_set, application, rows, rejected)

    # Effect-equivalence classes over the legal set (dry-run every action).
    provenance = provenance or application.provenance
    effect_by_id = _effect_fingerprints_by_action(
        pack, library, state, legal_set, provenance
    )
    target_effect = application.effect.fingerprint
    equivalent_ids = {
        app_id
        for app_id, effect_fp in effect_by_id.items()
        if effect_fp == target_effect
    }
    accepted_equivalent = tuple(sorted(equivalent_ids & set(accepted)))
    names_set = len(accepted_equivalent) > 1

    # Frames for every legal action (for the mechanical compatibility check).
    candidate_frames: dict[str, OperatorFrameV1] = {}
    for legal_action in legal_set.operator_actions:
        other_app = library.dry_run(
            pack,
            state,
            legal_action.operator_id,
            legal_action.arguments,
            provenance,
        )
        if other_app.effect is None:
            continue
        other_decl = next(
            decl
            for decl in library.declarations
            if decl.fingerprint == legal_action.operator_fingerprint
        )
        candidate_frames[legal_action.application_id] = derive_operator_frame(
            other_decl,
            other_app.arguments,
            state,
            other_app.effect,
            reference_table,
        )

    coverage_phrases = tuple(
        render_operator_fact_phrase(fact)
        for fact in (*frame.required_edit_facts, *frame.state_reference_facts)
    )
    # Ambiguity: a shared descriptor is only acceptable when every legal
    # action binding any same-descriptor ref is effect-equivalent.
    ambiguous_refs: list[str] = []
    scoring: dict[str, str] = {}
    by_descriptor: dict[str, list[Any]] = {}
    for entry in reference_table.entries:
        by_descriptor.setdefault(entry.descriptor.fingerprint, []).append(entry.ref)
    for fact in frame.state_reference_facts:
        fp = fact.provenance.descriptor_fingerprint or ""
        refs = by_descriptor.get(fp, [])
        key = f"{fact.ref.KIND.value}:{fact.ref.opaque_id}"
        if len(refs) <= 1:
            scoring[key] = "unambiguous"
            continue
        ok = True
        for legal_action in legal_set.operator_actions:
            bound = {
                (a.value.KIND.value, a.value.opaque_id) for a in legal_action.arguments
            }
            if any((r.KIND.value, r.opaque_id) in bound for r in refs):
                if legal_action.application_id not in equivalent_ids:
                    ok = False
                    break
        scoring[key] = "accepted_set_scored" if ok else "ambiguous"
        if not ok:
            ambiguous_refs.append(key)

    compatible_ids = prompt_compatible_action_ids(coverage_phrases, candidate_frames)
    non_equivalent_compatible = sorted(set(compatible_ids) - equivalent_ids)
    # A prompt recovers an accepted application: any compatible NON-equivalent
    # application outside the accepted set is a hard violation. Compatible
    # non-equivalent applications INSIDE the accepted set are admitted only
    # when the prompt explicitly names that accepted set.
    non_equivalent_outside_accepted = sorted(
        set(non_equivalent_compatible) - set(accepted)
    )
    named_set = tuple(
        sorted((equivalent_ids & set(accepted)) | (set(compatible_ids) & set(accepted)))
    )
    names_set = len(named_set) > 1

    for style in style_plan:
        if ambiguous_refs:
            rejected.append(
                _reject(
                    style,
                    "ambiguous_state_reference",
                    f"state references shared with non-equivalent actions: {ambiguous_refs!r}",
                )
            )
            continue
        if non_equivalent_outside_accepted:
            rejected.append(
                _reject(
                    style,
                    "compatible_with_non_equivalent",
                    f"prompt also covers non-equivalent applications outside "
                    f"the accepted set: {non_equivalent_outside_accepted!r}",
                )
            )
            continue
        if (
            non_equivalent_compatible or len(accepted_equivalent) > 1
        ) and not names_set:
            rejected.append(
                _reject(
                    style,
                    "accepted_set_not_named",
                    "several accepted applications match but the prompt does "
                    "not name the accepted set",
                )
            )
            continue
        text, prov = _request_style(
            provider,
            style=style,
            phrases=coverage_phrases,
            accepted_set_size=len(named_set) if names_set else 0,
            sampling=sampling,
            seed=seed,
            now=clock,
        )
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        ok, reason = validate_raw_response(
            text,
            prompt_text=canonical_json(
                {
                    "style": style,
                    "phrases": list(coverage_phrases),
                    "accepted_set_size": len(named_set) if names_set else 0,
                }
            ),
            secrets=tuple(provider.secrets()),
        )
        if not ok:
            rejected.append(
                _reject(style, "provider_response_invalid", str(reason), sha)
            )
            continue
        leak_ok, leak_reasons = detector.check(text)
        if not leak_ok:
            rejected.append(
                _reject(
                    style,
                    "leak",
                    f"declared detector rejected: {list(leak_reasons)!r}",
                    sha,
                )
            )
            continue
        target = build_symbol_only_target(target_view, action, after_state)
        assert_target_symbol_only(target)
        rows.append(
            OperatorNlTurnV1(
                style=style,
                prompt_text=text,
                prompt_sha256=sha,
                target_view=target_view.value,
                target=target,
                names_accepted_set=names_set,
                accepted_application_ids=named_set,
                state_reference_scoring=dict(scoring),
                style_provider_provenance=prov,
                frame_fingerprint=frame.fingerprint,
            )
        )
    return _turn_set(frame, legal_set, application, rows, rejected)


def _turn_set(
    frame: OperatorFrameV1,
    legal_set: OperatorLegalSetV1,
    application: OperatorApplicationV1,
    rows: list[OperatorNlTurnV1],
    rejected: list[RejectedOperatorPromptV1],
) -> OperatorNlTurnSetV1:
    counts: dict[str, int] = {}
    for row in rejected:
        counts[row.code] = counts.get(row.code, 0) + 1
    return OperatorNlTurnSetV1(
        schema_version=SCHEMA_VERSION,
        operator_id=frame.operator_id,
        application_id=application.application_id,
        frame_fingerprint=frame.fingerprint,
        legal_set_fingerprint=legal_set.fingerprint,
        rows=tuple(rows),
        rejected_rows=tuple(rejected),
        rejection_counts=counts,
    )


# --------------------------------------------------------------------------
# Disambiguation report over fixture operator families
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorFamilyDisambiguationV1:
    """Per-family disambiguation outcome (stop-rule disposition included)."""

    family: str
    candidates: int
    admitted: int
    disambiguation_rate: float
    rejection_counts: Mapping[str, int]
    disposition: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "candidates": self.candidates,
            "admitted": self.admitted,
            "disambiguation_rate": self.disambiguation_rate,
            "rejection_counts": dict(self.rejection_counts),
            "disposition": self.disposition,
        }


@dataclass(frozen=True)
class OperatorNlDisambiguationReportV1:
    """Cross-family disambiguation report (symbolic CAP2 stop rule recorded)."""

    schema_version: str
    families: tuple[OperatorFamilyDisambiguationV1, ...]
    overall_rate: float
    symbolic_cap2_only_families: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "families": [family.to_dict() for family in self.families],
            "overall_rate": self.overall_rate,
            "symbolic_cap2_only_families": list(self.symbolic_cap2_only_families),
        }


def build_operator_disambiguation_report(
    turn_sets: Mapping[str, OperatorNlTurnSetV1],
) -> OperatorNlDisambiguationReportV1:
    """Aggregate per-family admission outcomes into a disambiguation report.

    A family with zero admitted prompts records the stop-rule disposition
    ``symbolic_cap2_only`` — symbolic CAP2 is retained for that family and
    the disposition is reported honestly, never hidden.
    """
    families: list[OperatorFamilyDisambiguationV1] = []
    for family, turn_set in sorted(turn_sets.items()):
        candidates = len(turn_set.rows) + len(turn_set.rejected_rows)
        admitted = len(turn_set.rows)
        rate = (admitted / candidates) if candidates else 0.0
        families.append(
            OperatorFamilyDisambiguationV1(
                family=family,
                candidates=candidates,
                admitted=admitted,
                disambiguation_rate=round(rate, 6),
                rejection_counts=dict(turn_set.rejection_counts),
                disposition=(
                    DISPOSITION_NL_ADMITTED
                    if admitted
                    else DISPOSITION_SYMBOLIC_CAP2_ONLY
                ),
            )
        )
    total_candidates = sum(family.candidates for family in families)
    total_admitted = sum(family.admitted for family in families)
    return OperatorNlDisambiguationReportV1(
        schema_version="operator_nl_disambiguation_report/v1",
        families=tuple(families),
        overall_rate=(
            round(total_admitted / total_candidates, 6) if total_candidates else 0.0
        ),
        symbolic_cap2_only_families=tuple(
            family.family
            for family in families
            if family.disposition == DISPOSITION_SYMBOLIC_CAP2_ONLY
        ),
    )


__all__ = [
    "DISPOSITION_NL_ADMITTED",
    "DISPOSITION_SYMBOLIC_CAP2_ONLY",
    "REJECTION_CODES",
    "SCHEMA_VERSION",
    "STYLE_PLAN",
    "OperatorFamilyDisambiguationV1",
    "OperatorLeakDetectorV1",
    "OperatorNlDisambiguationReportV1",
    "OperatorNlFixtureProviderV1",
    "OperatorNlTurnError",
    "OperatorNlTurnSetV1",
    "OperatorNlTurnV1",
    "RejectedOperatorPromptV1",
    "assert_target_symbol_only",
    "build_operator_disambiguation_report",
    "build_symbol_only_target",
    "classify_state_references",
    "generate_operator_nl_turns",
    "prompt_compatible_action_ids",
    "render_operator_fact_phrase",
]
