"""ParaphraseRequestV1/ResponseV1: provenance-complete NL paraphrase contract.

DSH2-03 (SLM-364): an external LLM may be used to turn a fixed
:class:`~slm_training.dsl.semantic_frame.SemanticFrameV1` (DSH2-02, SLM-363)
into prompt-side natural language, but its output must never become
authoritative or untraceable. This module is the provider-neutral,
content-addressed request/response contract that makes that call auditable
and replayable: every admitted paraphrase resolves exact provider, model,
prompt, schema, and frame provenance, and a raw response only becomes usable
after it passes quarantine validation.

**SLM-343 substitution.** The issue text names ``StructuredDslSchemaV1``.
That class does not exist anywhere in this repository -- SLM-343 ("CAP1-GEN-01:
define the generic NL<->DSL grounding contract and capability certificate")
is a design-doc-only backlog issue with no merged code (see the identical
interpretation note in ``semantic_frame.py``, one issue prior in this same
milestone). This module reuses :class:`~slm_training.dsl.semantic_frame.CAP1SchemaV1`
as the closest already-merged equivalent of "the structured DSL schema" the
issue means, and does not invent a second parallel schema type.

**Only schema + frame are sendable.** :func:`build_prompt_payload` projects
exactly two things into the wire payload: a schema projection and a frame
projection. The frame projection deliberately omits
``SemanticFrameV1.canonical_source`` -- the literal canonical DSL target text
-- because that is exactly the kind of target-only content the issue says
must never be sent (an LLM that can read the literal DSL source is a copier,
not a paraphraser of declared facts, and its output would risk leaking raw
DSL/placeholder syntax into prose, the same failure mode
``docs/design/abstraction-house-style.md`` rejects). No split, label, or other
gold-only field exists on either input type, so nothing else needs excluding.

**Content addressing.** :func:`cache_key_for` hashes exactly the fields that
determine reproducibility -- provider id, model id/revision, prompt/system
digests, sampling params, seed -- via the existing
:func:`slm_training.harness_core.lineage.records.content_sha` primitive
(reused, not reinvented). Request id and timestamps are deliberately excluded:
they are incidental identity, not reproducibility inputs.

**Quarantine before promotion.** Every raw provider response is redacted
(:func:`redact_text`, applied against the provider's own declared
:meth:`ParaphraseProvider.secrets`) and stored as a :class:`QuarantinedResponseV1`
before anything else happens. Only a response that passes
:func:`validate_raw_response` is promoted to a :class:`QuestionArtifactV1`.
Malformed, echoed, secret-leaking, or (after ``max_attempts`` retries)
never-obtained responses stay quarantined or never occupy a cache identity at
all -- they can never reach the promoted store.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from slm_training.dsl.semantic_frame import CAP1SchemaV1, SemanticFrameV1
from slm_training.harness_core.lineage.records import canonical_json, content_sha

PARAPHRASE_SYSTEM_PROMPT_V1 = (
    "You are a natural-language paraphraser for a fixed set of declared DSL "
    "facts. Describe every required and optional fact in plain language. "
    "Never invent a fact absent from the provided schema and frame, and "
    "never state a forbidden fact. Do not reproduce DSL syntax, symbol "
    "identifiers, or placeholder tokens verbatim -- describe their role in "
    "prose instead."
)

_MAX_RESPONSE_CHARS = 4000
_CONTROL_CHARS = frozenset(chr(c) for c in range(0x20) if c not in (0x09, 0x0A, 0x0D))

_FIXTURE_TEMPLATES: tuple[str, ...] = (
    "Build a {dsl} layout with {node_count} elements, {required_count} "
    "required facts, and {effect_count} wired actions.",
    "Create a {dsl} screen with {node_count} components; {optional_count} "
    "facts are left flexible and {effect_count} effects are invoked.",
    "Assemble a {dsl} view of {node_count} nodes covering {required_count} "
    "required and {optional_count} optional facts.",
)


class ProviderError(RuntimeError):
    """Base class for provider-side failures (network, malformed, rate limit)."""


class ProviderRateLimitedError(ProviderError):
    """The provider rejected the call due to rate limiting."""


class ProviderNetworkDisabledError(ProviderError):
    """Raised by a contract-only provider stub with no wired transport."""


class QuarantineIntegrityError(RuntimeError):
    """A promotion was attempted without a validated quarantine entry."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_text(text: str, secrets: Sequence[str]) -> str:
    """Replace every occurrence of a declared secret with a fixed marker.

    Applied to every raw provider response before it is stored, logged, or
    digested, so a provider that (buggily) echoes its own credential can
    never carry it into a persisted artifact.
    """
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


# --------------------------------------------------------------------------
# Typed request/response contracts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingParamsV1:
    """Sampling parameters that affect reproducibility of a paraphrase call."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature out of range: {self.temperature!r}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p out of range: {self.top_p!r}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive: {self.max_tokens!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class ParaphraseRequestV1:
    """Exact identity + reproducibility record for one paraphrase call.

    ``system_digest``/``prompt_digest`` are sha256 digests of the exact
    system/prompt text sent, never the raw text itself (nothing here can
    carry more than the schema+frame projection already declares sendable).
    ``schema_fingerprint``/``frame_fingerprint`` are separate explicit
    provenance fields so a consumer never has to reverse a digest to resolve
    which schema/frame produced a given prompt.
    """

    request_id: str
    provider_id: str
    model_id: str
    model_revision: str
    schema_fingerprint: str
    frame_fingerprint: str
    system_digest: str
    prompt_digest: str
    sampling: SamplingParamsV1
    seed: int | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "schema_fingerprint": self.schema_fingerprint,
            "frame_fingerprint": self.frame_fingerprint,
            "system_digest": self.system_digest,
            "prompt_digest": self.prompt_digest,
            "sampling": self.sampling.to_dict(),
            "seed": self.seed,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ParaphraseResponseV1:
    """Pure response provenance: digest + timestamp, never raw text.

    Raw (redacted) text lives only in :class:`QuarantinedResponseV1` /
    :class:`QuestionArtifactV1`, keyed by the request's cache key.
    """

    response_digest: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {"response_digest": self.response_digest, "created_at": self.created_at}


QuarantineStatus = Literal["pending", "validated", "rejected"]


@dataclass(frozen=True)
class QuarantinedResponseV1:
    """A raw (redacted) provider response, always written before promotion."""

    cache_key: str
    request: ParaphraseRequestV1
    response: ParaphraseResponseV1
    raw_text: str
    status: QuarantineStatus
    quarantined_at: str
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "request": self.request.to_dict(),
            "response": self.response.to_dict(),
            "raw_text": self.raw_text,
            "status": self.status,
            "quarantined_at": self.quarantined_at,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class QuestionArtifactV1:
    """A validated paraphrase, usable downstream only after promotion."""

    cache_key: str
    request: ParaphraseRequestV1
    response: ParaphraseResponseV1
    paraphrase_text: str
    promoted_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "request": self.request.to_dict(),
            "response": self.response.to_dict(),
            "paraphrase_text": self.paraphrase_text,
            "promoted_at": self.promoted_at,
        }


@dataclass(frozen=True)
class ParaphraseOutcomeV1:
    """Result of one :func:`request_paraphrase` call."""

    cache_key: str
    request: ParaphraseRequestV1
    from_cache: bool
    dry_run: bool
    promoted: bool
    artifact: QuestionArtifactV1 | None
    quarantine: QuarantinedResponseV1 | None
    rejection_reason: str | None = None


# --------------------------------------------------------------------------
# Sendable payload projection (schema + frame only)
# --------------------------------------------------------------------------


def _schema_projection(schema: CAP1SchemaV1) -> dict[str, Any]:
    return {
        "dsl": schema.dsl,
        "component_names": sorted(schema.component_names),
        "component_prop_order": {
            name: list(props)
            for name, props in sorted(schema.component_prop_order.items())
        },
        "content_props": sorted(schema.content_props),
        "fingerprint": schema.fingerprint(),
    }


def _frame_projection(frame: SemanticFrameV1) -> dict[str, Any]:
    """Project a frame to its sendable fields -- excludes ``canonical_source``.

    ``canonical_source`` is the literal canonical DSL target text: exactly the
    target-only content the issue says must be excluded (see module
    docstring). Everything else on ``SemanticFrameV1`` is declared fact
    content, safe to send.
    """
    return {
        "dsl": frame.dsl,
        "schema_fingerprint": frame.schema_fingerprint,
        "nodes": [asdict(n) for n in frame.nodes],
        "roles": [asdict(r) for r in frame.roles],
        "relations": [asdict(r) for r in frame.relations],
        "effects": [asdict(e) for e in frame.effects],
        "facts": [asdict(f) for f in frame.facts],
    }


def build_prompt_payload(
    schema: CAP1SchemaV1, frame: SemanticFrameV1
) -> dict[str, Any]:
    """The only sendable content for a paraphrase request: schema + frame."""
    return {"schema": _schema_projection(schema), "frame": _frame_projection(frame)}


def cache_key_for(request: ParaphraseRequestV1) -> str:
    """Content-addressed key over exactly the reproducibility-determining fields.

    Deliberately excludes ``request_id`` and every timestamp: those are
    incidental identity, not inputs to what the provider would return.
    """
    payload = {
        "provider_id": request.provider_id,
        "model_id": request.model_id,
        "model_revision": request.model_revision,
        "prompt_digest": request.prompt_digest,
        "system_digest": request.system_digest,
        "sampling": request.sampling.to_dict(),
        "seed": request.seed,
    }
    return content_sha(payload)


def validate_raw_response(
    raw_text: str | None, *, prompt_text: str, secrets: Sequence[str]
) -> tuple[bool, str | None]:
    """Well-formedness + shape + leak checks a raw response must pass to promote."""
    if raw_text is None or not raw_text.strip():
        return False, "response text is empty"
    if raw_text.strip() == prompt_text.strip():
        return (
            False,
            "response echoes the raw structured payload instead of paraphrasing it",
        )
    if len(raw_text) > _MAX_RESPONSE_CHARS:
        return False, f"response exceeds max length ({_MAX_RESPONSE_CHARS} chars)"
    if any(ch in _CONTROL_CHARS for ch in raw_text):
        return False, "response contains control characters"
    for secret in secrets:
        if secret and secret in raw_text:
            return False, "response leaks a registered secret"
    return True, None


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------


class ParaphraseProvider(Protocol):
    """Structural contract every paraphrase backend implements."""

    provider_id: str
    model_id: str
    model_revision: str

    def generate(
        self,
        *,
        system_text: str,
        prompt_text: str,
        sampling: SamplingParamsV1,
        seed: int | None,
    ) -> str:
        """Return raw provider text, or raise :class:`ProviderError`."""
        ...

    def secrets(self) -> tuple[str, ...]:
        """Return every credential/secret string this provider holds."""
        ...


@dataclass
class DeterministicFixtureProviderV1:
    """Fully offline, credential-free deterministic paraphraser.

    No network call and no external state: the same
    ``(system_text, prompt_text, sampling, seed)`` always yields the same
    text, selected by hashing the exact call inputs with the shared
    ``content_sha`` primitive. This is the provider every acceptance test in
    this module exercises end-to-end.
    """

    provider_id: str = "fixture-deterministic-v1"
    model_id: str = "fixture-template-paraphraser"
    model_revision: str = "v1"

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
        frame = payload.get("frame") or {}
        schema = payload.get("schema") or {}
        facts = frame.get("facts") or []
        required_count = sum(1 for f in facts if f.get("category") == "required")
        optional_count = sum(1 for f in facts if f.get("category") == "optional")
        selector = content_sha(
            {
                "system": system_text,
                "prompt": prompt_text,
                "sampling": sampling.to_dict(),
                "seed": seed,
            }
        )
        index = int(selector[:8], 16) % len(_FIXTURE_TEMPLATES)
        template = _FIXTURE_TEMPLATES[index]
        return template.format(
            dsl=schema.get("dsl", "dsl"),
            node_count=len(frame.get("nodes") or ()),
            effect_count=len(frame.get("effects") or ()),
            required_count=required_count,
            optional_count=optional_count,
        )


class ApiKeyStubProviderV1:
    """Structurally-wired real-API-shaped provider stub.

    Ships to make "pluggable providers" real, not to call a real LLM: this
    sandbox has no network access, and this issue is about the
    request/response provenance contract, not a hosted-API integration.
    Without an injected ``transport`` callable, :meth:`generate` raises
    :class:`ProviderNetworkDisabledError` -- so a bare instance never performs
    a network call. Tests inject a deterministic fake transport (a plain
    Python callable, never a socket) to exercise the full contract, including
    the redaction negative control, without ever touching a network.
    """

    def __init__(
        self,
        *,
        provider_id: str,
        model_id: str,
        model_revision: str,
        api_key: str,
        endpoint: str = "",
        transport: Callable[[str, str], str] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self.model_revision = model_revision
        self.endpoint = endpoint
        self._api_key = api_key
        self._transport = transport

    def secrets(self) -> tuple[str, ...]:
        return (self._api_key,)

    def generate(
        self,
        *,
        system_text: str,
        prompt_text: str,
        sampling: SamplingParamsV1,
        seed: int | None,
    ) -> str:
        if self._transport is None:
            raise ProviderNetworkDisabledError(
                "no transport wired; this stub never performs a real network call"
            )
        return self._transport(system_text, prompt_text)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(provider_id={self.provider_id!r}, "
            f"model_id={self.model_id!r}, model_revision={self.model_revision!r}, "
            f"endpoint={self.endpoint!r}, api_key='[REDACTED]')"
        )

    __str__ = __repr__


# --------------------------------------------------------------------------
# Replay cache + quarantine store
# --------------------------------------------------------------------------


class ReplayCache:
    """In-memory content-addressed quarantine + promoted-artifact store.

    The same cache key always resolves to the one quarantine entry (and, once
    validated, the one promoted question artifact) written for it -- a
    provider is never called twice for the same request identity.
    """

    def __init__(self) -> None:
        self._quarantine: dict[str, QuarantinedResponseV1] = {}
        self._promoted: dict[str, QuestionArtifactV1] = {}

    def get_quarantine(self, key: str) -> QuarantinedResponseV1 | None:
        return self._quarantine.get(key)

    def put_quarantine(self, entry: QuarantinedResponseV1) -> None:
        self._quarantine[entry.cache_key] = entry

    def get_promoted(self, key: str) -> QuestionArtifactV1 | None:
        return self._promoted.get(key)

    def promote(self, key: str, artifact: QuestionArtifactV1) -> None:
        entry = self._quarantine.get(key)
        if entry is None or entry.status != "validated":
            raise QuarantineIntegrityError(
                f"cannot promote {key!r}: quarantine entry is not validated"
            )
        self._promoted[key] = artifact

    def __len__(self) -> int:
        return len(self._quarantine)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def request_paraphrase(
    provider: ParaphraseProvider,
    schema: CAP1SchemaV1,
    frame: SemanticFrameV1,
    cache: ReplayCache,
    *,
    sampling: SamplingParamsV1 | None = None,
    seed: int | None = None,
    request_id: str | None = None,
    dry_run: bool = False,
    max_attempts: int = 1,
    now: Callable[[], str] | None = None,
) -> ParaphraseOutcomeV1:
    """Run one provenance-complete paraphrase request through cache + quarantine.

    Same request identity (``cache_key_for``) always reuses the one cached
    quarantine/promoted artifact and never calls the provider twice. In
    ``dry_run`` mode the request/cache key are built and returned without
    ever calling the provider or writing to the cache. On total provider
    failure (including after ``max_attempts`` retries) nothing is written to
    the cache -- a partial/failed call can never occupy or publish an
    identity slot. A response that fails :func:`validate_raw_response` is
    quarantined with ``status="rejected"`` and never promoted.
    """
    sampling = sampling or SamplingParamsV1()
    clock = now or _utcnow_iso
    payload = build_prompt_payload(schema, frame)
    system_text = PARAPHRASE_SYSTEM_PROMPT_V1
    prompt_text = canonical_json(payload)
    request = ParaphraseRequestV1(
        request_id=request_id or uuid.uuid4().hex,
        provider_id=provider.provider_id,
        model_id=provider.model_id,
        model_revision=provider.model_revision,
        schema_fingerprint=schema.fingerprint(),
        frame_fingerprint=content_sha(_frame_projection(frame)),
        system_digest=_sha256_text(system_text),
        prompt_digest=_sha256_text(prompt_text),
        sampling=sampling,
        seed=seed,
        created_at=clock(),
    )
    key = cache_key_for(request)

    promoted = cache.get_promoted(key)
    if promoted is not None:
        return ParaphraseOutcomeV1(
            cache_key=key,
            request=request,
            from_cache=True,
            dry_run=False,
            promoted=True,
            artifact=promoted,
            quarantine=cache.get_quarantine(key),
        )

    existing = cache.get_quarantine(key)
    if existing is not None:
        return ParaphraseOutcomeV1(
            cache_key=key,
            request=request,
            from_cache=True,
            dry_run=False,
            promoted=False,
            artifact=None,
            quarantine=existing,
            rejection_reason=existing.rejection_reason,
        )

    if dry_run:
        return ParaphraseOutcomeV1(
            cache_key=key,
            request=request,
            from_cache=False,
            dry_run=True,
            promoted=False,
            artifact=None,
            quarantine=None,
        )

    secrets = tuple(provider.secrets())
    raw_text: str | None = None
    last_error: ProviderError | None = None
    for _ in range(max(1, max_attempts)):
        try:
            raw_text = provider.generate(
                system_text=system_text,
                prompt_text=prompt_text,
                sampling=sampling,
                seed=seed,
            )
            last_error = None
            break
        except ProviderError as exc:
            last_error = exc
            raw_text = None
            continue

    if last_error is not None or raw_text is None:
        # Total failure: nothing is cached or quarantined, so a retried,
        # rate-limited, or otherwise failed call never occupies this
        # identity's cache slot and can never later look like a completed row.
        raise (
            last_error
            if last_error is not None
            else ProviderError("provider returned no response and raised no error")
        )

    redacted_text = redact_text(raw_text, secrets)
    response = ParaphraseResponseV1(
        response_digest=_sha256_text(redacted_text), created_at=clock()
    )
    ok, reason = validate_raw_response(
        redacted_text, prompt_text=prompt_text, secrets=secrets
    )
    quarantine_entry = QuarantinedResponseV1(
        cache_key=key,
        request=request,
        response=response,
        raw_text=redacted_text,
        status="validated" if ok else "rejected",
        quarantined_at=clock(),
        rejection_reason=None if ok else reason,
    )
    cache.put_quarantine(quarantine_entry)
    if not ok:
        return ParaphraseOutcomeV1(
            cache_key=key,
            request=request,
            from_cache=False,
            dry_run=False,
            promoted=False,
            artifact=None,
            quarantine=quarantine_entry,
            rejection_reason=reason,
        )

    artifact = QuestionArtifactV1(
        cache_key=key,
        request=request,
        response=response,
        paraphrase_text=redacted_text,
        promoted_at=clock(),
    )
    cache.promote(key, artifact)
    return ParaphraseOutcomeV1(
        cache_key=key,
        request=request,
        from_cache=False,
        dry_run=False,
        promoted=True,
        artifact=artifact,
        quarantine=quarantine_entry,
    )


__all__ = [
    "PARAPHRASE_SYSTEM_PROMPT_V1",
    "ApiKeyStubProviderV1",
    "DeterministicFixtureProviderV1",
    "ParaphraseOutcomeV1",
    "ParaphraseProvider",
    "ParaphraseRequestV1",
    "ParaphraseResponseV1",
    "ProviderError",
    "ProviderNetworkDisabledError",
    "ProviderRateLimitedError",
    "QuarantineIntegrityError",
    "QuarantineStatus",
    "QuarantinedResponseV1",
    "QuestionArtifactV1",
    "ReplayCache",
    "SamplingParamsV1",
    "build_prompt_payload",
    "cache_key_for",
    "redact_text",
    "request_paraphrase",
    "validate_raw_response",
]
