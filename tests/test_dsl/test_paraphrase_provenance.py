"""DSH2-03: provenance-complete NL paraphrase provider + cache contract."""

from __future__ import annotations

import json

import pytest

from tests.casefiles import case_values

from slm_training.dsl.paraphrase_provenance import (
    ApiKeyStubProviderV1,
    DeterministicFixtureProviderV1,
    ParaphraseProvider,
    ProviderError,
    ProviderRateLimitedError,
    QuarantineIntegrityError,
    ReplayCache,
    SamplingParamsV1,
    build_prompt_payload,
    cache_key_for,
    request_paraphrase,
    validate_raw_response,
)
from slm_training.dsl.semantic_frame import CAP1SchemaV1, derive_semantic_frame

CARD_SRC = (
    'a = TextContent(":ns.body", 14)\n'
    'root = Card([a], "filled", "column", 8, "start", "start", "nowrap")\n'
)
CALLOUT_SRC = 'root = Callout("info", ":ns.title", ":ns.desc", true)'


@pytest.fixture(scope="module")
def schema() -> CAP1SchemaV1:
    return CAP1SchemaV1.from_pack("openui")


@pytest.fixture()
def frame(schema: CAP1SchemaV1):
    return derive_semantic_frame(CARD_SRC, schema)


class _CountingProvider:
    """Wraps a deterministic provider and counts real ``generate`` calls."""

    def __init__(self, inner: ParaphraseProvider) -> None:
        self._inner = inner
        self.provider_id = inner.provider_id
        self.model_id = inner.model_id
        self.model_revision = inner.model_revision
        self.calls = 0

    def secrets(self) -> tuple[str, ...]:
        return self._inner.secrets()

    def generate(self, **kwargs) -> str:
        self.calls += 1
        return self._inner.generate(**kwargs)


class _EmptyProvider:
    provider_id = "empty-fixture"
    model_id = "empty"
    model_revision = "v1"

    def secrets(self) -> tuple[str, ...]:
        return ()

    def generate(self, **kwargs) -> str:
        return ""


class _EchoProvider:
    provider_id = "echo-fixture"
    model_id = "echo"
    model_revision = "v1"

    def secrets(self) -> tuple[str, ...]:
        return ()

    def generate(self, *, prompt_text: str, **kwargs) -> str:
        return prompt_text


class _FlakyProvider:
    """Fails ``fail_times`` calls with a rate-limit error, then succeeds."""

    provider_id = "flaky-fixture"
    model_id = "flaky"
    model_revision = "v1"

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def secrets(self) -> tuple[str, ...]:
        return ()

    def generate(self, **kwargs) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderRateLimitedError("rate limited")
        return "a deterministic recovered paraphrase of the declared facts"


class _AlwaysFailsProvider:
    provider_id = "always-fails"
    model_id = "fail"
    model_revision = "v1"

    def secrets(self) -> tuple[str, ...]:
        return ()

    def generate(self, **kwargs) -> str:
        raise ProviderError("upstream exploded")


# --------------------------------------------------------------------------
# Payload projection: schema + frame only
# --------------------------------------------------------------------------


def test_prompt_payload_contains_only_schema_and_frame_keys(schema, frame) -> None:
    payload = build_prompt_payload(schema, frame)
    assert set(payload) == {"schema", "frame"}
    assert payload["schema"]["dsl"] == "openui"
    assert payload["frame"]["dsl"] == "openui"


def test_prompt_payload_excludes_canonical_dsl_source_text(schema, frame) -> None:
    payload = build_prompt_payload(schema, frame)
    blob = json.dumps(payload)
    assert "canonical_source" not in payload["frame"]
    # The literal DSL target text must never appear in the sendable payload.
    assert "Card([" not in blob
    assert frame.canonical_source not in blob


def test_prompt_payload_has_no_split_or_label_fields(schema, frame) -> None:
    payload = build_prompt_payload(schema, frame)
    blob = json.dumps(payload).lower()
    for forbidden in ("split", "gold", "eval_only", "target_kind"):
        assert forbidden not in blob


# --------------------------------------------------------------------------
# Content-addressed cache key
# --------------------------------------------------------------------------


def test_cache_key_ignores_request_id_and_timestamp(schema, frame) -> None:
    provider = DeterministicFixtureProviderV1()
    cache_a = ReplayCache()
    cache_b = ReplayCache()
    out_a = request_paraphrase(
        provider,
        schema,
        frame,
        cache_a,
        request_id="req-a",
        now=lambda: "2026-01-01T00:00:00+00:00",
    )
    out_b = request_paraphrase(
        provider,
        schema,
        frame,
        cache_b,
        request_id="req-b",
        now=lambda: "2099-12-31T23:59:59+00:00",
    )
    assert out_a.cache_key == out_b.cache_key
    assert out_a.request.request_id != out_b.request.request_id
    assert out_a.request.created_at != out_b.request.created_at


def test_cache_key_changes_with_seed_and_sampling(schema, frame) -> None:
    provider = DeterministicFixtureProviderV1()
    cache = ReplayCache()
    base = request_paraphrase(provider, schema, frame, cache, seed=1)
    different_seed = request_paraphrase(provider, schema, frame, cache, seed=2)
    different_sampling = request_paraphrase(
        provider,
        schema,
        frame,
        cache,
        seed=1,
        sampling=SamplingParamsV1(temperature=0.7),
    )
    assert base.cache_key != different_seed.cache_key
    assert base.cache_key != different_sampling.cache_key


def test_cache_key_matches_manual_computation(schema, frame) -> None:
    provider = DeterministicFixtureProviderV1()
    cache = ReplayCache()
    out = request_paraphrase(provider, schema, frame, cache)
    assert cache_key_for(out.request) == out.cache_key


# --------------------------------------------------------------------------
# Acceptance: same request identity reuses one cached artifact
# --------------------------------------------------------------------------


def test_same_identity_reuses_one_cached_artifact_without_recalling_provider(
    schema, frame
) -> None:
    provider = _CountingProvider(DeterministicFixtureProviderV1())
    cache = ReplayCache()
    first = request_paraphrase(provider, schema, frame, cache, seed=7)
    second = request_paraphrase(provider, schema, frame, cache, seed=7)
    assert provider.calls == 1
    assert first.promoted and second.promoted
    assert second.from_cache is True
    assert first.artifact is not None and second.artifact is not None
    assert first.artifact.cache_key == second.artifact.cache_key
    assert first.artifact.paraphrase_text == second.artifact.paraphrase_text


# --------------------------------------------------------------------------
# Acceptance: exact provenance resolution
# --------------------------------------------------------------------------


def test_promoted_artifact_resolves_exact_provenance(schema, frame) -> None:
    provider = DeterministicFixtureProviderV1()
    cache = ReplayCache()
    outcome = request_paraphrase(provider, schema, frame, cache, seed=3)
    assert outcome.promoted
    artifact = outcome.artifact
    assert artifact is not None
    assert artifact.request.provider_id == provider.provider_id
    assert artifact.request.model_id == provider.model_id
    assert artifact.request.model_revision == provider.model_revision
    assert artifact.request.schema_fingerprint == schema.fingerprint()
    assert len(artifact.request.frame_fingerprint) == 64
    assert len(artifact.request.prompt_digest) == 64
    assert len(artifact.request.system_digest) == 64
    # response digest is over the (redacted) response text, not the prompt.
    import hashlib

    assert (
        artifact.response.response_digest
        == hashlib.sha256(artifact.paraphrase_text.encode("utf-8")).hexdigest()
    )


def test_distinct_frames_yield_distinct_frame_fingerprints(schema) -> None:
    provider = DeterministicFixtureProviderV1()
    frame_a = derive_semantic_frame(CARD_SRC, schema)
    frame_b = derive_semantic_frame(CALLOUT_SRC, schema)
    cache = ReplayCache()
    out_a = request_paraphrase(provider, schema, frame_a, cache)
    out_b = request_paraphrase(provider, schema, frame_b, cache)
    assert out_a.request.frame_fingerprint != out_b.request.frame_fingerprint
    assert out_a.cache_key != out_b.cache_key


# --------------------------------------------------------------------------
# Dry-run mode
# --------------------------------------------------------------------------


def test_dry_run_never_calls_provider_or_writes_cache(schema, frame) -> None:
    provider = _CountingProvider(DeterministicFixtureProviderV1())
    cache = ReplayCache()
    outcome = request_paraphrase(provider, schema, frame, cache, dry_run=True)
    assert outcome.dry_run is True
    assert outcome.promoted is False
    assert outcome.artifact is None
    assert provider.calls == 0
    assert len(cache) == 0
    assert cache.get_promoted(outcome.cache_key) is None


# --------------------------------------------------------------------------
# Negative control: malformed response stays quarantined, never promoted
# --------------------------------------------------------------------------


def test_empty_response_stays_quarantined_and_is_never_promoted(schema, frame) -> None:
    provider = _EmptyProvider()
    cache = ReplayCache()
    outcome = request_paraphrase(provider, schema, frame, cache)
    assert outcome.promoted is False
    assert outcome.artifact is None
    assert outcome.quarantine is not None
    assert outcome.quarantine.status == "rejected"
    assert outcome.quarantine.rejection_reason is not None
    assert cache.get_promoted(outcome.cache_key) is None
    # Repeating the identical identity replays the same rejected quarantine
    # entry and still never promotes it.
    repeat = request_paraphrase(provider, schema, frame, cache)
    assert repeat.promoted is False
    assert repeat.from_cache is True
    assert cache.get_promoted(outcome.cache_key) is None


def test_echoed_structured_payload_is_rejected_not_promoted(schema, frame) -> None:
    provider = _EchoProvider()
    cache = ReplayCache()
    outcome = request_paraphrase(provider, schema, frame, cache)
    assert outcome.promoted is False
    assert outcome.quarantine is not None
    assert outcome.quarantine.status == "rejected"
    assert "echoes" in (outcome.rejection_reason or "")


def test_validate_raw_response_rejects_empty_and_echo_and_secret_leak() -> None:
    ok, reason = validate_raw_response("", prompt_text="p", secrets=())
    assert not ok and reason
    ok, reason = validate_raw_response("p", prompt_text="p", secrets=())
    assert not ok and "echo" in reason
    ok, reason = validate_raw_response(
        "has sk-secret inside", prompt_text="x", secrets=("sk-secret",)
    )
    assert not ok and "leak" in reason
    ok, reason = validate_raw_response("a fine paraphrase", prompt_text="x", secrets=())
    assert ok and reason is None


def test_quarantine_promote_refuses_without_validated_entry() -> None:
    cache = ReplayCache()
    fake_artifact = object()
    with pytest.raises(QuarantineIntegrityError):
        cache.promote("nonexistent-key", fake_artifact)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Partial failure / retry / rate limit never publish incomplete rows
# --------------------------------------------------------------------------


def test_total_provider_failure_publishes_nothing(schema, frame) -> None:
    provider = _AlwaysFailsProvider()
    cache = ReplayCache()
    with pytest.raises(ProviderError):
        request_paraphrase(provider, schema, frame, cache)
    assert len(cache) == 0


def test_retry_recovers_and_promotes_only_the_final_successful_response(
    schema, frame
) -> None:
    provider = _FlakyProvider(fail_times=1)
    cache = ReplayCache()
    outcome = request_paraphrase(provider, schema, frame, cache, max_attempts=2)
    assert outcome.promoted is True
    assert provider.calls == 2
    assert len(cache) == 1


def test_rate_limit_without_enough_retries_publishes_nothing(schema, frame) -> None:
    provider = _FlakyProvider(fail_times=5)
    cache = ReplayCache()
    with pytest.raises(ProviderRateLimitedError):
        request_paraphrase(provider, schema, frame, cache, max_attempts=2)
    assert len(cache) == 0
    assert provider.calls == 2


# --------------------------------------------------------------------------
# Negative control: secret redaction
# --------------------------------------------------------------------------


def test_secret_never_leaks_into_any_persisted_or_repr_form(schema, frame) -> None:
    secret = "sk-super-secret-fake-token-should-never-leak"

    def leaky_transport(system_text: str, prompt_text: str) -> str:
        # Simulates a buggy upstream that echoes the credential back.
        return f"paraphrase generated using credential {secret} for this request"

    provider = ApiKeyStubProviderV1(
        provider_id="openai-shaped-stub",
        model_id="gpt-shaped",
        model_revision="2026-01-01",
        api_key=secret,
        endpoint="https://example.invalid/v1/responses",
        transport=leaky_transport,
    )
    cache = ReplayCache()
    outcome = request_paraphrase(provider, schema, frame, cache)

    assert outcome.promoted is True
    assert outcome.artifact is not None
    assert outcome.quarantine is not None

    serialized_quarantine = json.dumps(outcome.quarantine.to_dict())
    serialized_artifact = json.dumps(outcome.artifact.to_dict())
    assert secret not in serialized_quarantine
    assert secret not in serialized_artifact
    assert secret not in outcome.quarantine.raw_text
    assert secret not in outcome.artifact.paraphrase_text
    assert "[REDACTED]" in outcome.artifact.paraphrase_text

    assert secret not in repr(provider)
    assert secret not in str(provider)


def test_api_key_stub_never_calls_network_without_a_wired_transport(
    schema, frame
) -> None:
    provider = ApiKeyStubProviderV1(
        provider_id="openai-shaped-stub",
        model_id="gpt-shaped",
        model_revision="2026-01-01",
        api_key="sk-unused",
    )
    cache = ReplayCache()
    with pytest.raises(ProviderError):
        request_paraphrase(provider, schema, frame, cache)
    assert len(cache) == 0


# --------------------------------------------------------------------------
# SamplingParamsV1 validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    case_values(__file__, "test_sampling_params_rejects_out_of_range_values"),
)
def test_sampling_params_rejects_out_of_range_values(kwargs) -> None:
    with pytest.raises(ValueError):
        SamplingParamsV1(**kwargs)


# --------------------------------------------------------------------------
# Deterministic fixture provider
# --------------------------------------------------------------------------


def test_deterministic_fixture_provider_is_deterministic(schema, frame) -> None:
    provider = DeterministicFixtureProviderV1()
    payload = build_prompt_payload(schema, frame)
    from slm_training.harness_core.lineage.records import canonical_json

    prompt_text = canonical_json(payload)
    sampling = SamplingParamsV1(temperature=0.2)
    first = provider.generate(
        system_text="sys", prompt_text=prompt_text, sampling=sampling, seed=5
    )
    second = provider.generate(
        system_text="sys", prompt_text=prompt_text, sampling=sampling, seed=5
    )
    assert first == second
    assert first.strip()
    assert provider.secrets() == ()


def test_deterministic_fixture_provider_output_changes_with_seed(schema, frame) -> None:
    provider = DeterministicFixtureProviderV1()
    payload = build_prompt_payload(schema, frame)
    from slm_training.harness_core.lineage.records import canonical_json

    prompt_text = canonical_json(payload)
    sampling = SamplingParamsV1()
    outputs = {
        provider.generate(
            system_text="sys", prompt_text=prompt_text, sampling=sampling, seed=seed
        )
        for seed in range(6)
    }
    # Not every seed need land on a distinct template, but they must not all
    # collide onto exactly one.
    assert len(outputs) > 1
