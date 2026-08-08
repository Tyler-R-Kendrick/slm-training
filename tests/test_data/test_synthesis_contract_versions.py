"""Cross-contract schema-version compatibility (SGS-010 / SLM-445).

One table-driven reader-behavior check over every serialized contract this
initiative introduced. Per-contract semantics live with their own tests; this
file certifies the *shared* compatibility policy so a new contract cannot land
with a version field nobody enforces:

* current version round-trips losslessly,
* a newer/unknown version is rejected (never coerced or reinterpreted),
* a missing version is rejected (absence is not "assume current"),
* the static registry in ``scripts.verify_ownership_map`` lists the contract.

Historical artifacts stay readable exactly because a semantics-changing
migration must mint a new version identity (and therefore a new row here)
rather than redefine an existing one in place.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from scripts.verify_ownership_map import SERIALIZED_CONTRACTS
from slm_training.data.progspec.prompt_requirements import (
    PromptSemanticRequirementsV1,
)
from slm_training.data.progspec.synthesis_problem import (
    PackIdentityV1,
    VerifiedSynthesisProblemV1,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.semantic_failure import (
    VerifierWitnessV1,
    build_verifier_witness,
    trace_semantic_failure,
)
from slm_training.models.decode_stats import (
    DecodeStats,
    DecodeStatsRecordV1,
    build_decode_stats_record,
)

Case = tuple[str, str, Callable[[], dict[str, Any]], Callable[[dict[str, Any]], Any]]


def _requirements_payload() -> dict[str, Any]:
    return PromptSemanticRequirementsV1(prompt_context_hash="abc").to_dict()


def _synthesis_payload() -> dict[str, Any]:
    return VerifiedSynthesisProblemV1(
        problem_id="p0", pack_identity=PackIdentityV1(pack_id="openui")
    ).to_dict()


def _witness_payload() -> dict[str, Any]:
    record = ExampleRecord(
        id="x",
        prompt="Build a Button. Placeholders: :cta.label",
        openui='root = Button(":cta.label")',
        split="smoke",
        source="fixture",
    )
    return build_verifier_witness(trace_semantic_failure("root = Stack([])", record)).to_dict()


def _decode_record_payload() -> dict[str, Any]:
    return build_decode_stats_record(
        DecodeStats(), measurement_stage="steady_state"
    ).to_dict()


CASES: tuple[Case, ...] = (
    (
        "PromptSemanticRequirementsV1",
        "requirements_version",
        _requirements_payload,
        PromptSemanticRequirementsV1.from_dict,
    ),
    (
        "VerifiedSynthesisProblemV1",
        "schema_version",
        _synthesis_payload,
        VerifiedSynthesisProblemV1.from_dict,
    ),
    ("VerifierWitnessV1", "schema_version", _witness_payload, VerifierWitnessV1.from_dict),
    (
        "DecodeStatsRecordV1",
        "schema_version",
        _decode_record_payload,
        DecodeStatsRecordV1.from_dict,
    ),
)

_IDS = [case[0] for case in CASES]


@pytest.mark.parametrize("name,field,build,read", CASES, ids=_IDS)
def test_current_version_round_trips(
    name: str, field: str, build: Callable[[], dict[str, Any]], read: Callable[..., Any]
) -> None:
    payload = build()
    assert payload[field], f"{name} serialized without a {field}"
    assert read(payload).to_dict() == payload


@pytest.mark.parametrize("name,field,build,read", CASES, ids=_IDS)
def test_future_version_is_rejected_not_coerced(
    name: str, field: str, build: Callable[[], dict[str, Any]], read: Callable[..., Any]
) -> None:
    payload = build()
    payload[field] = f"{payload[field]}-from-the-future"
    with pytest.raises((ValueError, TypeError)):
        read(payload)


@pytest.mark.parametrize("name,field,build,read", CASES, ids=_IDS)
def test_missing_version_is_rejected(
    name: str, field: str, build: Callable[[], dict[str, Any]], read: Callable[..., Any]
) -> None:
    payload = build()
    del payload[field]
    with pytest.raises((ValueError, TypeError, KeyError)):
        read(payload)


def test_every_case_is_registered_for_static_certification() -> None:
    # The CI-side static check and this runtime check must cover the same set;
    # a contract in one but not the other is exactly the gap SGS-010 closes.
    assert {case[0] for case in CASES} == {c.contract_id for c in SERIALIZED_CONTRACTS}
