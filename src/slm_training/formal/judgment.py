"""Total solver judgment (KERN-01 / SLM-516).

Mirrors ``LeverProofLean.Judgment`` with typed payloads, explicit authority
rules, and lossless adapters from VSS / closure / support-certificate
outcomes. Theorem-preflight lifecycle status stays separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

JUDGMENT_SCHEMA = "solver_judgment/v1"

JudgmentOutcomeName = Literal["witnessed", "refuted", "unknown", "invalid"]
UnknownReasonName = Literal[
    "timeout",
    "skipped_replay",
    "unsupported_capability",
    "incomplete_coverage",
    "missing_tool",
    "inconclusive",
]
InvalidReasonName = Literal["malformed_input", "payload_mismatch"]
VssVerdictName = Literal["supported", "unsupported", "unknown"]


class JudgmentOutcome(str, Enum):
    WITNESSED = "witnessed"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    INVALID = "invalid"


class UnknownReason(str, Enum):
    TIMEOUT = "timeout"
    SKIPPED_REPLAY = "skipped_replay"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    INCOMPLETE_COVERAGE = "incomplete_coverage"
    MISSING_TOOL = "missing_tool"
    INCONCLUSIVE = "inconclusive"


class InvalidReason(str, Enum):
    MALFORMED_INPUT = "malformed_input"
    PAYLOAD_MISMATCH = "payload_mismatch"


@dataclass(frozen=True)
class WitnessPayloadV1:
    witness_digest: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {"witness_digest": self.witness_digest, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WitnessPayloadV1:
        return cls(
            witness_digest=str(data["witness_digest"]),
            source=str(data["source"]),
        )


@dataclass(frozen=True)
class RefutationPayloadV1:
    refutation_digest: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {"refutation_digest": self.refutation_digest, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RefutationPayloadV1:
        return cls(
            refutation_digest=str(data["refutation_digest"]),
            source=str(data["source"]),
        )


@dataclass(frozen=True)
class ClassificationInputsV1:
    malformed_input: bool = False
    payload_mismatch: bool = False
    timed_out: bool = False
    skipped_replay: bool = False
    unsupported_capability: bool = False
    incomplete_coverage: bool = False
    missing_tool: bool = False
    vss_verdict: VssVerdictName | None = None
    has_witness_digest: bool = False
    has_counterexample_digest: bool = False
    exhausted: bool = False
    replay_checked: bool = False
    replay_ok: bool = False


@dataclass(frozen=True)
class SolverJudgmentV1:
    outcome: JudgmentOutcome
    payload: (
        WitnessPayloadV1
        | RefutationPayloadV1
        | UnknownReason
        | InvalidReason
    )
    checked: bool

    def authorizes_semantic_conclusion(self) -> bool:
        return self.checked and self.outcome in (
            JudgmentOutcome.WITNESSED,
            JudgmentOutcome.REFUTED,
        )

    def authorizes_removal(self) -> bool:
        return (
            self.checked
            and self.outcome is JudgmentOutcome.REFUTED
            and isinstance(self.payload, RefutationPayloadV1)
        )

    def payload_matches_outcome(self) -> bool:
        if self.outcome is JudgmentOutcome.WITNESSED:
            return isinstance(self.payload, WitnessPayloadV1)
        if self.outcome is JudgmentOutcome.REFUTED:
            return isinstance(self.payload, RefutationPayloadV1)
        if self.outcome is JudgmentOutcome.UNKNOWN:
            return isinstance(self.payload, UnknownReason)
        return isinstance(self.payload, InvalidReason)

    def __bool__(self) -> bool:
        raise TypeError(
            "SolverJudgmentV1 forbids truthy/falsy coercion; "
            "call authorizes_semantic_conclusion() explicitly"
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any]
        if isinstance(self.payload, WitnessPayloadV1):
            payload = {"kind": "witness", **self.payload.to_dict()}
        elif isinstance(self.payload, RefutationPayloadV1):
            payload = {"kind": "refutation", **self.payload.to_dict()}
        elif isinstance(self.payload, UnknownReason):
            payload = {"kind": "unknown_reason", "reason": self.payload.value}
        else:
            payload = {"kind": "invalid_reason", "reason": self.payload.value}
        return {
            "schema": JUDGMENT_SCHEMA,
            "outcome": self.outcome.value,
            "payload": payload,
            "checked": self.checked,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolverJudgmentV1:
        if data.get("schema") != JUDGMENT_SCHEMA:
            raise ValueError(f"expected schema {JUDGMENT_SCHEMA!r}")
        outcome = JudgmentOutcome(str(data["outcome"]))
        raw = data["payload"]
        if not isinstance(raw, dict):
            raise ValueError("payload must be an object")
        kind = raw.get("kind")
        if kind == "witness":
            payload: (
                WitnessPayloadV1
                | RefutationPayloadV1
                | UnknownReason
                | InvalidReason
            ) = WitnessPayloadV1.from_dict(raw)
        elif kind == "refutation":
            payload = RefutationPayloadV1.from_dict(raw)
        elif kind == "unknown_reason":
            payload = UnknownReason(str(raw["reason"]))
        elif kind == "invalid_reason":
            payload = InvalidReason(str(raw["reason"]))
        else:
            raise ValueError(f"unknown payload kind: {kind!r}")
        judgment = cls(
            outcome=outcome,
            payload=payload,
            checked=bool(data["checked"]),
        )
        if not judgment.payload_matches_outcome():
            raise ValueError("status tag / payload mismatch")
        return judgment


def _classify_unknown_reason(inputs: ClassificationInputsV1) -> UnknownReason:
    if inputs.timed_out:
        return UnknownReason.TIMEOUT
    if inputs.skipped_replay:
        return UnknownReason.SKIPPED_REPLAY
    if inputs.unsupported_capability:
        return UnknownReason.UNSUPPORTED_CAPABILITY
    if inputs.incomplete_coverage:
        return UnknownReason.INCOMPLETE_COVERAGE
    if inputs.missing_tool:
        return UnknownReason.MISSING_TOOL
    return UnknownReason.INCONCLUSIVE


def _classify_invalid_reason(inputs: ClassificationInputsV1) -> InvalidReason:
    if inputs.malformed_input:
        return InvalidReason.MALFORMED_INPUT
    return InvalidReason.PAYLOAD_MISMATCH


def classify_outcome(inputs: ClassificationInputsV1) -> JudgmentOutcome:
    if inputs.malformed_input or inputs.payload_mismatch:
        return JudgmentOutcome.INVALID
    if (
        inputs.timed_out
        or inputs.skipped_replay
        or inputs.unsupported_capability
        or inputs.incomplete_coverage
        or inputs.missing_tool
    ):
        return JudgmentOutcome.UNKNOWN
    if inputs.vss_verdict is None:
        return JudgmentOutcome.UNKNOWN
    if inputs.vss_verdict == "supported":
        if inputs.has_witness_digest and inputs.replay_checked and inputs.replay_ok:
            return JudgmentOutcome.WITNESSED
        return JudgmentOutcome.UNKNOWN
    if inputs.vss_verdict == "unsupported":
        if inputs.has_witness_digest:
            return JudgmentOutcome.INVALID
        if inputs.exhausted and inputs.replay_checked and inputs.replay_ok:
            return JudgmentOutcome.REFUTED
        return JudgmentOutcome.UNKNOWN
    return JudgmentOutcome.UNKNOWN


def classify_payload(
    inputs: ClassificationInputsV1, outcome: JudgmentOutcome
) -> (
    WitnessPayloadV1
    | RefutationPayloadV1
    | UnknownReason
    | InvalidReason
):
    if outcome is JudgmentOutcome.WITNESSED:
        return WitnessPayloadV1(
            witness_digest="checked",
            source="vss_certificate",
        )
    if outcome is JudgmentOutcome.REFUTED:
        digest = "checked"
        source = "vss_exhaustion"
        if inputs.has_counterexample_digest:
            source = "counterexample"
        return RefutationPayloadV1(refutation_digest=digest, source=source)
    if outcome is JudgmentOutcome.UNKNOWN:
        return _classify_unknown_reason(inputs)
    return _classify_invalid_reason(inputs)


def classify(inputs: ClassificationInputsV1, *, checked: bool) -> SolverJudgmentV1:
    outcome = classify_outcome(inputs)
    payload = classify_payload(inputs, outcome)
    judgment = SolverJudgmentV1(outcome=outcome, payload=payload, checked=checked)
    if not judgment.payload_matches_outcome():
        raise ValueError("internal classification produced status/payload mismatch")
    return judgment


def _certificate_signal_flags(
    certificate: Any,
    *,
    replay_checked: bool,
    replay_ok: bool,
    skipped_replay: bool = False,
    missing_tool: bool = False,
) -> ClassificationInputsV1:
    from slm_training.dsl.solver.state import SupportVerdict

    stop = certificate.stop_reason
    timed_out = stop is not None and "timeout" in str(stop).lower()
    incomplete = any(
        cov in {"partial", "none"} for cov in certificate.coverage_observations
    )
    unsupported_capability = any(
        key.startswith("incomplete:") or key.startswith("unavailable:")
        for key in getattr(certificate, "failure_counts", {}) or {}
    )
    has_witness = certificate.witness_digest is not None
    payload_mismatch = False
    if certificate.verdict is SupportVerdict.SUPPORTED and not has_witness:
        payload_mismatch = True
    if certificate.verdict is SupportVerdict.UNSUPPORTED and has_witness:
        payload_mismatch = True
    return ClassificationInputsV1(
        payload_mismatch=payload_mismatch,
        timed_out=timed_out,
        skipped_replay=skipped_replay,
        unsupported_capability=unsupported_capability,
        incomplete_coverage=incomplete,
        missing_tool=missing_tool,
        vss_verdict=certificate.verdict.value,
        has_witness_digest=has_witness,
        has_counterexample_digest=False,
        exhausted=bool(certificate.exhausted),
        replay_checked=replay_checked,
        replay_ok=replay_ok,
    )


def from_support_certificate(
    certificate: Any,
    *,
    replay_checked: bool,
    replay_ok: bool,
    checked: bool,
    skipped_replay: bool = False,
    missing_tool: bool = False,
) -> SolverJudgmentV1:
    """Lossless adapter from a VSS support certificate."""

    inputs = _certificate_signal_flags(
        certificate,
        replay_checked=replay_checked,
        replay_ok=replay_ok,
        skipped_replay=skipped_replay,
        missing_tool=missing_tool,
    )
    judgment = classify(inputs, checked=checked)
    if judgment.outcome is JudgmentOutcome.WITNESSED and isinstance(
        judgment.payload, WitnessPayloadV1
    ):
        return SolverJudgmentV1(
            outcome=judgment.outcome,
            payload=WitnessPayloadV1(
                witness_digest=str(certificate.witness_digest),
                source="vss_certificate",
            ),
            checked=checked,
        )
    if judgment.outcome is JudgmentOutcome.REFUTED and isinstance(
        judgment.payload, RefutationPayloadV1
    ):
        return SolverJudgmentV1(
            outcome=judgment.outcome,
            payload=RefutationPayloadV1(
                refutation_digest=certificate.digest,
                source="vss_exhaustion",
            ),
            checked=checked,
        )
    return judgment


def from_exact_closure_query(
    *,
    verdict: VssVerdictName,
    replay_ok: bool,
    replay_checked: bool,
    checked: bool,
    exhausted: bool = False,
    timed_out: bool = False,
    skipped_replay: bool = False,
) -> SolverJudgmentV1:
    """Adapter from exact-closure query results."""

    inputs = ClassificationInputsV1(
        timed_out=timed_out,
        skipped_replay=skipped_replay,
        vss_verdict=verdict,
        has_witness_digest=verdict == "supported",
        exhausted=exhausted or verdict == "unsupported",
        replay_checked=replay_checked,
        replay_ok=replay_ok,
    )
    return classify(inputs, checked=checked)


def from_formal_preflight_status(
    status: str,
    *,
    counterexample: dict[str, Any] | None,
    checked: bool,
) -> SolverJudgmentV1:
    """Historical adapter from theorem-preflight lifecycle status.

    Preflight status remains authoritative for campaign lifecycle; this
    produces the parallel solver judgment view without conflating enums.
    """

    if status in {"timed_out", "unknown", "conditional"}:
        reason = UnknownReason.TIMEOUT if status == "timed_out" else UnknownReason.INCONCLUSIVE
        return SolverJudgmentV1(
            outcome=JudgmentOutcome.UNKNOWN,
            payload=reason,
            checked=checked,
        )
    if status == "refuted":
        if counterexample is None:
            return SolverJudgmentV1(
                outcome=JudgmentOutcome.INVALID,
                payload=InvalidReason.PAYLOAD_MISMATCH,
                checked=checked,
            )
        digest = str(counterexample.get("digest", "checked"))
        return SolverJudgmentV1(
            outcome=JudgmentOutcome.REFUTED,
            payload=RefutationPayloadV1(
                refutation_digest=digest,
                source="counterexample",
            ),
            checked=checked,
        )
    if status == "proved":
        return SolverJudgmentV1(
            outcome=JudgmentOutcome.WITNESSED,
            payload=WitnessPayloadV1(
                witness_digest="preflight_proof",
                source="formal_preflight",
            ),
            checked=checked,
        )
    return SolverJudgmentV1(
        outcome=JudgmentOutcome.INVALID,
        payload=InvalidReason.MALFORMED_INPUT,
        checked=checked,
    )


def evaluate_truth_case(raw: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one frozen truth-table row."""

    inputs_raw = raw["inputs"]
    inputs = ClassificationInputsV1(
        malformed_input=bool(inputs_raw.get("malformed_input", False)),
        payload_mismatch=bool(inputs_raw.get("payload_mismatch", False)),
        timed_out=bool(inputs_raw.get("timed_out", False)),
        skipped_replay=bool(inputs_raw.get("skipped_replay", False)),
        unsupported_capability=bool(inputs_raw.get("unsupported_capability", False)),
        incomplete_coverage=bool(inputs_raw.get("incomplete_coverage", False)),
        missing_tool=bool(inputs_raw.get("missing_tool", False)),
        vss_verdict=inputs_raw.get("vss_verdict"),
        has_witness_digest=bool(inputs_raw.get("has_witness_digest", False)),
        has_counterexample_digest=bool(
            inputs_raw.get("has_counterexample_digest", False)
        ),
        exhausted=bool(inputs_raw.get("exhausted", False)),
        replay_checked=bool(inputs_raw.get("replay_checked", False)),
        replay_ok=bool(inputs_raw.get("replay_ok", False)),
    )
    checked = bool(raw.get("checked", False))
    judgment = classify(inputs, checked=checked)
    return {
        "outcome": judgment.outcome.value,
        "payload_kind": (
            "witness"
            if isinstance(judgment.payload, WitnessPayloadV1)
            else "refutation"
            if isinstance(judgment.payload, RefutationPayloadV1)
            else "unknown_reason"
            if isinstance(judgment.payload, UnknownReason)
            else "invalid_reason"
        ),
        "unknown_reason": (
            judgment.payload.value
            if isinstance(judgment.payload, UnknownReason)
            else None
        ),
        "invalid_reason": (
            judgment.payload.value
            if isinstance(judgment.payload, InvalidReason)
            else None
        ),
        "authorizes_semantic": judgment.authorizes_semantic_conclusion(),
        "authorizes_removal": judgment.authorizes_removal(),
    }


__all__ = [
    "JUDGMENT_SCHEMA",
    "ClassificationInputsV1",
    "InvalidReason",
    "InvalidReasonName",
    "JudgmentOutcome",
    "JudgmentOutcomeName",
    "RefutationPayloadV1",
    "SolverJudgmentV1",
    "UnknownReason",
    "UnknownReasonName",
    "VssVerdictName",
    "WitnessPayloadV1",
    "classify",
    "classify_outcome",
    "classify_payload",
    "evaluate_truth_case",
    "from_exact_closure_query",
    "from_formal_preflight_status",
    "from_support_certificate",
]
