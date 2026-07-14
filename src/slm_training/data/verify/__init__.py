"""Layered OpenUI record verification and confidence tiers."""

from slm_training.data.verify.runtime import RuntimeEvidence, run_preview_verifier
from slm_training.data.verify.stack import (
    Gate,
    GateResult,
    GateStatus,
    Tier,
    VerificationContext,
    VerificationReport,
    evaluate_gate,
    stamp_record,
    verify_record,
)

__all__ = [
    "Gate",
    "GateResult",
    "GateStatus",
    "RuntimeEvidence",
    "Tier",
    "VerificationContext",
    "VerificationReport",
    "evaluate_gate",
    "run_preview_verifier",
    "stamp_record",
    "verify_record",
]
