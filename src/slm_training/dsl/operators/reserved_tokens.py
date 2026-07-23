"""Default-off reserved operator targets with compiler-owned application."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from slm_training.dsl.operators.contracts import (
    ApplicationProvenanceV1,
    _fingerprint,
)
from slm_training.dsl.operators.legal_set import (
    LegalOperatorActionV1,
    OperatorLegalSetV1,
    deserialize_operator_action,
)
from slm_training.dsl.operators.registry import (
    OperatorApplyResultV1,
    OperatorLibraryV1,
    OperatorStateV1,
)
from slm_training.dsl.pack import DslPack

RESERVED_OPERATOR_PREFIX = "<|openui_operator:v1|>"
RESERVED_OPERATOR_SUFFIX = "<|end_openui_operator|>"


class ReservedOperatorTargetMode(str, Enum):
    RESULT_AST_ONLY = "result_ast_only"
    OPERATOR_ONLY = "operator_only"
    OPERATOR_PLUS_RESULT = "operator_plus_result"


class ReservedOperatorDisposition(str, Enum):
    APPLY = "apply"
    DEFER = "defer"
    REJECT = "reject"


@dataclass(frozen=True)
class ReservedOperatorTokenConfigV1:
    enabled: bool = False
    format_version: str = "v1"
    schema: str = "reserved_operator_token_config/v1"

    def __post_init__(self) -> None:
        if self.format_version != "v1":
            raise ValueError("unsupported reserved operator token format")

    @property
    def compatibility_fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "enabled": self.enabled,
            "format_version": self.format_version,
        }


@dataclass(frozen=True)
class ReservedOperatorDecisionV1:
    disposition: ReservedOperatorDisposition
    reason: str
    action: LegalOperatorActionV1 | None = None
    result_ast: str | None = None
    schema: str = "reserved_operator_decision/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "disposition": self.disposition.value,
            "reason": self.reason,
            "action": self.action.to_dict() if self.action is not None else None,
            "result_ast": self.result_ast,
        }


def serialize_reserved_operator_target(
    *,
    action: LegalOperatorActionV1 | str,
    result_ast: str,
    mode: ReservedOperatorTargetMode,
    config: ReservedOperatorTokenConfigV1 | None = None,
) -> str:
    """Serialize one versioned target; the feature is disabled by default."""
    active = config or ReservedOperatorTokenConfigV1()
    if not active.enabled:
        raise ValueError("reserved operator tokens are disabled")
    payload: dict[str, Any] = {
        "schema": "reserved_operator_target/v1",
        "format_version": active.format_version,
        "mode": mode.value,
    }
    if mode is not ReservedOperatorTargetMode.RESULT_AST_ONLY:
        serialized = action.serialized if isinstance(action, LegalOperatorActionV1) else action
        deserialize_operator_action(serialized)
        payload["operator"] = serialized
    if mode is not ReservedOperatorTargetMode.OPERATOR_ONLY:
        payload["result_ast"] = result_ast
    return (
        RESERVED_OPERATOR_PREFIX
        + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        + RESERVED_OPERATOR_SUFFIX
    )


def parse_reserved_operator_target(
    value: str,
    *,
    config: ReservedOperatorTokenConfigV1 | None = None,
) -> dict[str, Any]:
    active = config or ReservedOperatorTokenConfigV1()
    if not active.enabled:
        raise ValueError("reserved operator tokens are disabled")
    if not value.startswith(RESERVED_OPERATOR_PREFIX) or not value.endswith(
        RESERVED_OPERATOR_SUFFIX
    ):
        raise ValueError("reserved operator target framing is invalid")
    encoded = value[
        len(RESERVED_OPERATOR_PREFIX) : -len(RESERVED_OPERATOR_SUFFIX)
    ]
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError("reserved operator target JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("reserved operator target must be an object")
    if payload.get("schema") != "reserved_operator_target/v1":
        raise ValueError("reserved operator target schema is incompatible")
    if payload.get("format_version") != active.format_version:
        raise ValueError("reserved operator target version is incompatible")
    try:
        mode = ReservedOperatorTargetMode(payload["mode"])
    except (KeyError, ValueError) as exc:
        raise ValueError("reserved operator target mode is invalid") from exc
    expected = {"schema", "format_version", "mode"}
    if mode is not ReservedOperatorTargetMode.RESULT_AST_ONLY:
        expected.add("operator")
    if mode is not ReservedOperatorTargetMode.OPERATOR_ONLY:
        expected.add("result_ast")
    if set(payload) != expected:
        raise ValueError("reserved operator target fields are not closed")
    if "operator" in payload:
        deserialize_operator_action(str(payload["operator"]))
    if "result_ast" in payload and not isinstance(payload["result_ast"], str):
        raise ValueError("reserved operator result AST must be text")
    return payload


def _match_action(
    payload: dict[str, Any], legal_set: OperatorLegalSetV1
) -> LegalOperatorActionV1 | None:
    if "operator" not in payload:
        return None
    operator_id, arguments = deserialize_operator_action(payload["operator"])
    return next(
        (
            action
            for action in legal_set.operator_actions
            if action.operator_id == operator_id and action.arguments == arguments
        ),
        None,
    )


def apply_reserved_operator_target(
    *,
    value: str,
    config: ReservedOperatorTokenConfigV1 | None,
    pack: DslPack,
    library: OperatorLibraryV1,
    state: OperatorStateV1,
    legal_set: OperatorLegalSetV1,
    provenance: ApplicationProvenanceV1,
) -> tuple[ReservedOperatorDecisionV1, OperatorApplyResultV1 | None]:
    """Parse, prove exact live membership, and apply only through the compiler."""
    active = config or ReservedOperatorTokenConfigV1()
    if not active.enabled:
        return (
            ReservedOperatorDecisionV1(
                ReservedOperatorDisposition.DEFER,
                "operator.reserved_tokens_disabled",
            ),
            None,
        )
    try:
        payload = parse_reserved_operator_target(value, config=active)
    except ValueError:
        return (
            ReservedOperatorDecisionV1(
                ReservedOperatorDisposition.DEFER,
                "operator.reserved_target_parse_failed",
            ),
            None,
        )
    if "operator" not in payload:
        return (
            ReservedOperatorDecisionV1(
                ReservedOperatorDisposition.DEFER,
                "operator.result_only_ordinary_path",
                result_ast=payload.get("result_ast"),
            ),
            None,
        )
    expected_state = _fingerprint(
        {
            "schema": "operator_legal_state/v1",
            "state_digest": state.state_digest,
            "ast_digest": state.ast_digest,
        }
    )
    if (
        legal_set.state_fingerprint != expected_state
        or legal_set.registry_fingerprint != library.registry_fingerprint
    ):
        return (
            ReservedOperatorDecisionV1(
                ReservedOperatorDisposition.REJECT,
                "operator.stale_legal_set",
            ),
            None,
        )
    action = _match_action(payload, legal_set)
    if action is None:
        return (
            ReservedOperatorDecisionV1(
                ReservedOperatorDisposition.REJECT,
                "operator.not_in_live_legal_set",
            ),
            None,
        )
    result = library.apply(
        pack,
        state,
        action.operator_id,
        action.arguments,
        provenance,
    )
    if not result.succeeded or result.state is None:
        return (
            ReservedOperatorDecisionV1(
                ReservedOperatorDisposition.REJECT,
                "operator.compiler_rejected_admitted_action",
                action=action,
            ),
            result,
        )
    result_ast = payload.get("result_ast")
    if result_ast is not None and result_ast != result.state.source:
        return (
            ReservedOperatorDecisionV1(
                ReservedOperatorDisposition.REJECT,
                "operator.result_ast_mismatch",
                action=action,
                result_ast=result.state.source,
            ),
            result,
        )
    return (
        ReservedOperatorDecisionV1(
            ReservedOperatorDisposition.APPLY,
            "operator.compiler_applied",
            action=action,
            result_ast=result.state.source,
        ),
        result,
    )


def reserved_operator_checkpoint_metadata(
    config: ReservedOperatorTokenConfigV1 | None = None,
) -> dict[str, Any]:
    active = config or ReservedOperatorTokenConfigV1()
    return {
        "schema": "reserved_operator_checkpoint_metadata/v1",
        "config": active.to_dict(),
        "compatibility_fingerprint": active.compatibility_fingerprint,
    }


def validate_reserved_operator_checkpoint(
    metadata: dict[str, Any] | None,
    config: ReservedOperatorTokenConfigV1 | None = None,
) -> None:
    """Fail closed when checkpoint and runtime token contracts differ."""
    active = config or ReservedOperatorTokenConfigV1()
    if metadata is None:
        if active.enabled:
            raise ValueError("checkpoint lacks reserved operator token metadata")
        return
    if metadata.get("schema") != "reserved_operator_checkpoint_metadata/v1":
        raise ValueError("checkpoint reserved operator metadata schema mismatch")
    if metadata.get("compatibility_fingerprint") != active.compatibility_fingerprint:
        raise ValueError("checkpoint reserved operator token config mismatch")


__all__ = [
    "RESERVED_OPERATOR_PREFIX",
    "RESERVED_OPERATOR_SUFFIX",
    "ReservedOperatorDecisionV1",
    "ReservedOperatorDisposition",
    "ReservedOperatorTargetMode",
    "ReservedOperatorTokenConfigV1",
    "apply_reserved_operator_target",
    "parse_reserved_operator_target",
    "reserved_operator_checkpoint_metadata",
    "serialize_reserved_operator_target",
    "validate_reserved_operator_checkpoint",
]
