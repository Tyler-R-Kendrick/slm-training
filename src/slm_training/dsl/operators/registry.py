"""Pack-owned immutable operator registry and pure execution boundary (DSH3-02)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, TypeAlias

from slm_training.dsl.operators.contracts import (
    ActionEffectV1,
    ApplicationProofV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BoundArgumentV1,
    OperatorApplicationV1,
    OperatorRejectionV1,
    _fingerprint,
)
from slm_training.dsl.pack import DslPack


class OperatorAuthorityError(ValueError):
    """A state or rewrite failed the owning pack's ordinary authority checks."""


class OperatorReplayError(ValueError):
    """Recorded application evidence did not reproduce exactly."""


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _oracle_passed(report: object) -> bool:
    if isinstance(report, bool):
        return report
    verdict = getattr(report, "ok", None)
    if isinstance(verdict, bool):
        return verdict
    raise OperatorAuthorityError("pack oracle returned no boolean or .ok verdict")


@dataclass(frozen=True)
class OperatorStateV1:
    """Immutable canonical source plus pack-derived state/AST identities."""

    pack_id: str
    source: str
    state_digest: str
    ast_digest: str
    schema: str = "operator_state/v1"

    @classmethod
    def from_source(cls, pack: DslPack, source: str) -> OperatorStateV1:
        canonical, serialized = validate_with_pack_authority(pack, source)
        return cls(
            pack_id=pack.pack_id,
            source=canonical,
            state_digest=_fingerprint(
                {
                    "schema": "operator_state/v1",
                    "pack_id": pack.pack_id,
                    "source": canonical,
                }
            ),
            ast_digest=_digest_text(serialized),
        )


@dataclass(frozen=True)
class OperatorMutationV1:
    """Pure executor output before pack authority validates the new source."""

    source: str
    effect: ActionEffectV1


OperatorExecutor: TypeAlias = Callable[
    [OperatorStateV1, tuple[BoundArgumentV1, ...]], OperatorMutationV1
]


@dataclass(frozen=True)
class RegisteredOperatorV1:
    declaration: AstOperatorV1
    execute: OperatorExecutor


@dataclass(frozen=True)
class OperatorApplyResultV1:
    state: OperatorStateV1 | None
    application: OperatorApplicationV1

    @property
    def succeeded(self) -> bool:
        return self.state is not None and self.application.succeeded


def validate_with_pack_authority(pack: DslPack, source: str) -> tuple[str, str]:
    """Run every ordinary pack authority required for operator-produced source."""
    canonicalize = pack.require("canonicalize")
    oracle = pack.require("oracle")
    scope_extractor = pack.require("scope_extractor")
    prop_order = pack.require("prop_order")

    try:
        parsed = pack.backend.parse(source)
        serialized = pack.backend.serialize(parsed)
        canonical = canonicalize(source)
        canonical_parsed = pack.backend.parse(canonical)
        canonical_serialized = pack.backend.serialize(canonical_parsed)
        roundtrip = canonicalize(canonical_serialized)
        if roundtrip != canonical:
            raise OperatorAuthorityError(
                "pack canonicalization/parse/serialize round-trip drifted"
            )
        if not _oracle_passed(oracle(canonical)):
            raise OperatorAuthorityError("pack static/schema oracle rejected source")
        scope_extractor(canonical)
        order = prop_order()
        if not isinstance(order, Mapping):
            raise OperatorAuthorityError("pack property-order provider is invalid")
    except OperatorAuthorityError:
        raise
    except Exception as exc:  # noqa: BLE001 - pack failures become typed rejection
        raise OperatorAuthorityError(str(exc)) from exc
    return canonical, canonical_serialized or serialized


@dataclass(frozen=True, init=False)
class OperatorLibraryV1:
    """Immutable registry. All execution routes through the owning ``DslPack``."""

    schema = "operator_library/v1"
    _by_id: Mapping[str, RegisteredOperatorV1]
    _by_fingerprint: Mapping[str, RegisteredOperatorV1]
    registry_fingerprint: str

    def __init__(self, entries: tuple[RegisteredOperatorV1, ...]) -> None:
        by_id: dict[str, RegisteredOperatorV1] = {}
        by_fingerprint: dict[str, RegisteredOperatorV1] = {}
        for entry in entries:
            operator_id = entry.declaration.operator_id
            fingerprint = entry.declaration.fingerprint
            if operator_id in by_id:
                raise ValueError(f"duplicate operator id {operator_id!r}")
            if fingerprint in by_fingerprint:
                raise ValueError(f"duplicate operator fingerprint {fingerprint!r}")
            by_id[operator_id] = entry
            by_fingerprint[fingerprint] = entry
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "_by_fingerprint", MappingProxyType(by_fingerprint))
        object.__setattr__(
            self,
            "registry_fingerprint",
            _fingerprint(
                {
                    "schema": self.schema,
                    "declarations": [
                        entry.declaration.to_dict()
                        for entry in sorted(
                            entries, key=lambda item: item.declaration.operator_id
                        )
                    ],
                }
            ),
        )

    @property
    def declarations(self) -> tuple[AstOperatorV1, ...]:
        return tuple(
            entry.declaration
            for entry in sorted(
                self._by_id.values(),
                key=lambda item: item.declaration.operator_id,
            )
        )

    def lookup(self, operator_id: str) -> AstOperatorV1:
        try:
            return self._by_id[operator_id].declaration
        except KeyError as exc:
            raise KeyError(f"unsupported operator {operator_id!r}") from exc

    def dry_run(
        self,
        pack: DslPack,
        state: OperatorStateV1,
        operator_id: str,
        arguments: tuple[BoundArgumentV1, ...],
        provenance: ApplicationProvenanceV1,
    ) -> OperatorApplicationV1:
        return self._execute(
            pack, state, operator_id, arguments, provenance
        ).application

    def apply(
        self,
        pack: DslPack,
        state: OperatorStateV1,
        operator_id: str,
        arguments: tuple[BoundArgumentV1, ...],
        provenance: ApplicationProvenanceV1,
    ) -> OperatorApplyResultV1:
        return self._execute(pack, state, operator_id, arguments, provenance)

    def replay(
        self,
        pack: DslPack,
        state: OperatorStateV1,
        recorded: OperatorApplicationV1,
    ) -> OperatorApplyResultV1:
        if state.state_digest != recorded.before_state_digest:
            raise OperatorReplayError("before state digest does not match replay state")
        if state.ast_digest != recorded.before_ast_digest:
            raise OperatorReplayError("before AST digest does not match replay state")
        try:
            entry = self._by_fingerprint[recorded.operator_fingerprint]
        except KeyError as exc:
            raise OperatorReplayError(
                "recorded operator fingerprint is unavailable"
            ) from exc
        replayed = self.apply(
            pack,
            state,
            entry.declaration.operator_id,
            recorded.arguments,
            recorded.provenance,
        )
        if replayed.application.application_id != recorded.application_id:
            raise OperatorReplayError(
                "replayed application identity differs from recorded evidence"
            )
        return replayed

    def _execute(
        self,
        pack: DslPack,
        state: OperatorStateV1,
        operator_id: str,
        arguments: tuple[BoundArgumentV1, ...],
        provenance: ApplicationProvenanceV1,
    ) -> OperatorApplyResultV1:
        if pack.require("operator_library") is not self:
            raise OperatorAuthorityError("operator library is not owned by this pack")
        if state.pack_id != pack.pack_id:
            raise OperatorAuthorityError("operator state belongs to another pack")
        if provenance.pack_id != pack.pack_id:
            raise OperatorAuthorityError("application provenance names another pack")
        verified_before = OperatorStateV1.from_source(pack, state.source)
        if verified_before != state:
            raise OperatorAuthorityError(
                "operator state identity does not match pack-authorized source"
            )
        before_source = state.source
        entry = self._by_id.get(operator_id)
        if entry is None:
            return self._reject(
                state,
                _fingerprint({"unsupported_operator_id": operator_id}),
                arguments,
                provenance,
                "operator.unsupported",
            )
        try:
            ordered_arguments = entry.declaration.validate_arguments(arguments)
            mutation = entry.execute(state, ordered_arguments)
            if state.source != before_source:
                raise OperatorAuthorityError("operator mutated its input state")
            after_state = OperatorStateV1.from_source(pack, mutation.source)
            proof = ApplicationProofV1(
                proof_kind="pack.authority",
                checks=(
                    "pack.parse",
                    "pack.static_schema",
                    "pack.scope",
                    "pack.property_order",
                    "pack.canonical",
                    "pack.roundtrip",
                    "input.immutable",
                ),
                compiler_result_digest=_fingerprint(
                    {
                        "registry_fingerprint": self.registry_fingerprint,
                        "before_state_digest": state.state_digest,
                        "after_state_digest": after_state.state_digest,
                        "effect_fingerprint": mutation.effect.fingerprint,
                    }
                ),
                effect_fingerprint=mutation.effect.fingerprint,
            )
            application = OperatorApplicationV1(
                operator_fingerprint=entry.declaration.fingerprint,
                arguments=ordered_arguments,
                before_state_digest=state.state_digest,
                before_ast_digest=state.ast_digest,
                provenance=provenance,
                effect=mutation.effect,
                after_state_digest=after_state.state_digest,
                after_ast_digest=after_state.ast_digest,
                proof=proof,
            )
            return OperatorApplyResultV1(after_state, application)
        except Exception as exc:  # noqa: BLE001 - every failure is typed/replayable
            if state.source != before_source:
                raise OperatorAuthorityError(
                    "operator mutated its input state"
                ) from exc
            if isinstance(exc, OperatorAuthorityError):
                rejection_code = "operator.authority_rejected"
            elif isinstance(exc, ValueError):
                rejection_code = "operator.arguments_rejected"
            else:
                rejection_code = "operator.executor_rejected"
            return self._reject(
                state,
                entry.declaration.fingerprint,
                arguments,
                provenance,
                rejection_code,
            )

    @staticmethod
    def _reject(
        state: OperatorStateV1,
        operator_fingerprint: str,
        arguments: tuple[BoundArgumentV1, ...],
        provenance: ApplicationProvenanceV1,
        code: str,
    ) -> OperatorApplyResultV1:
        rejection = OperatorRejectionV1(
            code=code,
            failed_precondition=None,
            compiler_result_digest=_fingerprint(
                {"code": code, "state": state.state_digest}
            ),
        )
        application = OperatorApplicationV1(
            operator_fingerprint=operator_fingerprint,
            arguments=arguments,
            before_state_digest=state.state_digest,
            before_ast_digest=state.ast_digest,
            provenance=provenance,
            rejection=rejection,
        )
        return OperatorApplyResultV1(None, application)
