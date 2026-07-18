"""Neural SurfaceRealizer with deterministic fallback for VSS3-05 (SLM-73).

The autoregressive realizer only handles slots already classified as
``SURFACE_ONLY`` by the pack classifier. Unsupported or invalid proposals fall
back to ``DeterministicSurfaceRealizer`` per slot, and the final assembled
program is still globally re-verified by ``realize_surface_and_verify``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from slm_training.dsl.surface import (
    DeterministicSurfaceRealizer,
    SurfaceAssignment,
    SurfaceAuthority,
    SurfaceRealizationRequest,
    SurfaceSlotKind,
)

# torch and the AR model module are imported lazily inside the methods that need
# them (mirroring main's ``models/grammar.py`` / ``models/causal_lm_openui.py``).
# This keeps ``NeuralSurfaceRealizerConfig`` and the no-model deterministic
# fallback path importable and exercisable without torch installed.
if TYPE_CHECKING:
    import torch

    from slm_training.models.surface_autoregressor import (
        DecorativeConstraint,
        IdentifierConstraint,
        SurfaceAutoregressor,
    )


@dataclass(frozen=True)
class NeuralSurfaceRealizerConfig:
    """Runtime knobs for the neural surface realizer."""

    model: SurfaceAutoregressor | None = None
    model_path: str | None = None
    device: str = "cpu"
    max_bytes: int = 64
    temperature: float = 0.0
    top_k: int = 1
    seed: int | None = None
    fallback_to_deterministic: bool = True


class NeuralSurfaceRealizer:
    """SurfaceRealizer backed by a small causal byte-autoregressor.

    Unsupported slot kinds are rejected before the model is invoked. Each slot
    is generated independently under its own constraint mask. If the model
    proposes an invalid value, hits a dead end, or raises an exception, the
    realizer optionally falls back to ``DeterministicSurfaceRealizer`` for that
    single slot and records the reason.
    """

    def __init__(self, config: NeuralSurfaceRealizerConfig) -> None:
        self.config = config
        self._fallback = DeterministicSurfaceRealizer()
        self._model = self._load_model()
        # The vocab lives on the (torch) model; the no-model fallback path never
        # touches it, so leave it ``None`` to stay torch-free without a model.
        self._vocab = self._model.vocab if self._model is not None else None

    def _load_model(self) -> SurfaceAutoregressor | None:
        if self.config.model is not None:
            return self.config.model.to(self.config.device)
        if self.config.model_path is not None:
            from slm_training.models.surface_autoregressor import SurfaceAutoregressor

            return SurfaceAutoregressor.load(self.config.model_path, device=self.config.device)
        # No model: every supported slot will fall back to deterministic.
        return None

    def _build_prompt(self, slot: Any, peers: set[str]) -> str:
        """Encode slot metadata and namespace context as a short prompt."""
        parts = [
            f"kind={slot.kind.value}",
            f"authority={slot.authority.value}",
            f"slot_id={slot.slot_id}",
        ]
        if slot.semantic_symbol_id is not None:
            parts.append(f"symbol={slot.semantic_symbol_id}")
        max_bytes = slot.constraints.max_bytes or self.config.max_bytes
        parts.append(f"max={max_bytes}")
        if slot.constraints.reserved:
            parts.append(f"reserved={','.join(slot.constraints.reserved)}")
        if peers:
            parts.append(f"peers={','.join(sorted(peers))}")
        return " ".join(parts)

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        import torch

        ids = self._vocab.encode(prompt, add_special=True)
        return torch.tensor(ids, dtype=torch.long)

    def _make_constraint(self, slot: Any, peers: set[str]) -> IdentifierConstraint | DecorativeConstraint:
        from slm_training.models.surface_autoregressor import (
            DecorativeConstraint,
            IdentifierConstraint,
        )

        max_bytes = slot.constraints.max_bytes or self.config.max_bytes
        reserved = set(slot.constraints.reserved)
        if slot.kind is SurfaceSlotKind.INTERNAL_IDENTIFIER:
            return IdentifierConstraint(
                self._vocab,
                max_bytes=max_bytes,
                reserved=reserved,
                peers=peers,
            )
        if slot.kind is SurfaceSlotKind.DECORATIVE_TEXT:
            return DecorativeConstraint(self._vocab, max_bytes=max_bytes)
        raise ValueError(f"unsupported slot kind {slot.kind.value}")

    def realize(self, request: SurfaceRealizationRequest) -> tuple[SurfaceAssignment, ...]:
        """Generate surface assignments with per-slot deterministic fallback."""
        assignments: list[SurfaceAssignment] = []
        peers: set[str] = set()

        for slot in request.slots:
            # Reject anything not explicitly approved for AR surface generation.
            if slot.authority is not SurfaceAuthority.SURFACE_ONLY:
                raise ValueError(
                    f"slot {slot.slot_id!r} has authority {slot.authority.value}; "
                    "only SURFACE_ONLY slots may be realized by the AR model"
                )
            if slot.kind not in {
                SurfaceSlotKind.INTERNAL_IDENTIFIER,
                SurfaceSlotKind.DECORATIVE_TEXT,
            }:
                raise ValueError(
                    f"slot {slot.slot_id!r} ({slot.kind.value}) is not supported by "
                    "the autoregressive surface realizer"
                )

            assignment = self._realize_one(slot, peers, request)
            assignments.append(assignment)
            peers.add(assignment.value)

        return tuple(assignments)

    def _realize_one(
        self, slot: Any, peers: set[str], request: SurfaceRealizationRequest
    ) -> SurfaceAssignment:
        """Generate one slot, falling back on invalid/dead-end/model error."""
        fallback_reason: str | None = None
        value: str | None = None
        provenance = "autoregressive"

        if self._model is not None:
            try:
                prompt = self._build_prompt(slot, peers)
                prompt_ids = self._encode_prompt(prompt)
                constraint = self._make_constraint(slot, peers)
                value = self._model.generate(
                    prompt_ids,
                    constraint,
                    max_bytes=slot.constraints.max_bytes or self.config.max_bytes,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    seed=self.config.seed,
                )
                if value is None:
                    fallback_reason = "dead_end"
                elif not constraint.is_complete(value):
                    fallback_reason = "invalid_proposal"
                    value = None
                else:
                    provenance = "autoregressive"
            except Exception as exc:  # noqa: BLE001
                fallback_reason = f"model_error:{type(exc).__name__}"
                value = None
        else:
            fallback_reason = "no_model"

        if value is None:
            if not self.config.fallback_to_deterministic:
                raise ValueError(
                    f"slot {slot.slot_id!r} could not be realized and fallback is disabled"
                )
            # Fall back to deterministic for this single slot.
            sub_request = SurfaceRealizationRequest(
                pack_id=request.pack_id,
                constraint_version=request.constraint_version,
                semantic_ir_fingerprint=request.semantic_ir_fingerprint,
                slots=(slot,),
                context=request.context,
            )
            fb_assignments = self._fallback.realize(sub_request)
            if not fb_assignments:
                raise ValueError(
                    f"slot {slot.slot_id!r}: AR failed ({fallback_reason}) and "
                    "deterministic fallback produced no assignment"
                )
            assignment = fb_assignments[0]
            return SurfaceAssignment(
                slot_id=assignment.slot_id,
                value=assignment.value,
                provenance=f"autoregressive_fallback:{fallback_reason}:{assignment.provenance}",
            )

        return SurfaceAssignment(slot_id=slot.slot_id, value=value, provenance=provenance)


__all__ = ["NeuralSurfaceRealizer", "NeuralSurfaceRealizerConfig"]
