"""AP-035 (SLM-336) admission gate: one-step MeanFlow-style prior + discrete refinement.

AP-035's own implementation contract says: "Start only after AP-034 clears
its oracle-support gate." AP-034 (SLM-335)'s disposition
(``docs/design/ap034-conditional-latent-prior-disposition.md``) recorded an
admission-denial: no conditional-prior model, connector, or training path
was added, and it named three missing successor artifacts. This module
enforces that precondition in code so a future AP-035 implementation cannot
silently start, or report a Pareto-frontier comparison, before those
artifacts exist.

See ``docs/design/one-step-latent-hybrid-disposition.md`` for the disposition
this gate implements.
"""

from __future__ import annotations

from dataclasses import dataclass


class Ap034GateNotClearedError(RuntimeError):
    """Raised when AP-035 work is attempted before AP-034's gate clears."""


@dataclass(frozen=True)
class Ap034PriorArtifacts:
    """The three successor artifacts AP-034's disposition named as required
    before AP-035 may build the conditional prior it depends on."""

    prompt_to_plan_manifest_exists: bool
    latent_to_program_decoder_exists: bool
    matched_prior_protocol_exists: bool

    @property
    def all_present(self) -> bool:
        return (
            self.prompt_to_plan_manifest_exists
            and self.latent_to_program_decoder_exists
            and self.matched_prior_protocol_exists
        )


def require_ap034_gate_cleared(artifacts: Ap034PriorArtifacts) -> None:
    """Raise :class:`Ap034GateNotClearedError` unless every AP-034 successor
    artifact is present.
    """
    if artifacts.all_present:
        return
    missing = []
    if not artifacts.prompt_to_plan_manifest_exists:
        missing.append("immutable prompt-to-SemanticPlanV1 manifest")
    if not artifacts.latent_to_program_decoder_exists:
        missing.append("verified latent-to-program decode bridge")
    if not artifacts.matched_prior_protocol_exists:
        missing.append(
            "matched oracle/nearest-encoder/random/conditional-prior protocol"
        )
    raise Ap034GateNotClearedError(
        "AP-035 (SLM-336) cannot start: AP-034's oracle-support gate is not "
        "cleared. Missing successor artifacts: " + ", ".join(missing)
    )


# Repository state as of this disposition (2026-07-26): AP-034 built none of
# its named successor artifacts. Update this literal, with evidence, once an
# artifact lands -- do not weaken `require_ap034_gate_cleared` to route
# around it.
CURRENT_AP034_ARTIFACTS = Ap034PriorArtifacts(
    prompt_to_plan_manifest_exists=False,
    latent_to_program_decoder_exists=False,
    matched_prior_protocol_exists=False,
)
