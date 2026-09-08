"""Existing preflight discovery adapter for locked, fully resolved arm designs."""

from slm_training.autoresearch.experiment_identity import InterventionContract, treatment_identity
from slm_training.autoresearch.intervention_matching import assert_intervention_match
from slm_training.autoresearch.preflight.lever_effects import validate_compiled_effects, validate_lever_effects
from slm_training.autoresearch.preflight import PreflightVerdict


class TreatmentDesignCheck:
    check_id = "treatment_design"

    def run(self, candidate: dict) -> PreflightVerdict:
        design = candidate.get("treatment_design")
        if design is None:
            # Legacy callers have no resolved inputs: don't invent strong evidence.
            return PreflightVerdict(check_id=self.check_id, verdict="warn",
                                    reasons=["legacy_unresolved_design"],
                                    data={"release_authorized": False})
        try:
            contract = InterventionContract.model_validate(design["intervention"])
            assert_intervention_match(
                design["control"], design["candidate"], contract=contract,
                resource_totals=design["resource_totals"], trainable_params=design["trainable_params"],
            )
            validate_lever_effects(design["control"], design["candidate"],
                                  varied_fields=contract.varied_fields, endpoint=design["endpoint"])
            validate_compiled_effects(design["compiled_commands"], design["candidate"], contract.varied_fields)
            identity = treatment_identity(design["candidate"], bindings=design["bindings"],
                                          intervention=contract)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return PreflightVerdict(check_id=self.check_id, verdict="block", reasons=[str(exc)])
        return PreflightVerdict(check_id=self.check_id, verdict="pass", reasons=["valid_typed_design"],
                                data={"treatment_id": identity, "scientific_improvement": False})


CHECK = TreatmentDesignCheck()

