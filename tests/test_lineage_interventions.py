"""LDI4-03 unified intervention manifest / registry / promotion gates (SLM-137).

Torch-free: the whole consolidation core (schema, registry validation, evaluation-bundle
eligibility, promotion state machine, one-active + cyclic-lineage rules) is exercised
without loading a model.
"""

from __future__ import annotations

import pytest

from slm_training.harnesses.model_build.checkpoint_reference import FileArtifact
from slm_training.lineage.interventions import (
    INTERVENTION_KINDS,
    BaseIdentity,
    EvaluationBundle,
    InterventionError,
    InterventionManifest,
    InterventionRegistry,
    assert_single_active,
    build_closeout_index,
    detect_lineage_cycle,
    promote,
)


def _base(**kw) -> BaseIdentity:
    d = dict(
        architecture="twotower",
        base_model_id="parent-x",
        base_model_revision="rev1",
        tokenizer_sha="tok1",
        base_compatibility_fingerprint="compat-abc",
    )
    d.update(kw)
    return BaseIdentity(**d)


def _manifest(kind: str = "twotower_delta", *, iid: str = "iv-1", **kw) -> InterventionManifest:
    d = dict(
        intervention_id=iid,
        kind=kind,
        method="low_rank",
        status="wiring",
        deployable=kind != "sae_diagnostic",
        base=_base(),
        module_site_map=(("denoiser.block.3", "residual"),),
        parameter_shapes=(("A", (4, 16)),),
        trainable_parameter_count=64,
        artifact_files=(FileArtifact("adapter_model.pt", 128, "deadbeef"),),
        config_fingerprint="cfg-123",
    )
    d.update(kw)
    return InterventionManifest(**d)


def _bundle(*, complete: bool = True, gates_pass: bool = True) -> EvaluationBundle:
    ident = {"base_sha": "b", "intervention_sha": "i", "corpus_sha": "c", "seed": 0, "commit_sha": "s"}
    end = {"ship_gates": {"pass": gates_pass, "failures": [] if gates_pass else ["parse"]},
           "adversarial": {}, "ood": {}, "agentv": {}}
    if not complete:
        del ident["seed"]
    return EvaluationBundle(
        identity=ident,
        event={"support_summary": {}, "local_objective_metrics": {}},
        locality={"legal_space_drift": {}, "preservation": {}, "disabled_parity": {}},
        end_to_end=end,
    )


# --------------------------------------------------------------------------- #
# Manifest schema / fail-closed
# --------------------------------------------------------------------------- #
def test_manifest_round_trips_and_fingerprint_is_stable():
    m = _manifest()
    again = InterventionManifest.from_dict(m.to_dict())
    assert again.fingerprint() == m.fingerprint()


def test_manifest_fails_closed_on_unknown_kind_field_version():
    with pytest.raises(InterventionError):
        _manifest(kind="bogus_kind")
    with pytest.raises(InterventionError):
        _manifest(status="deployed")  # not a promotion status
    with pytest.raises(InterventionError):
        _manifest(version="v9")
    with pytest.raises(InterventionError):
        InterventionManifest.from_dict({**_manifest().to_dict(), "surprise": 1})


def test_each_kind_validates_through_one_interface():
    reg = InterventionRegistry()
    for kind in INTERVENTION_KINDS:
        m = _manifest(kind, deployable=kind != "sae_diagnostic")
        assert reg.is_valid(m), (kind, reg.validate(m))


def test_diagnostic_only_cannot_be_deployable():
    with pytest.raises(InterventionError):
        _manifest("sae_diagnostic", deployable=True)
    # inspect never reports a diagnostic kind as deployable
    reg = InterventionRegistry()
    assert reg.inspect(_manifest("sae_diagnostic", deployable=False))["deployable"] is False


# --------------------------------------------------------------------------- #
# Integrity / compatibility validation
# --------------------------------------------------------------------------- #
def test_validation_flags_missing_compat_and_uncontent_addressed_files():
    reg = InterventionRegistry()
    no_compat = _manifest(base=_base(base_compatibility_fingerprint=""))
    assert "missing base_compatibility_fingerprint" in reg.validate(no_compat)
    unknown_file = _manifest(artifact_files=(FileArtifact("x.pt", 10, "UNKNOWN"),))
    assert any("not content-addressed" in f for f in reg.validate(unknown_file))
    no_map = _manifest(module_site_map=())
    assert "missing module/site map" in reg.validate(no_map)


# --------------------------------------------------------------------------- #
# One-active + cyclic lineage
# --------------------------------------------------------------------------- #
def test_single_active_intervention_enforced():
    assert_single_active([_manifest(iid="a")])  # ok
    with pytest.raises(InterventionError):
        assert_single_active([_manifest(iid="a"), _manifest(iid="b")])


def test_cyclic_lineage_detected():
    a = _manifest(iid="a", parent_intervention_ids=("b",))
    b = _manifest(iid="b", parent_intervention_ids=("a",))
    assert detect_lineage_cycle([a, b]) is not None
    chain_a = _manifest(iid="a")
    chain_b = _manifest(iid="b", parent_intervention_ids=("a",))
    assert detect_lineage_cycle([chain_a, chain_b]) is None


def test_self_parent_rejected_at_construction():
    with pytest.raises(InterventionError):
        _manifest(iid="a", parent_intervention_ids=("a",))


# --------------------------------------------------------------------------- #
# Evaluation bundle eligibility (missing => ineligible, not pass)
# --------------------------------------------------------------------------- #
def test_missing_required_evidence_makes_ineligible():
    assert _bundle(complete=True, gates_pass=True).eligible() is True
    assert _bundle(complete=False).missing_fields()  # some fields missing
    assert _bundle(complete=False).eligible() is False
    assert _bundle(gates_pass=False).eligible() is False  # complete but gate fails


# --------------------------------------------------------------------------- #
# Promotion state machine (deterministic, fail-closed)
# --------------------------------------------------------------------------- #
def test_promotion_transitions_are_deterministic():
    m = _manifest(status="wiring")
    assert promote(m, "diagnostic").ok is True
    # skipping states is illegal
    assert promote(m, "eligible").ok is False
    assert promote(m, "promoted").ok is False


def test_diagnostic_to_eligible_requires_complete_passing_bundle():
    m = _manifest(status="diagnostic")
    assert promote(m, "eligible", evidence=None).ok is False
    assert promote(m, "eligible", evidence=_bundle(complete=False)).ok is False
    assert promote(m, "eligible", evidence=_bundle(gates_pass=False)).ok is False
    assert promote(m, "eligible", evidence=_bundle()).ok is True


def test_promotion_cannot_override_failed_ship_gate():
    m = _manifest(status="eligible", deployable=True)
    failed = promote(m, "promoted", evidence=_bundle(gates_pass=False))
    assert failed.ok is False
    assert any("ship gate" in f for f in failed.failures)
    assert promote(m, "promoted", evidence=_bundle()).ok is True


def test_diagnostic_sae_cannot_be_promoted():
    m = _manifest("sae_diagnostic", status="eligible", deployable=False)
    assert promote(m, "promoted", evidence=_bundle()).ok is False


def test_run_outcome_is_not_a_promotion_target():
    m = _manifest(status="diagnostic")
    assert promote(m, "no_safe_direction").ok is False
    assert promote(m, "expired").ok is False


# --------------------------------------------------------------------------- #
# Closeout index
# --------------------------------------------------------------------------- #
def test_closeout_index_reports_best_or_none():
    none_yet = build_closeout_index([_manifest(iid="a", status="diagnostic")])
    assert none_yet["best_deployable"] is None
    assert "no intervention currently qualifies" in none_yet["best_deployable_statement"]
    promoted = build_closeout_index([
        _manifest(iid="a", status="diagnostic"),
        _manifest(iid="b", status="promoted"),
    ])
    assert promoted["best_deployable"] == "b"
    assert promoted["lineage_cycle"] is None
