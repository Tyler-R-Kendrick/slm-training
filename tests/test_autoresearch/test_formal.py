from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
)
from slm_training.autoresearch.formal import (
    bind_preflight,
    check_formal_trace,
    formal_trace_from_closure,
    run_formal_preflight,
    validate_formal_preflights,
)
from slm_training.autoresearch.schemas import (
    CampaignBudget,
    ExperimentKnobs,
    ExperimentSpec,
    FormalClaimV1,
    FormalObligationV1,
    FormalTraceStepV1,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.model_build.eval_runner import structural_similarity
from slm_training.levers import MAX_RUN_MINUTES


def _experiment(claim: FormalClaimV1) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="formal-exp",
        campaign_id="formal-campaign",
        hypothesis="A structural change preserves a declared invariant.",
        rationale="The formal preflight should reject invalid structural claims.",
        expected_effect="A smaller empirical search space.",
        falsification_criteria=("The theorem or counterexample rejects the claim.",),
        stop_conditions=("Stop before training when the required proof fails.",),
        citations=("docs/design/formal-autoresearch.md",),
        knobs=ExperimentKnobs(steps=1),
        formal_claims=(claim,),
    )


def _manifest(
    obligation: FormalObligationV1,
) -> ExperimentCampaignV1:
    return ExperimentCampaignV1(
        campaign_id="formal-campaign",
        experiment_id="formal-exp",
        hypothesis="A structural change preserves a declared invariant.",
        decision="Run only after required formal obligations pass.",
        endpoints=(
            CampaignEndpointV1(
                endpoint_id="primary",
                metric="structural_similarity",
                role="primary",
                direction="increase",
                minimum_effect=0.0,
            ),
        ),
        arms=(
            CampaignArmV1(
                arm_id="control", role="control", config_sha256="a" * 64
            ),
            CampaignArmV1(
                arm_id="candidate", role="candidate", config_sha256="b" * 64
            ),
        ),
        seeds=(7,),
        budget=CampaignBudget(
            max_experiments=1,
            max_wall_minutes=MAX_RUN_MINUTES,
        ),
        stopping_rules=("Stop after the declared seed.",),
        controls=(
            CampaignControlV1(
                control_id="matched",
                description="Size-matched unchanged baseline.",
                kind="negative",
            ),
        ),
        negative_controls=("matched",),
        multiplicity_families=(
            MultiplicityFamilyV1(
                family_id="primary-family",
                hypothesis_ids=("primary",),
                alpha=0.05,
            ),
        ),
        promotion_gates=(
            CampaignGateV1(
                gate_id="promote",
                endpoint_id="primary",
                operator="ge",
                threshold=0.0,
            ),
        ),
        rollback_gates=(
            CampaignGateV1(
                gate_id="rollback",
                endpoint_id="primary",
                operator="lt",
                threshold=0.0,
            ),
        ),
        artifact_requirements=(
            ArtifactRequirementV1(kind="version_stamp"),
            ArtifactRequirementV1(kind="formal_preflight"),
        ),
        formal_obligations=(obligation,),
        claim_class="diagnostic",
        source_commit="c" * 40,
        source_dirty=False,
        author="test",
    )


def _successful_lean(
    command: list[str], *, timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    del timeout_seconds
    stdout = "Lean (version 4.30.0)" if "--version" in command else "build passed"
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def test_trace_checker_matches_history_and_adjacency_contract() -> None:
    first = FormalTraceStepV1(
        before_removed=(),
        after_removed=(0,),
        certified=(0,),
        before_history=(),
        after_history=("remove-a",),
    )
    second = FormalTraceStepV1(
        before_removed=(0,),
        after_removed=(0, 1),
        certified=(1,),
        before_history=("remove-a",),
        after_history=("remove-a", "remove-b"),
    )
    assert check_formal_trace((first, second))
    assert not check_formal_trace(
        (first, second.model_copy(update={"before_history": ()}))
    )


def test_metric_examples_follow_the_lean_monotonicity_and_counterexample() -> None:
    gold = "Page(Column(Text()))"

    assert structural_similarity("Page()", gold) <= structural_similarity(
        "Page(Column())", gold
    )
    assert structural_similarity("Page(Column())", gold) <= structural_similarity(
        gold, gold
    )
    assert structural_similarity("Page(Column(Text(), Button()))", gold) < 1.0


def test_closure_deductions_project_to_valid_stable_ordinals() -> None:
    class Encoded:
        def __init__(self, value: str) -> None:
            self.value = value

        def to_dict(self) -> dict[str, str]:
            return {"value": self.value}

    result = SimpleNamespace(
        deductions=(
            SimpleNamespace(
                hole_id=Encoded("root"),
                removed=(Encoded("left"),),
                after_fingerprint="state-1",
            ),
            SimpleNamespace(
                hole_id=Encoded("root"),
                removed=(Encoded("right"),),
                after_fingerprint="state-2",
            ),
        )
    )

    trace = formal_trace_from_closure(result)

    assert check_formal_trace(trace)
    assert trace[0].certified == (0,)
    assert trace[1].before_removed == (0,)
    assert trace[1].certified == (1,)


def test_required_proof_is_bound_and_validated(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "slm_training.autoresearch.formal._run",
        _successful_lean,
    )
    claim = FormalClaimV1(
        template_id="metrics.structural_similarity_monotone",
        claim="Nondecreasing jaccard and depth components cannot reduce the proxy.",
        policy="required",
    )
    experiment = _experiment(claim)
    preflight, obligation = run_formal_preflight(
        "formal-campaign", experiment, claim
    )
    store = CampaignStore("formal-campaign", tmp_path)
    path = store.write_artifact("formal_preflights", preflight)
    manifest = _manifest(bind_preflight(obligation, path.stem))

    assert validate_formal_preflights(store.root, experiment, manifest) == (preflight,)


def test_required_conditional_claim_blocks_execution_gate(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "slm_training.autoresearch.formal._run",
        _successful_lean,
    )
    claim = FormalClaimV1(
        template_id="recurrence.layerscale_stability",
        claim="LayerScale makes every trained recursive transition globally stable.",
        policy="required",
    )
    experiment = _experiment(claim)
    preflight, obligation = run_formal_preflight(
        "formal-campaign", experiment, claim
    )
    store = CampaignStore("formal-campaign", tmp_path)
    path = store.write_artifact("formal_preflights", preflight)
    manifest = _manifest(bind_preflight(obligation, path.stem))

    assert preflight.status == "conditional"
    with pytest.raises(ValueError, match="required formal claim is not proved"):
        validate_formal_preflights(store.root, experiment, manifest)


def test_formal_obligation_requires_artifact_requirement() -> None:
    obligation = FormalObligationV1(
        obligation_id="formal-" + "a" * 16,
        template_id="forest.history_preservation",
        policy="required",
        preflight_sha256="b" * 64,
    )
    payload = _manifest(obligation).model_dump(mode="json")
    payload["artifact_requirements"] = [{"kind": "version_stamp"}]

    with pytest.raises(ValueError, match="formal_preflight artifact requirement"):
        ExperimentCampaignV1.model_validate(payload)


def test_source_drift_invalidates_bound_preflight(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "slm_training.autoresearch.formal._run",
        _successful_lean,
    )
    claim = FormalClaimV1(
        template_id="forest.history_preservation",
        claim="Every accepted certified closure trace preserves append-only history.",
        policy="required",
    )
    experiment = _experiment(claim)
    preflight, obligation = run_formal_preflight(
        "formal-campaign", experiment, claim
    )
    first_source = next(iter(preflight.source_digests))
    stale = preflight.model_copy(
        update={
            "source_digests": {
                **preflight.source_digests,
                first_source: "0" * 64,
            }
        }
    )
    store = CampaignStore("formal-campaign", tmp_path)
    path = store.write_artifact("formal_preflights", stale)
    manifest = _manifest(bind_preflight(obligation, path.stem))

    with pytest.raises(ValueError, match="formal preflight sources are stale"):
        validate_formal_preflights(store.root, experiment, manifest)
