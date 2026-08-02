from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
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
    FORMAL_TEMPLATES,
    LEVERPROOF_ROOT,
    _run as _run_formal,
    bind_preflight,
    check_formal_trace,
    formal_trace_from_closure,
    run_formal_preflight,
    validate_formal_preflight_artifact,
    validate_formal_preflights,
)
from slm_training.autoresearch.schemas import (
    CampaignBudget,
    ExperimentKnobs,
    ExperimentSpec,
    FormalClaimV1,
    FormalObligationV1,
    FormalPreflightV1,
    FormalTraceStepV1,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.model_build.eval_runner import structural_similarity
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.lineage.records import canonical_json

ROOT = Path(__file__).resolve().parents[2]
LEVERPROOF_MAKEFILE = ROOT / "src/leverproof_lean/Makefile"


def _make_proofs(
    cwd: Path, *, lake: Path = Path("/bin/true"), path: str | None = None
) -> subprocess.CompletedProcess[str]:
    make = shutil.which("make")
    assert make is not None
    env = dict(os.environ)
    if path is not None:
        env["PATH"] = path
    return subprocess.run(
        [make, "-f", str(LEVERPROOF_MAKEFILE), "proofs", f"LAKE={lake}"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_leverproof_source_audit_requires_rg(tmp_path: Path) -> None:
    empty_bin = tmp_path / "bin"
    empty_bin.mkdir()

    result = _make_proofs(tmp_path, path=str(empty_bin))

    assert result.returncode != 0
    assert "proof source audit requires rg" in result.stderr


def test_leverproof_source_audit_fails_on_rg_error(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    rg = fake_bin / "rg"
    rg.write_text("#!/bin/sh\nexit 2\n", encoding="utf-8")
    rg.chmod(0o755)

    result = _make_proofs(tmp_path, path=str(fake_bin))

    assert result.returncode != 0
    assert "proof source audit failed (rg exit 2)" in result.stderr


def test_leverproof_source_audit_rejects_explicit_sorry_ax(tmp_path: Path) -> None:
    (tmp_path / "Escape.lean").write_text(
        'def forbiddenProofEscape := "sorryAx"\n', encoding="utf-8"
    )

    result = _make_proofs(tmp_path)

    assert result.returncode != 0
    assert "forbidden proof escape found" in result.stderr


def test_leverproof_theorem_audit_rejects_sorry_ax_output(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    lake.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "env" ]; then\n'
        '  echo "proof depends on axioms: [sorryAx]"\n'
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    lake.chmod(0o755)

    result = _make_proofs(tmp_path, lake=lake)

    assert result.returncode != 0
    assert "theorem audit found sorryAx" in result.stderr


def test_formal_timeout_reaps_lean_process_tree(tmp_path: Path) -> None:
    survivor = tmp_path / "lean-grandchild-survived"
    grandchild = (
        "import signal,time,pathlib; "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "time.sleep(0.5); "
        f"pathlib.Path({str(survivor)!r}).write_text('alive')"
    )
    parent = (
        "import signal,subprocess,sys,time; "
        "signal.signal(signal.SIGINT, lambda *_: sys.exit(23)); "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
        "time.sleep(30)"
    )

    result = _run_formal(
        [sys.executable, "-c", parent], cwd=tmp_path, timeout_seconds=0.3
    )
    time.sleep(0.3)

    assert result.returncode == 124
    assert "timed out" in result.stderr
    assert not survivor.exists()


def test_formal_verifier_loads_before_project_dependencies_are_installed() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import runpy; "
                "ns = runpy.run_path('scripts/verify_formal_contracts.py', "
                "run_name='formal_audit_import'); "
                "print(ns['_canonical_run_seconds']())"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == str(MAX_RUN_MINUTES * 60)


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
            CampaignArmV1(arm_id="control", role="control", config_sha256="a" * 64),
            CampaignArmV1(arm_id="candidate", role="candidate", config_sha256="b" * 64),
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
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    on_start=None,
    on_heartbeat=None,
) -> subprocess.CompletedProcess[str]:
    del cwd
    del timeout_seconds
    del on_start
    del on_heartbeat
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
    preflight, obligation = run_formal_preflight("formal-campaign", experiment, claim)
    store = CampaignStore("formal-campaign", tmp_path)
    path = store.write_artifact("formal_preflights", preflight)
    manifest = _manifest(bind_preflight(obligation, path.stem))

    assert validate_formal_preflights(store.root, experiment, manifest) == (preflight,)


def test_metric_preflight_uses_mathlib_free_leverproof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], Path]] = []

    def successful(
        command: list[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        on_start=None,
        on_heartbeat=None,
    ) -> subprocess.CompletedProcess[str]:
        del timeout_seconds
        del on_start
        del on_heartbeat
        calls.append((command, cwd))
        stdout = "Lean (version 4.30.0)" if "--version" in command else "passed"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("slm_training.autoresearch.formal._run", successful)
    claim = FormalClaimV1(
        template_id="metrics.structural_similarity_monotone",
        claim="Nondecreasing jaccard and depth components cannot reduce the proxy.",
        policy="required",
    )

    preflight, _ = run_formal_preflight("formal-campaign", _experiment(claim), claim)

    assert calls[0] == (["make", "test"], LEVERPROOF_ROOT)
    assert all(cwd == LEVERPROOF_ROOT for _, cwd in calls)
    assert preflight.template_version == "v2"
    assert preflight.theorem == (
        "LeverProofLean.StructuralMetrics.structural_similarity_mono"
    )
    assert preflight.mathlib_version == "none"
    assert preflight.status == "proved"


def test_formal_preflight_forwards_liveness_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts: list[int] = []
    heartbeats: list[int] = []
    forwarded: list[tuple[object, object]] = []

    def successful(
        command: list[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        on_start=None,
        on_heartbeat=None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        del timeout_seconds
        forwarded.append((on_start, on_heartbeat))
        assert on_start is not None
        assert on_heartbeat is not None
        on_start(321)
        on_heartbeat(321)
        stdout = "Lean (version 4.30.0)" if "--version" in command else "passed"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("slm_training.autoresearch.formal._run", successful)
    claim = FormalClaimV1(
        template_id="metrics.structural_similarity_monotone",
        claim="Nondecreasing jaccard and depth components cannot reduce the proxy.",
        policy="required",
    )

    preflight, _ = run_formal_preflight(
        "formal-campaign",
        _experiment(claim),
        claim,
        on_start=starts.append,
        on_heartbeat=heartbeats.append,
    )

    assert preflight.status == "proved"
    assert forwarded
    assert starts == [321, 321]
    assert heartbeats == [321, 321]


def test_sff_preflights_resolve_through_canonical_formal_registry() -> None:
    root = (
        ROOT
        / "src/slm_training/resources/experiments/semantic_factor_frontier"
        / "formal_preflights"
    )
    for path in sorted(root.glob("sff_*.json")):
        preflight = FormalPreflightV1.model_validate_json(path.read_text())
        claim = FormalClaimV1(
            template_id=preflight.template_id,
            claim=preflight.claim,
            policy=preflight.policy,
        )
        expected_sha = hashlib.sha256(
            canonical_json(preflight.model_dump(mode="json")).encode("utf-8")
        ).hexdigest()

        assert (
            validate_formal_preflight_artifact(
                path,
                campaign_id=preflight.campaign_id,
                experiment_id=preflight.experiment_id,
                claim=claim,
                expected_sha256=expected_sha,
            )
            == preflight
        )


def test_cached_preflight_validator_rejects_template_binding_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("slm_training.autoresearch.formal._run", _successful_lean)
    claim = FormalClaimV1(
        template_id="metrics.structural_similarity_monotone",
        claim="Nondecreasing jaccard and depth components cannot reduce the proxy.",
        policy="required",
    )
    experiment = _experiment(claim)
    preflight, _ = run_formal_preflight("formal-campaign", experiment, claim)
    stale = preflight.model_copy(update={"theorem": "Old.Metric.theorem"})
    path = CampaignStore("formal-campaign", tmp_path).write_artifact(
        "formal_preflights", stale
    )

    with pytest.raises(ValueError, match="formal preflight binding mismatch"):
        validate_formal_preflight_artifact(
            path,
            campaign_id="formal-campaign",
            experiment_id="formal-exp",
            claim=claim,
            expected_sha256=path.stem,
        )


def test_leverproof_templates_are_explicitly_routed() -> None:
    expected = {
        "metrics.structural_similarity_monotone",
        "sff.advisory-keys-legal",
        "sff.advisory-singleton-zero-work",
        "sff.factor-membership-roundtrip",
        "sff.golden-incidence-degrees",
        "sff.role-shuffle-membership",
        "sff.soft-token-collision",
    }
    actual = {
        template_id
        for template_id, template in FORMAL_TEMPLATES.items()
        if template.lean_project == "leverproof"
    }

    assert actual == expected


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
    preflight, obligation = run_formal_preflight("formal-campaign", experiment, claim)
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
    preflight, obligation = run_formal_preflight("formal-campaign", experiment, claim)
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
