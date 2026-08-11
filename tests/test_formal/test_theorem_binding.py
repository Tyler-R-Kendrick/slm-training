"""EVID-07 / SLM-534: exact proposition + environment theorem binding mutations."""

from __future__ import annotations

import json
from pathlib import Path


from slm_training.formal.authority import from_formal_object_v1
from slm_training.formal.judgment import JudgmentOutcome, SolverJudgmentV1, UnknownReason
from slm_training.formal.objects import export_lean_claim
from slm_training.formal.theorem_binding import (
    LeanTheoremBindingV1,
    ascription_to_expected_proposition,
    extract_declaration_type,
    parse_axiom_print,
    render_bridge_lean,
    seal_theorem_binding,
    verify_theorem_binding,
)


THEOREM_ID = "ExactClosure.closePass_subset"
MODULE = "LeverProofLean.ExactClosure"
LOCAL = "closePass_subset"


def _seed_project(root: Path, *, proposition: str = "True") -> None:
    (root / "lean-toolchain").write_text("leanprover/lean4:v4.30.0\n", encoding="utf-8")
    (root / "lakefile.toml").write_text('name = "stub"\n', encoding="utf-8")
    (root / "lake-manifest.json").write_text("{}\n", encoding="utf-8")
    path = root.joinpath(*MODULE.split(".")).with_suffix(".lean")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"theorem {LOCAL} : {proposition} := trivial\n",
        encoding="utf-8",
    )


def _seal(root: Path, *, axioms: tuple[str, ...] = ()) -> LeanTheoremBindingV1:
    return seal_theorem_binding(
        fq_name=THEOREM_ID,
        module=MODULE,
        lean_root=root,
        axiom_footprint=axioms,
    )


def test_extract_and_pi_conversion_round_trip() -> None:
    source = (
        "theorem closePass_subset (live : List Candidate) "
        "(results : List QueryResult) :\n"
        "    Subset (closePass live results) live := by\n"
        "  sorry\n"
    )
    ascription = extract_declaration_type(source, local_name="closePass_subset")
    expected = ascription_to_expected_proposition(ascription)
    assert "∀" in expected
    assert "Subset (closePass live results) live" in expected


def test_bridge_embeds_check_and_example(tmp_path: Path) -> None:
    _seed_project(tmp_path, proposition="True")
    binding = _seal(tmp_path)
    bridge = render_bridge_lean(
        module=binding.module,
        fq_name=binding.fq_name,
        expected_proposition=binding.expected_proposition,
    )
    assert f"#check {THEOREM_ID}" in bridge
    assert f"example : ExpectedProposition := {THEOREM_ID}" in bridge
    assert f"#print axioms {THEOREM_ID}" in bridge
    assert binding.bridge_digest


def test_parse_axiom_print_variants() -> None:
    none = parse_axiom_print(
        f"{THEOREM_ID} does not depend on any axioms",
        fq_name=THEOREM_ID,
    )
    assert none == ()
    some = parse_axiom_print(
        f"{THEOREM_ID} depends on axioms: [propext, Quot.sound]",
        fq_name=THEOREM_ID,
    )
    assert some == ("Quot.sound", "propext")


def test_export_seals_binding_into_object_and_authority(tmp_path: Path) -> None:
    _seed_project(tmp_path, proposition="True")
    obj = export_lean_claim(THEOREM_ID, lean_root=tmp_path, axiom_footprint=())
    raw = obj.payload["theorem_binding"]
    binding = LeanTheoremBindingV1.from_dict(raw)
    assert binding.fq_name == THEOREM_ID
    assert binding.expected_proposition_fingerprint
    authority = from_formal_object_v1(
        obj,
        judgment=SolverJudgmentV1(
            outcome=JudgmentOutcome.UNKNOWN,
            payload=UnknownReason.INCONCLUSIVE,
            checked=False,
        ),
    )
    assert authority.theorem_binding is not None
    assert (
        authority.source_digests["theorem_binding.expected_proposition"]
        == binding.expected_proposition_fingerprint
    )
    assert authority.toolchain.lean_version == binding.lean_version
    assert authority.toolchain.lake_manifest_digest == binding.lake_manifest_digest


def test_mutation_same_name_to_true_rejected(tmp_path: Path) -> None:
    _seed_project(
        tmp_path,
        proposition="∀ (live : Nat), live = live",
    )
    binding = _seal(tmp_path)
    # Mutate declaration to True while keeping the sealed binding.
    _seed_project(tmp_path, proposition="True")
    violations = verify_theorem_binding(
        binding,
        lean_root=tmp_path,
        catalog_module=MODULE,
        live_axiom_footprint=(),
    )
    assert any("proposition mutation" in item for item in violations)


def test_mutation_changed_conclusion_rejected(tmp_path: Path) -> None:
    _seed_project(tmp_path, proposition="True")
    binding = _seal(tmp_path)
    _seed_project(tmp_path, proposition="False")
    violations = verify_theorem_binding(
        binding,
        lean_root=tmp_path,
        live_axiom_footprint=(),
    )
    assert any("proposition mutation" in item for item in violations)


def test_mutation_changed_hypotheses_rejected(tmp_path: Path) -> None:
    path = tmp_path
    (path / "lean-toolchain").write_text("leanprover/lean4:v4.30.0\n", encoding="utf-8")
    (path / "lakefile.toml").write_text('name = "stub"\n', encoding="utf-8")
    (path / "lake-manifest.json").write_text("{}\n", encoding="utf-8")
    lean = path.joinpath(*MODULE.split(".")).with_suffix(".lean")
    lean.parent.mkdir(parents=True, exist_ok=True)
    lean.write_text(
        f"theorem {LOCAL} (n : Nat) : n = n := rfl\n",
        encoding="utf-8",
    )
    binding = _seal(path)
    lean.write_text(
        f"theorem {LOCAL} (n : Nat) (m : Nat) : n = n := rfl\n",
        encoding="utf-8",
    )
    # source-tree digest also drifts; either class of failure is rejection.
    violations = verify_theorem_binding(
        binding,
        lean_root=path,
        live_axiom_footprint=(),
    )
    assert violations


def test_mutation_toolchain_lock_rejected(tmp_path: Path) -> None:
    _seed_project(tmp_path, proposition="True")
    binding = _seal(tmp_path)
    (tmp_path / "lake-manifest.json").write_text('{"packages":[]}\n', encoding="utf-8")
    violations = verify_theorem_binding(
        binding,
        lean_root=tmp_path,
        live_axiom_footprint=(),
    )
    assert any("lake manifest" in item for item in violations)


def test_mutation_new_axiom_rejected(tmp_path: Path) -> None:
    _seed_project(tmp_path, proposition="True")
    binding = _seal(tmp_path, axioms=())
    violations = verify_theorem_binding(
        binding,
        lean_root=tmp_path,
        live_axiom_footprint=("classical.choice",),
    )
    assert any("unexpected axiom" in item for item in violations)


def test_mutation_catalog_redirection_rejected(tmp_path: Path) -> None:
    _seed_project(tmp_path, proposition="True")
    binding = _seal(tmp_path)
    violations = verify_theorem_binding(
        binding,
        lean_root=tmp_path,
        catalog_module="LeverProofLean.OtherModule",
        live_axiom_footprint=(),
    )
    assert any("catalog redirection" in item for item in violations)


def test_checker_rejects_binding_mutations(monkeypatch, tmp_path: Path) -> None:
    import slm_training.formal.checkers as checkers

    _seed_project(tmp_path, proposition="∀ (n : Nat), n = n")
    obj = export_lean_claim(THEOREM_ID, lean_root=tmp_path, axiom_footprint=())
    monkeypatch.setattr(checkers, "LEAN_ROOT", tmp_path)
    # Healthy seal should pass without bridge/project audit.
    ok = checkers.check_lean_kernel(
        obj,
        enabled=True,
        run_binding_bridge=False,
        run_project_audit=False,
        live_axiom_footprint=(),
    )
    assert ok.ok, ok.detail

    _seed_project(tmp_path, proposition="True")
    bad = checkers.check_lean_kernel(
        obj,
        enabled=True,
        run_binding_bridge=False,
        run_project_audit=False,
        live_axiom_footprint=(),
    )
    assert not bad.ok
    assert "proposition" in bad.detail or "digest" in bad.detail


def test_project_audit_is_additional_not_substitute(monkeypatch, tmp_path: Path) -> None:
    """A green make test must not wash out a broken proposition binding."""

    import slm_training.formal.checkers as checkers
    from slm_training.harness_core.bounded_process import (
        BoundedProcessResult,
        ProcessOutcome,
    )

    _seed_project(tmp_path, proposition="True")
    obj = export_lean_claim(THEOREM_ID, lean_root=tmp_path, axiom_footprint=())
    # Break the sealed proposition after export.
    _seed_project(tmp_path, proposition="False")
    monkeypatch.setattr(checkers, "LEAN_ROOT", tmp_path)

    def fake_runner(command, *, cwd, interrupt_after_seconds, kill_grace_seconds):
        return BoundedProcessResult(
            command=tuple(command),
            outcome=ProcessOutcome.COMPLETED,
            returncode=0,
            stdout="make test green",
            stderr="",
            duration_seconds=0.0,
        )

    monkeypatch.setattr(checkers, "run_bounded_process", fake_runner)
    result = checkers.check_lean_kernel(
        obj,
        enabled=True,
        run_binding_bridge=False,
        run_project_audit=True,
        live_axiom_footprint=(),
    )
    assert not result.ok
    assert "proposition" in result.detail or "digest" in result.detail


def test_binding_json_round_trip(tmp_path: Path) -> None:
    _seed_project(tmp_path)
    binding = _seal(tmp_path, axioms=("propext",))
    raw = json.loads(json.dumps(binding.to_dict()))
    again = LeanTheoremBindingV1.from_dict(raw)
    assert again == binding
