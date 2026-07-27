"""The decode-invariant certificate must fail on real regressions.

A gate nobody has watched fail is a gate nobody knows works. Each test below
reproduces one way the invariants have actually been broken in this repo's
history and asserts the certificate catches it.
"""

from __future__ import annotations

import pytest

from scripts import verify_agent_surfaces
from scripts import verify_decode_invariants as verifier


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A minimal repo tree that passes, so each test can break exactly one thing."""
    (tmp_path / "src/slm_training/models").mkdir(parents=True)
    (tmp_path / "src/slm_training/harnesses/model_build").mkdir(parents=True)
    (tmp_path / "src/slm_training/dsl/operators").mkdir(parents=True)
    (tmp_path / "src/slm_training/web").mkdir(parents=True)
    (tmp_path / "tests/test_models").mkdir(parents=True)
    (tmp_path / "tests/test_web").mkdir(parents=True)
    (tmp_path / "docs/design").mkdir(parents=True)
    (tmp_path / "docs/openwiki").mkdir(parents=True)
    (tmp_path / ".github").mkdir(parents=True)
    (tmp_path / ".cursor/rules").mkdir(parents=True)
    (tmp_path / ".grok/workflows").mkdir(parents=True)
    for skill in (
        "autotrain",
        "honest-ship-eval",
        "improve-openui-harnesses",
        "running-experiment-matrices",
    ):
        (tmp_path / ".agents/skills" / skill).mkdir(parents=True)

    def write(relative: str, text: str) -> None:
        (tmp_path / relative).write_text(text, encoding="utf-8")

    write(
        "src/slm_training/levers.py",
        "CONSTRAINT_WEAKENING_LEVERS = {\n"
        '    "grammar_constrained": {"safe_value": True, "invariant": "I6",'
        ' "effect": "legality", "note": ""},\n'
        '    "allow_unconstrained_fallback": {"safe_value": False,'
        ' "invariant": "I6", "effect": "legality", "note": ""},\n'
        '    "grammar_fastpath": {"safe_value": True, "invariant": "I2",'
        ' "effect": "bypass", "note": ""},\n'
        "}\n"
        "CAPACITY_SCALING_LEVERS = {\n"
        '    "d_model": {"baseline_value": 128, "axis": "width", "note": ""},\n'
        '    "denoiser_layers": {"baseline_value": 4, "axis": "depth",'
        ' "note": ""},\n'
        "}\n",
    )
    config_body = (
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\n"
        "class {name}:\n"
        "    grammar_constrained: bool = True\n"
        "    allow_unconstrained_fallback: bool = False\n"
        "    grammar_fastpath: bool = True\n"
    )
    write(
        "src/slm_training/models/twotower.py",
        config_body.format(name="TwoTowerConfig"),
    )
    write(
        "src/slm_training/harnesses/model_build/config.py",
        config_body.format(name="ModelBuildConfig"),
    )
    write(
        "src/slm_training/harnesses/model_build/eval_policy.py",
        "MANDATORY_GENERATION_POLICY = {\n"
        '    "grammar_constrained": True,\n'
        '    "allow_unconstrained_fallback": False,\n'
        '    "grammar_fastpath": True,\n'
        "}\n",
    )
    write(
        "src/slm_training/dsl/operators/reserved_tokens.py",
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\n"
        "class ReservedOperatorTokenConfigV1:\n"
        "    enabled: bool = False\n",
    )
    write(
        "src/slm_training/models/onnx_inference.py",
        "require_constrained_generation\nfallback_used\n",
    )
    write(
        "src/slm_training/web/service.py",
        "raise GenerationExhausted\nrequire_constrained_production_config\n"
        "_raise_on_substituted_generation\n",
    )
    write("tests/test_models/test_inference_speed.py", "assert forwards_count == 0\n")
    write(
        "tests/test_web/test_onnx_inference.py",
        'assert int(evidence["singleton_bypasses"]) > 0\n',
    )
    doc = "# Decode invariants\n" + "".join(
        f"### {name} — x\n\n" for name in ("I1", "I2", "I3", "I4", "I6", "I11", "I12", "I13", "I14")
    )
    write(verifier.CANONICAL_DOC, doc)
    surface = "Non-negotiable architecture invariants\ndocs/design/decode-invariants.md\n"
    for relative in ("AGENTS.md", "CLAUDE.md", "GEMINI.md"):
        write(relative, surface)
    write(
        ".github/copilot-instructions.md",
        "Non-negotiable architecture invariants\nAGENTS.md\n",
    )
    write(
        ".cursor/rules/decode-invariants.mdc",
        "alwaysApply: true\ndocs/design/decode-invariants.md\n",
    )
    write(
        ".grok/workflows/autotrain.rhai",
        "// docs/design/decode-invariants.md\n",
    )
    for skill in (
        "autotrain",
        "honest-ship-eval",
        "improve-openui-harnesses",
        "running-experiment-matrices",
    ):
        write(f".agents/skills/{skill}/SKILL.md", "decode-invariants.md\n")
    for relative in verifier.LINKING_DOCS:
        if relative == verifier.CANONICAL_DOC:
            continue
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        write(relative, "see decode-invariants.md\n")

    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    # check_agent_surfaces() imports scripts.verify_agent_surfaces directly, which
    # reads paths off its own module-level ROOT rather than verifier.ROOT — patch
    # it too so the orphaned-surface tests exercise the sandboxed tree instead of
    # silently falling through to the real repository's surfaces.
    monkeypatch.setattr(verify_agent_surfaces, "ROOT", tmp_path)
    return tmp_path, write


def test_sandbox_baseline_certifies(sandbox) -> None:
    assert verifier.certify()["weakening_levers"] == [
        "allow_unconstrained_fallback",
        "grammar_constrained",
        "grammar_fastpath",
    ]


def test_reenabling_the_unconstrained_fallback_default_fails(sandbox) -> None:
    """The exact regression Phase A closed."""
    _, write = sandbox
    write(
        "src/slm_training/models/twotower.py",
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\n"
        "class TwoTowerConfig:\n"
        "    grammar_constrained: bool = True\n"
        "    allow_unconstrained_fallback: bool = True\n"
        "    grammar_fastpath: bool = True\n",
    )
    with pytest.raises(verifier.DecodeInvariantError, match="allow_unconstrained_fallback"):
        verifier.certify()


def test_unpinning_a_strict_policy_lever_fails(sandbox) -> None:
    _, write = sandbox
    write(
        "src/slm_training/harnesses/model_build/eval_policy.py",
        'MANDATORY_GENERATION_POLICY = {"grammar_constrained": True}\n',
    )
    with pytest.raises(verifier.DecodeInvariantError, match="does not pin"):
        verifier.certify()


def test_serving_path_returning_uncertified_text_fails(sandbox) -> None:
    """ONNX used to `return self._certify(text) or text`."""
    _, write = sandbox
    write(
        "src/slm_training/models/onnx_inference.py",
        "return self._certify(text) or text\n",
    )  # no require_constrained_generation, no fallback_used evidence
    with pytest.raises(verifier.DecodeInvariantError, match="fail-closed marker"):
        verifier.certify()


def test_missing_bypass_test_fails(sandbox) -> None:
    _, write = sandbox
    write("tests/test_models/test_inference_speed.py", "assert True\n")
    with pytest.raises(verifier.DecodeInvariantError, match="deterministic-bypass"):
        verifier.certify()


def test_orphaned_agent_surface_fails(sandbox) -> None:
    """copilot-instructions.md never referenced AGENTS.md before this change."""
    _, write = sandbox
    write(".github/copilot-instructions.md", "# RTK\n")
    with pytest.raises(verifier.DecodeInvariantError, match="copilot-instructions"):
        verifier.certify()


def test_doc_that_stops_linking_the_canonical_statement_fails(sandbox) -> None:
    _, write = sandbox
    write("docs/MODEL_CARD.md", "# Model card\n")
    with pytest.raises(verifier.DecodeInvariantError, match="MODEL_CARD"):
        verifier.certify()


def test_dropping_an_invariant_from_the_canonical_doc_fails(sandbox) -> None:
    _, write = sandbox
    write(verifier.CANONICAL_DOC, "# Decode invariants\n### I2 — x\n")
    with pytest.raises(verifier.DecodeInvariantError, match="invariant I3"):
        verifier.certify()


def test_reserved_op_tokens_flipping_on_fails(sandbox) -> None:
    _, write = sandbox
    write(
        "src/slm_training/dsl/operators/reserved_tokens.py",
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\n"
        "class ReservedOperatorTokenConfigV1:\n"
        "    enabled: bool = True\n",
    )
    with pytest.raises(verifier.DecodeInvariantError, match="default-off"):
        verifier.certify()


def test_emptying_the_registry_fails(sandbox) -> None:
    """Deleting the registry must not be a way to pass the gate."""
    _, write = sandbox
    write("src/slm_training/levers.py", "CONSTRAINT_WEAKENING_LEVERS = {}\n")
    with pytest.raises(verifier.DecodeInvariantError, match="empty"):
        verifier.certify()


def test_removing_the_registry_fails(sandbox) -> None:
    _, write = sandbox
    write("src/slm_training/levers.py", "MAX_RUN_MINUTES = 3\n")
    with pytest.raises(verifier.DecodeInvariantError, match="CONSTRAINT_WEAKENING_LEVERS"):
        verifier.certify()


def test_main_reports_a_regression_without_traceback(sandbox, capsys) -> None:
    _, write = sandbox
    write("src/slm_training/levers.py", "CONSTRAINT_WEAKENING_LEVERS = {}\n")
    assert verifier.main([]) == 1
    assert "decode-invariant regression" in capsys.readouterr().err


def test_main_prints_the_certificate_on_success(sandbox, capsys) -> None:
    assert verifier.main([]) == 0
    assert "canonical_doc" in capsys.readouterr().out
