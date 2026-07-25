from scripts.run_bridge_curriculum_matrix import main


def test_describe_and_manifest_modes(capsys) -> None:
    assert main(["--describe"]) == 0
    described = capsys.readouterr().out
    assert "oracle_difficulty" in described
    assert "analyze-exposure" in described
    assert main(["--manifest", "--epochs", "1", "--seeds", "0"]) == 0
    assert "BridgeCurriculumManifestV1" in capsys.readouterr().out


def test_resume_mode(capsys) -> None:
    assert main(["--resume", "--epochs", "1", "--seeds", "0"]) == 0
    assert "BridgeCurriculumResumeProofV1" in capsys.readouterr().out
