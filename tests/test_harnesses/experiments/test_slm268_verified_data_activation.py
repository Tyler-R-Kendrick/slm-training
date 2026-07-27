from pathlib import Path
from slm_training.harnesses.experiments.slm268_verified_data_activation import build_activation_report

def test_activation_is_deterministic_and_fail_closed():
    root = Path(__file__).resolve().parents[3]
    first = build_activation_report(root)
    second = build_activation_report(root)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["activation_status"] == "blocked"
    assert first["launchable_rungs"] == []
    assert len(first["gates"]) == 8
    assert not any(g["qualified"] for g in first["gates"])
    assert len(first["manifest_sha256"]) == 64
