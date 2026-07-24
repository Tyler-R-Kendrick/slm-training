import json
from pathlib import Path

from scripts.run_valid_edit_flow_closeout import main


def test_verify_writes_replayable_closeout(tmp_path: Path) -> None:
    design_dir = tmp_path / "design"
    args = ["--output-dir", str(tmp_path), "--design-dir", str(design_dir)]
    assert main(["--mode", "run", *args]) == 0
    assert main(["--mode", "verify", *args]) == 0
    report = json.loads((tmp_path / "valid-edit-flow-closeout.json").read_text())
    assert report["issue"] == "SLM-207"
    assert report["selected_stack"]["runtime"] == "exact_cached_decoder_control"
