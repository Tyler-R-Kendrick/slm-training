"""Real child imports and resources must come from the selected workload."""

import json
import subprocess
import sys
from types import SimpleNamespace

from slm_training.autoresearch.engine import _stage_environment


def test_scientific_child_drops_controller_imports(tmp_path, monkeypatch):
    workload = tmp_path / "workload"
    controller = tmp_path / "controller"
    for root in (workload, controller):
        scripts = root / "scripts"
        package = root / "src" / "slm_training"
        scripts.mkdir(parents=True)
        package.mkdir(parents=True)
        (scripts / "__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        (package / "resource.txt").write_text(root.name)
        (scripts / "evaluate_model.py").write_text(
            "import json, pathlib, slm_training\n"
            "p = pathlib.Path(slm_training.__file__).parent\n"
            "print(json.dumps({'module': __file__, 'package': str(p), "
            "'cwd': str(pathlib.Path.cwd()), "
            "'resource': (p / 'resource.txt').read_text()}))\n"
        )
    monkeypatch.setenv("PYTHONPATH", str(controller / "src") + ":" + str(controller))
    monkeypatch.setenv("PYTHONHOME", str(controller / "invalid-python-home"))
    command = [sys.executable, "-m", "scripts.evaluate_model"]
    experiment = SimpleNamespace(knobs=SimpleNamespace(context_backend="scratch"))
    environment = _stage_environment(experiment, command, cwd=workload)
    result = subprocess.run(command, cwd=workload, env=environment,
                            capture_output=True, text=True, timeout=15, check=True)
    assert json.loads(result.stdout) == {
        "module": str(workload / "scripts/evaluate_model.py"),
        "package": str(workload / "src/slm_training"),
        "cwd": str(workload),
        "resource": "workload",
    }
    assert environment["PYTHONPATH"] == str(workload / "src")
    assert "PYTHONHOME" not in environment
