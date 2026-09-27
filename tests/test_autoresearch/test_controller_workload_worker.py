"""Fresh processes prove import separation and live delegation, not model quality."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from slm_training.harness_core.execution_release import prepare_release


def _controller(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    source = tmp_path / "controller-source"
    # Copy actual controller modules/resources, never hardlink candidate files.
    for prefix in ("scripts", "src/slm_training"):
        for path in (repo / prefix).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".lark", ".txt"}:
                target = source / path.relative_to(repo)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    shutil.copyfile(repo / "pyproject.toml", source / "pyproject.toml")
    helper = Path(__file__).with_name("controller_worker_fixture.py")
    target = source / helper.relative_to(repo)
    target.parent.mkdir(parents=True)
    shutil.copyfile(helper, target)
    _profile_validations(source, tmp_path / "controller-validation-profile.jsonl")
    execution = tmp_path / "controller"
    prepare_release(source, tmp_path / "controller-release", execution, tmp_path / "controller-outputs")
    return execution, helper.relative_to(repo)



def _profile_validations(source, destination):
    """Observe real validation calls; neither result nor identity checks change."""
    path = source / "src/slm_training/harness_core/controller_execution.py"
    with path.open("a") as stream:
        stream.write(
            "\n_measure = 'validated_controller_environment' if 'validated_controller_environment' in globals() else 'validate_controller'\n"
            "_validate_uninstrumented = globals()[_measure]\n"
            "def _profile_validation(*args, **kwargs):\n"
            "    import json, os, sys, time\n"
            "    started = time.monotonic()\n"
            "    try:\n"
            "        return _validate_uninstrumented(*args, **kwargs)\n"
            "    finally:\n"
            f"        with open({str(destination)!r}, 'a') as profile:\n"
            "            profile.write(json.dumps({'pid': os.getpid(), "
            "'caller': sys._getframe(1).f_code.co_name, "
            "'seconds': time.monotonic() - started}) + '\\n')\n"
            "globals()[_measure] = _profile_validation\n"
        )

def _workload(tmp_path):
    source = tmp_path / "workload-source"
    (source / "scripts").mkdir(parents=True)
    package = source / "src/slm_training"
    package.mkdir(parents=True)
    (source / "scripts/__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "resource.txt").write_text("accepted workload fixture")
    (source / "scripts/evaluate_model.py").write_text(
        "import json, pathlib, sys, slm_training\n"
        "root = pathlib.Path(sys.argv[1]); package = pathlib.Path(slm_training.__file__).parent\n"
        "assert (root / 'checkpoint').read_bytes() == b'committed training prefix'\n"
        "rows = root / 'rows'; assert rows.read_text() == 'preserved\\n'\n"
        "with rows.open('a') as stream: stream.write('new\\n')\n"
        "print(json.dumps({'module': __file__, 'package': str(package), "
        "'resource': (package / 'resource.txt').read_text(), 'cwd': str(pathlib.Path.cwd())}))\n"
    )
    execution = tmp_path / "workload"
    prepare_release(source, tmp_path / "workload-release", execution, tmp_path / "workload-outputs")
    return execution


def test_fresh_controller_delegates_with_workload_only_scientific_imports(tmp_path):
    controller, helper = _controller(tmp_path)
    workload = _workload(tmp_path)
    outputs = tmp_path / "campaigns"
    outputs.mkdir()
    checkpoint = outputs / "checkpoint"
    checkpoint.write_bytes(b"committed training prefix")
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (outputs / "rows").write_text("preserved\n")
    poison = tmp_path / "inherited-imports"
    poison.mkdir()
    marker = tmp_path / "sitecustomize-executed"
    (poison / "sitecustomize.py").write_text(
        f"from pathlib import Path; Path({str(marker)!r}).write_text('untrusted startup')\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"controller": str(controller), "outputs": str(outputs)}))
    bootstrap = "import runpy,sys;sys.path[:0]=[sys.argv.pop(1),sys.argv.pop(1)];runpy.run_path(sys.argv.pop(1),run_name='__main__')"
    result = subprocess.run([sys.executable, "-I", "-c", bootstrap, str(controller),
        str(controller / "src"), str(controller / helper), str(config)], cwd=workload,
        env={**os.environ, "PYTHONPATH": str(poison), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=100, check=False)
    assert result.returncode == 0, result.stderr
    assert not marker.exists()
    origins = json.loads((outputs / "origins.json").read_text())
    assert origins["controller_module"] == str(controller / "scripts/autoresearch.py")
    assert origins["controller_ROOT"] == str(controller)
    assert origins["activity"] == "origin-proof"
    assert origins["scientific"] == {
        "module": str(workload / "scripts/evaluate_model.py"),
        "package": str(workload / "src/slm_training"), "cwd": str(workload),
        "resource": "accepted workload fixture",
    }
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    assert (outputs / "rows").read_text() == "preserved\nnew\n"
