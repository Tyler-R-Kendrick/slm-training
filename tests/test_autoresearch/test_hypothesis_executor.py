import json
import os
from types import SimpleNamespace

import pytest

from slm_training.autoresearch.heal import hypothesis_executor
from slm_training.autoresearch.heal.isolation import probe_isolation
from slm_training.autoresearch.heal.isolation_workspace import (
    manifest_digest,
    tree_manifest,
)


def test_hypothesis_output_survives_real_worker_exit(tmp_path):
    capability = probe_isolation()
    if not capability.available:
        if os.environ.get("SLM_REQUIRE_ISOLATION") == "1":
            pytest.fail(capability.reason)
        pytest.skip(f"real isolation unavailable: {capability.reason}")
    source = tmp_path / "source"
    source.mkdir()
    (source / "source.py").write_text("pass\n")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    worker = runtime / "worker"
    worker.write_text(
        "#!/usr/bin/python3\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "inputs = json.loads(Path('/workspace/proposal-input/instructions.json').read_text())\n"
        "output = Path(sys.argv[sys.argv.index('--output-last-message') + 1])\n"
        "output.write_text(json.dumps({'worker_output': inputs['request']}))\n"
        "print('stdout is not the proposal')\n"
    )
    worker.chmod(0o700)
    executor = hypothesis_executor.CodexHypothesisExecutor(
        source=source,
        attempt_root=tmp_path / "attempts",
        source_digest=manifest_digest(tree_manifest(source)),
        grant=SimpleNamespace(
            executable=str(worker), provider="openai",
            resources=SimpleNamespace(interrupt_seconds=10),
        ),
        runtime_roots=(runtime,),
        context={"instructions": "test", "owner_contract": "test", "supported_levers": []},
    )
    result = executor._run({"request": "producer-consumer regression"})
    assert json.loads(result) == {"worker_output": "producer-consumer regression"}
    output = executor.attempt_root / "candidate/proposal-output/matrix.json"
    assert output.read_text() == result
