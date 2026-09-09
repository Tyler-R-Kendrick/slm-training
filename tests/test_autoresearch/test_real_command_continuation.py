"""Actual bounded CPU trainer through the durable command cursor.

The first trainer deadline predicate is fault-injected after a real optimizer
update. Outer clocks, trainer computation, results, and bundles remain real.
This fixture certifies same-host continuation, not OpenUI quality or promotion.
"""

import json
from pathlib import Path
import sys

import pytest

from scripts.autoresearch_continuation import execute_with_continuation
from slm_training.autoresearch import engine
from slm_training.autoresearch.schemas import ExperimentKnobs
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.model_build.full_state import load_full_state
from tests.test_autoresearch.test_harness import experiment
from tests.test_harnesses.model_build.test_full_state_resume import (
    train_dir as corpus_fixture,
)

train_dir = corpus_fixture
ROOT = Path(__file__).resolve().parents[2]
UPDATES = 64


def _equal(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _equal(a, b)
    else:
        assert left == right


def _command(train_dir, runs, run_id):
    return [
        sys.executable,
        "-m",
        "scripts.train_model",
        "--train-dir",
        str(train_dir),
        "--run-root",
        str(runs),
        "--run-id",
        run_id,
        "--steps",
        str(UPDATES),
        "--model",
        "twotower",
        "--context-backend",
        "scratch",
        "--no-freeze-context",
        "--batch-size",
        "2",
        "--grad-accum",
        "2",
        "--lr",
        "0.003",
        "--seed",
        "0",
        "--d-model",
        "32",
        "--n-heads",
        "4",
        "--context-layers",
        "1",
        "--denoiser-layers",
        "1",
        "--device",
        "cpu",
        "--no-telemetry",
        "--no-sync-checkpoints",
        "--no-fast-train",
    ]


@pytest.mark.training
def test_actual_trainer_cursor_resume_is_bit_exact(train_dir, tmp_path, monkeypatch):
    from scripts.autoresearch_command_cursor import resolved_continuation_grant

    runs = tmp_path / "runs"
    grants = resolved_continuation_grant(ROOT, 120)
    spec = experiment(
        hypothesis="A bounded yield preserves same-environment CPU training state.",
        rationale="Exercise actual trainer subprocesses through the command cursor.",
        expected_effect="Exact state and consumed-resource parity, not quality gain.",
        falsification_criteria=("Any state or consumption mismatch.",),
        stop_conditions=(f"{UPDATES} optimizer updates; 120 seconds total per trial.",),
        # The full immutable argv binds architecture and accumulation as well.
        knobs=ExperimentKnobs(steps=UPDATES, batch_size=2, lr=0.003, seed=0),
    )
    options = dict(
        wall_seconds=100,
        campaign_manifest_sha256="c" * 64,
        execute_commands=engine.execute_commands,
        cwd=ROOT,
        grant=grants,
    )
    whole = execute_with_continuation(
        spec,
        [_command(train_dir, runs, "whole")],
        store=CampaignStore(spec.campaign_id, tmp_path / "whole-store"),
        **options,
    )
    assert whole.status == "completed", whole.model_dump()
    original_runner = engine.run_bounded_process
    launches = []
    process_commands = []

    def first_deadline(argv, **kwargs):
        argv = list(argv)
        # Force expiry at a safe boundary without host-speed-dependent sleeps.
        # The small test launcher invokes the actual canonical trainer main.
        if not launches:
            index = argv.index("--max-wall-minutes")
            argv[index + 1] = str(min(float(argv[index + 1]), 0.5))
            child = [
                argv[0],
                "-m",
                "tests.test_autoresearch.deadline_train_fixture",
                *argv[2:],
            ]
        else:
            child = argv
        launches.append(argv)
        process_commands.append(child)
        return original_runner(child, **kwargs)

    monkeypatch.setattr(engine, "run_bounded_process", first_deadline)
    plan = [_command(train_dir, runs, "chunked")]
    store = CampaignStore(spec.campaign_id, tmp_path / "chunked-store")
    resumed = execute_with_continuation(spec, plan, store=store, **options)
    assert resumed.status == "completed", resumed.model_dump()
    assert len(launches) == 2
    assert "--max-wall-minutes" not in plan[0]
    assert "--resume-from" in launches[1]
    first = resumed.stage_telemetry[0]
    assert first["resume_kind"] == "training"
    assert first["completed_optimizer_updates"] == 1
    assert first["requested_optimizer_updates"] == UPDATES
    a = load_full_state(runs / "whole/checkpoints/last_full_state.pt")
    b = load_full_state(runs / "chunked/checkpoints/last_full_state.pt")
    for key in (
        "model",
        "optimizer",
        "scaler",
        "torch_rng",
        "python_rng",
        "numpy_rng",
        "cuda_rng",
        "loop_rng",
        "model_mask_rng",
        "pending_batch_ids",
        "step",
        "accumulation_position",
        "seen_primary_examples",
        "seen_replay_examples",
        "seen_target_tokens",
        "seen_prompt_tokens",
        "resume_contract",
    ):
        _equal(a[key], b[key])
    summaries = [
        json.loads((runs / name / "train_summary.json").read_text())
        for name in ("whole", "chunked")
    ]
    assert summaries[0]["exposure_by_snapshot"] == summaries[1]["exposure_by_snapshot"]
    assert summaries[1]["continuation_kind"] == "exact_same_environment"
    assert a["step"] == b["step"] == UPDATES and a["seen_primary_examples"] > 0
    events = store.verify_event_chain()
    assert sum(e["event_type"] == "command_cursor_started" for e in events) == 2
    assert not any("promot" in e["event_type"] for e in events)
    print(
        json.dumps(
            {
                "evidence_class": "actual_cpu_training_with_injected_deadline_predicate",
                "clock_fault": "trainer deadline predicate advanced after first actual update; outer wall and elapsed telemetry real",
                "execution_identity": grants.execution_identity,
                "commands": launches,
                "actual_process_commands": process_commands,
                "whole": whole.model_dump(mode="json"),
                "resumed": resumed.model_dump(mode="json"),
                "examples": b["seen_primary_examples"],
                "target_tokens": b["seen_target_tokens"],
                "prompt_tokens": b["seen_prompt_tokens"],
                "exposure_by_snapshot": summaries[1]["exposure_by_snapshot"],
                "exact_state_parity": True,
                "resume_contract": b["resume_contract"],
            }
        )
    )
