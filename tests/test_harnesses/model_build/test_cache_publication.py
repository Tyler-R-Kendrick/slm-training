"""Cache-publication regressions on the actual evaluator; SDK calls are mocked."""

import hashlib
import json
from dataclasses import replace

import pytest

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harness_core.eval_cache import EvalCache, EvalCacheConfig, EvalCacheMode
from slm_training.harnesses.model_build import ModelBuildConfig, eval_runner
from slm_training.harnesses.model_build.plugin import StubModel


@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize("sdk_failure", [False, True])
def test_cache_hit_honors_current_publication(tmp_path, monkeypatch, preflight, sdk_failure):
    train, test = tmp_path / "train", tmp_path / "test"
    record = ExampleRecord(id="one", prompt="CTA", openui='root = Button(":slot_0")',
                           placeholders=[":slot_0"])
    write_jsonl(train / "records.jsonl", [record])
    write_jsonl(test / "records.jsonl", [record])
    write_jsonl(test / "suites/smoke/records.jsonl", [record])
    config = ModelBuildConfig(train_dir=train, test_dir=test, suite="smoke",
                              run_root=tmp_path / "runs", run_id="origin", model_name="stub")
    model = StubModel(noise_rate=0.0, seed=0)
    model.forward([record])
    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint)
    identity = dict(model_checkpoint_path=checkpoint,
                    model_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    cache = EvalCache(EvalCacheConfig(mode=EvalCacheMode.READ_WRITE, root=tmp_path / "cache"))
    original = eval_runner.evaluate(config, model=model, cache=cache, publish_agentv=False, **identity)
    assert original["completed_document_n"] == 1
    config = replace(config, run_id="destination")
    calls = []

    def publish(run_dir, board, **kwargs):
        calls.append(run_dir)
        if sdk_failure:
            raise RuntimeError("fixture SDK unavailable")
        return {"output": str(run_dir / "fixture-sdk-result.json")}

    def forbidden(*args, **kwargs):
        pytest.fail("cache hit must not construct or decode a model")

    monkeypatch.setattr("slm_training.evals.agentv.publish_model_evaluation", publish)
    monkeypatch.setattr(eval_runner, "build_model", forbidden)
    monkeypatch.setattr(model, "generate", forbidden)
    arguments = {"checkpoint": checkpoint} if preflight else {"model": model, **identity}
    if sdk_failure:
        with pytest.raises(RuntimeError, match="fixture SDK unavailable"):
            eval_runner.evaluate(config, cache=cache, **arguments)
    else:
        result = eval_runner.evaluate(config, cache=cache, **arguments)
        assert result["publication_complete"] is True
    assert calls == [config.run_dir]
    durable = json.loads((config.run_dir / "eval_smoke.json").read_text())
    assert durable["cache_replay"] and durable["completed_document_n"] == 1
    assert durable["publication_complete"] is (not sdk_failure)
    assert json.loads((config.run_dir / "eval.json").read_text()) == durable


def test_unpublished_replay_cannot_inherit_sdk_receipt(tmp_path):
    from types import SimpleNamespace

    config = SimpleNamespace(run_dir=tmp_path, suite="smoke")
    prior = {"n": 1, "publication_complete": True, "agentv": {"output": "old-run"}}
    replay = eval_runner._replay_cached_suite(config, prior, record_n=1)
    assert replay["publication_complete"] is False and "agentv" not in replay
    assert prior["publication_complete"] is True and prior["agentv"]["output"] == "old-run"
