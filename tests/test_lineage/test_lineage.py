from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.model_cycle import main as model_cycle_main
from slm_training.lineage.data_cycle import (
    annotations_to_cycle_data,
    sample_on_policy_replay,
)
from slm_training.lineage.merge import merge_checkpoints, validate_merge_manifests
from slm_training.lineage.evaluation_snapshot import build_evaluation_snapshot
from slm_training.lineage.promotion import (
    deployment_failures,
    promotion_failures,
    wilson_lower_bound,
)
from slm_training.lineage.records import (
    ChampionPointer,
    EvaluationReport,
    RunManifest,
    content_sha,
)
from slm_training.lineage.store import LineageStore
from slm_training.lineage.tracks import (
    CAUSAL_BASE_CANDIDATES,
    TWOTOWER_E53_RECIPE,
)


def run_manifest(run_id: str, *, parent_ids: tuple[str, ...] = ()) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        track="twotower",
        parent_ids=parent_ids,
        base_model_id="base",
        base_model_revision="abc123",
        architecture_sha="arch",
        tokenizer_sha="tok",
        parameter_shapes_sha="shapes",
        data_snapshot_sha="data",
        eval_snapshot_sha="eval",
        recipe_sha="recipe",
        code_sha="code",
        seed=1,
        hardware={"device": "cpu"},
        artifact_uris=(),
        metrics={},
        lifecycle_state="running",
        initialization="parent" if parent_ids else "scratch",
        recipe={"lr": 1e-3},
        created_at="2026-01-01T00:00:00Z",
    )


def report(run_id: str, *, seed: int = 0, rung: float = 3.0) -> EvaluationReport:
    return EvaluationReport(
        report_id=f"{run_id}-{seed}-{rung}",
        run_id=run_id,
        eval_snapshot_sha="eval",
        created_at="2026-01-01T00:00:00Z",
        ship_gates_pass=True,
        weighted_nll=0.9,
        category_nll={"binding": 0.9, "structural": 0.9, "repair": 0.9},
        metrics={
            "parse_rate": 0.8,
            "meaningful_program_rate": 0.8,
            "placeholder_fidelity": 0.7,
            "contract_recall": 0.7,
            "structural_similarity": 0.8,
        },
        suite_sizes={"rico_held": 1500},
        seed=seed,
        token_rung=rung,
        artifact_size_bytes=900_000_000,
        warm_p95_seconds=14.0,
        comparisons={"total": 500, "candidate_wins": 290},
        metadata={"loss_suite_complete": True, "ranking_stable": True},
    )


def test_manifest_hash_is_canonical_and_run_directory_is_immutable(
    tmp_path: Path,
) -> None:
    assert content_sha({"b": 2, "a": 1}) == content_sha({"a": 1, "b": 2})
    store = LineageStore(tmp_path)
    manifest = run_manifest("run-a")
    store.create_run(manifest)
    with pytest.raises(FileExistsError):
        store.create_run(manifest)
    assert store.load_run("run-a") == manifest


def test_lifecycle_and_champion_pointer_are_atomic(tmp_path: Path) -> None:
    store = LineageStore(tmp_path)
    store.create_run(run_manifest("run-a"))
    validated = store.transition_run("run-a", "validated", artifact_uris=("model.pt",))
    champion = store.transition_run("run-a", "champion")
    pointer = ChampionPointer(
        pointer_id="pointer-a",
        track="twotower",
        run_id="run-a",
        artifact_uri="model.pt",
        manifest_sha=champion.sha,
        evaluation_report_sha="report",
        created_at="2026-01-01T00:00:00Z",
    )
    store.promote(pointer)
    assert validated.lifecycle_state == "validated"
    assert store.champion("twotower") == pointer
    assert json.loads((tmp_path / "champions/twotower/current.json").read_text())[
        "record"
    ].startswith("history/")


def test_parent_compatibility_and_merge_methods(tmp_path: Path) -> None:
    parent = run_manifest("parent")
    children = [
        run_manifest("a", parent_ids=("parent",)),
        run_manifest("b", parent_ids=("parent",)),
    ]
    assert validate_merge_manifests(parent, children) == children
    incompatible = RunManifest(
        **{**children[0].to_dict(), "run_id": "bad", "tokenizer_sha": "other"}
    )
    with pytest.raises(ValueError, match="incompatible"):
        validate_merge_manifests(parent, [children[0], incompatible])

    parent_path = tmp_path / "parent.pt"
    one_path = tmp_path / "one.pt"
    two_path = tmp_path / "two.pt"
    torch.save({"state_dict": {"w": torch.tensor([0.0, 0.0])}}, parent_path)
    torch.save({"state_dict": {"w": torch.tensor([1.0, 3.0])}}, one_path)
    torch.save({"state_dict": {"w": torch.tensor([3.0, 1.0])}}, two_path)
    average = merge_checkpoints(
        parent_path, [one_path, two_path], tmp_path / "average.pt", method="average"
    )
    assert torch.equal(
        torch.load(average, weights_only=True)["state_dict"]["w"],
        torch.tensor([2.0, 2.0]),
    )
    ties = merge_checkpoints(
        parent_path,
        [one_path, two_path],
        tmp_path / "ties.pt",
        method="ties",
        density=1.0,
    )
    assert torch.equal(
        torch.load(ties, weights_only=True)["state_dict"]["w"], torch.tensor([2.0, 2.0])
    )


def test_feedback_conversion_keeps_invalid_outputs_out_of_sft_and_dpo() -> None:
    rows = [
        {"id": "up", "prompt": "p", "openui": "good", "rating": "up", "valid": True},
        {
            "id": "down",
            "prompt": "p",
            "openui": "less good",
            "rating": "down",
            "valid": True,
        },
        {
            "id": "invalid",
            "prompt": "p",
            "openui": "broken",
            "rating": "down",
            "valid": False,
        },
    ]
    cycle = annotations_to_cycle_data(rows)
    assert [row["id"] for row in cycle.sft_positives] == ["up"]
    assert len(cycle.dpo_pairs) == 1
    assert cycle.dpo_pairs[0]["rejected"] == "less good"
    assert [row["id"] for row in cycle.verifier_negatives] == ["invalid"]


def test_replay_is_ten_percent_of_final_snapshot() -> None:
    new = [{"id": f"new-{index}"} for index in range(90)]
    history = [{"id": f"old-{index}"} for index in range(20)]
    mixed = sample_on_policy_replay(new, history, seed=3)
    replay = [row for row in mixed if (row.get("meta") or {}).get("on_policy_replay")]
    assert len(mixed) == 100
    assert len(replay) == 10


def test_promotion_and_deployment_gates() -> None:
    parent = EvaluationReport(**{**report("parent").to_dict(), "weighted_nll": 1.0})
    candidate = report("candidate")
    finalist = [
        report("candidate", seed=0, rung=1.0),
        report("candidate", seed=1),
        report("candidate", seed=2),
    ]
    assert promotion_failures(candidate, parent, finalist) == []
    assert deployment_failures(candidate) == []
    assert wilson_lower_bound(290, 500) > 0.5
    regressed = EvaluationReport(**{**candidate.to_dict(), "weighted_nll": 1.0})
    assert "weighted NLL did not improve over parent" in promotion_failures(
        regressed, parent, finalist
    )
    incomplete = EvaluationReport(
        **{**finalist[0].to_dict(), "metadata": {"ranking_stable": True}}
    )
    assert "loss suite is incomplete" in promotion_failures(
        candidate, parent, [incomplete, *finalist[1:]]
    )


def test_track_recipes_and_cli_init_pin_revisions(tmp_path: Path) -> None:
    assert TWOTOWER_E53_RECIPE["d_model"] == 192
    assert TWOTOWER_E53_RECIPE["output_tokenizer"] == "lexer"
    assert all(len(revision) == 40 for revision in CAUSAL_BASE_CANDIDATES.values())
    root = tmp_path / "lineage"
    assert (
        model_cycle_main(
            [
                "--lineage-root",
                str(root),
                "init",
                "--track",
                "twotower",
                "--run-id",
                "baseline",
                "--data-snapshot-sha",
                "data",
                "--eval-snapshot-sha",
                "eval",
            ]
        )
        == 0
    )
    manifest = LineageStore(root).load_run("baseline")
    assert manifest.base_model_revision == "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
    assert manifest.initialization == "scratch"
    LineageStore(root).transition_run("baseline", "validated")
    with pytest.raises(ValueError, match="cannot change architecture"):
        model_cycle_main(
            [
                "--lineage-root",
                str(root),
                "branch",
                "--parent",
                "baseline",
                "--run-id",
                "bad-layout",
                "--recipe-json",
                '{"d_model": 384}',
            ]
        )


def test_production_eval_snapshot_requires_full_rico_and_untrained_feedback(
    tmp_path: Path,
) -> None:
    suites: dict[str, Path] = {}
    for name, count in {
        "smoke": 3,
        "held_out": 5,
        "adversarial": 4,
        "ood": 4,
        "rico_held": 1500,
    }.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            "".join(
                json.dumps({"id": f"{name}-{index}"}) + "\n" for index in range(count)
            )
        )
        suites[name] = path
    holdout = tmp_path / "feedback.jsonl"
    holdout.write_text(json.dumps({"id": "feedback-1"}) + "\n")
    snapshot = build_evaluation_snapshot(
        "eval-v1", suites, holdout, training_ids={"train-1"}
    )
    assert snapshot.metadata["suite_sizes"]["rico_held"] == 1500
    with pytest.raises(ValueError, match="overlaps training"):
        build_evaluation_snapshot("bad", suites, holdout, training_ids={"feedback-1"})


def test_causal_base_lock_cannot_move(tmp_path: Path) -> None:
    store = LineageStore(tmp_path)
    first = RunManifest(
        **{
            **run_manifest("qwen2").to_dict(),
            "track": "causal_lm",
            "base_model_id": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "base_model_revision": CAUSAL_BASE_CANDIDATES[
                "Qwen/Qwen2.5-Coder-0.5B-Instruct"
            ],
        }
    )
    second = RunManifest(
        **{
            **first.to_dict(),
            "run_id": "qwen3",
            "base_model_id": "Qwen/Qwen3-0.6B",
            "base_model_revision": CAUSAL_BASE_CANDIDATES["Qwen/Qwen3-0.6B"],
        }
    )
    store.lock_base(first)
    with pytest.raises(ValueError, match="permanently locked"):
        store.lock_base(second)


def test_twotower_quantized_export_loads_in_browser_adapter(tmp_path: Path) -> None:
    from slm_training.models.onnx_inference import OnnxTwoTowerModel
    from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT
    from slm_training.models.twotower import TwoTowerModel

    model = TwoTowerModel.from_checkpoint(PLAYGROUND_DEMO_CHECKPOINT, device="cpu")
    artifacts = model.export(tmp_path / "export", format="onnx")
    assert sum(path.stat().st_size for path in artifacts) < 1_000_000_000
    loaded = OnnxTwoTowerModel.from_checkpoint(tmp_path / "export/model.pt")
    assert loaded.tokenizer.vocab_size == model.tokenizer.vocab_size


def test_causal_plugin_identity_and_grammar_masks_are_cached(monkeypatch) -> None:
    import slm_training.models.causal_lm_openui as causal

    class FakeModel:
        config = SimpleNamespace(model_type="fake")

        def state_dict(self):
            return {"w": torch.zeros(2, 2)}

    class FakeTokenizer:
        init_kwargs = {"name": "fake"}
        eos_token_id = 1

        def __len__(self):
            return 2

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "".join("x" if value == 0 else "" for value in ids)

    calls = 0

    def fake_stream(_text):
        nonlocal calls
        calls += 1
        return SimpleNamespace(hard_error=False)

    monkeypatch.setattr(causal, "stream_check", fake_stream)
    monkeypatch.setattr(
        causal, "validate", lambda _text: SimpleNamespace(serialized="x")
    )
    plugin = causal.CausalLMOpenUIPlugin(
        FakeModel(),
        FakeTokenizer(),
        causal.CausalLMOpenUIConfig("base", "revision"),
    )
    assert plugin.artifact_identity()["base_model_revision"] == "revision"
    first = plugin._allowed_ids(())
    first_calls = calls
    assert plugin._allowed_ids(()) == first
    assert calls == first_calls
