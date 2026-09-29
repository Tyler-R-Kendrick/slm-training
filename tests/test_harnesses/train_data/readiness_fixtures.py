"""Actual ExampleRecord/DataStore fixtures shared across readiness boundaries."""
import json
from pathlib import Path

from slm_training.data.readiness_contract import DataReadinessRequest
from slm_training.data.store import write_common_manifest
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.train_data.readiness import snapshot_input

CLEAN = 'root = Stack([c1])\nc1 = TextContent([":slot_0"])'
BAD = 'root = Stack([c1])\nc1 = RadialChart([":slot_0"], [1])'


def record(*, bad=False, **kwargs):
    return ExampleRecord(**{"id": "train-case", "prompt": "Show a card",
                            "openui": BAD if bad else CLEAN, "placeholders": [":slot_0"],
                            "meta": {"root_parent_id": "train-family"}, **kwargs})


def write_snapshot(root: Path, version: str, rows, *, kind="train", directory=None):
    directory = directory or root / "outputs" / "data" / kind / version
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "records.jsonl", rows)
    (directory / "manifest.json").write_text(json.dumps({"dataset_id": version, "kind": kind}))
    write_common_manifest(directory, kind=kind, dataset_id=version, immutable=True)
    return snapshot_input(root, directory, exposure="train_only" if kind == "train"
                          else "public_regression")


def request_fixture(root: Path, *, bad=True, minimum=1):
    original = write_snapshot(root, "original", [record(bad=bad)])
    suite = write_snapshot(root, "public-eval", [record(
        id="eval-case", prompt="Empty composition", openui="root = Stack([])",
        placeholders=[], split="held_out", meta={"root_parent_id": "eval-family"})], kind="eval")
    return DataReadinessRequest(
        action_id="data-action", campaign_id="campaign", purpose="validity",
        original=original, successor_id="successor", minimum_unique_cases=minimum,
        minimum_unique_families=minimum, scored_suites=[suite], training_ancestors=[],
        learned_tables=[], initialization="scratch", preprocessing_identity="fixture-certified-v1",
        trainer_config={"d_model": 16, "n_heads": 2, "context_layers": 1,
                        "denoiser_layers": 1, "max_prompt_len": 32, "max_target_len": 48})


def stage_rows(root, request, rows):
    stage = root / "outputs" / "runs" / "candidate" / request.successor_id
    stage.mkdir(parents=True)
    write_jsonl(stage / "records.jsonl", rows)
    return stage


def screening_context(root, train_version="wf_smoke_v2",
                      eval_version="e938_role_safe_all_targets_smoke96_v2"):
    """Explicit scratch fixture; never infer a checkpoint from a dataset name."""
    from dataclasses import asdict
    from slm_training.models.twotower import TwoTowerConfig
    from slm_training.harnesses.train_data.readiness import _screening_snapshots

    _, _, ancestor, suites = _screening_snapshots(root, train_version, eval_version)
    return {"trainer_config": asdict(TwoTowerConfig(d_model=16, n_heads=2)),
            "scored_suites": [ref.model_dump() for ref in suites],
            "training_ancestors": [ancestor.model_dump()], "learned_tables": [],
            "initialization": "scratch", "lineage_root": None, "starting_run_id": None,
            "preprocessing_identity": "fixture-locked-preprocessing/v1"}


def certified_screening_fixture(root, *, absolute_paths=False):
    """Fresh, content-valid published inputs; no inherited historical hashes."""
    write_snapshot(root, "wf_smoke_v2", [record()])
    directory = root / "outputs/data/eval/e938_role_safe_all_targets_smoke96_v2"
    directory.mkdir(parents=True)
    suites = {}
    for name in ("smoke", "held_out", "adversarial", "ood", "rico_held"):
        path = directory / "suites" / name / "records.jsonl"
        path.parent.mkdir(parents=True)
        write_jsonl(path, [record(id=name, prompt="Separate the page", openui="root = Separator()",
                                 placeholders=[], split=name, meta={"root_parent_id": name})])
        suites[name] = str(path) if absolute_paths else path.relative_to(root).as_posix()
    corpus = root / "corpus.jsonl"
    from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1
    policy = RootFamilySplitPolicyV1()
    family = next(f"sample-family-{i}" for i in range(1000)
                  if policy.assign(f"sample-family-{i}") == "validation")
    write_jsonl(corpus, [record(id="corpus", prompt="Submit the form",
        openui='root = Stack([action])\naction = Button(":slot_0")',
        meta={"root_parent_id": family})])
    (directory / "manifest.json").write_text(json.dumps({"suites": suites}))
    (directory / "quality_report.json").write_text(json.dumps({
        "certified": {"corpus": "corpus.jsonl", "samples": {"smoke": {"seed": 0, "splits": ["validation"]}}},
        "version_stamp": {"components": {"data.test": "v1", "data.train": "v2"}}}))
    write_common_manifest(directory, kind="eval", dataset_id=directory.name, immutable=True)
    return root
