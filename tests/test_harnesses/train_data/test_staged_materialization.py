import hashlib
import json
from dataclasses import replace
from pathlib import Path

from slm_training.dsl.schema import load_jsonl
from slm_training.harnesses.synthesis_plan import SynthesisPlanV1
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data
from slm_training.harnesses.train_data.artifact_graph import (
    ArtifactGraphStore,
    ArtifactNodeV1,
)
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

REPO_ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = (
    REPO_ROOT / "src/slm_training/resources/synthesis_plans/dsh0_cap0_fixture.json"
)

PROMPT = "Create a vertical hero card with a title and body."
CANONICAL = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)
EQUIVALENT = (
    'root = Stack([panel], "column")\n'
    'title = TextContent(":hero.title")\n'
    'body = TextContent(":hero.body")\n'
    "panel = Card([title, body])"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _train_root() -> str:
    policy = RootFamilySplitPolicyV1()
    for index in range(10_000):
        root = f"staged-family-{index}"
        if policy.assign(root) == "train":
            return root
    raise AssertionError("no train root")


def _node(
    name: str,
    artifact_type: str,
    root: str,
    payload: dict,
    *,
    parents: tuple[str, ...] = (),
    surface: str | None = None,
) -> ArtifactNodeV1:
    return ArtifactNodeV1(
        artifact_id=_sha(name),
        artifact_type=artifact_type,
        root_family_id=root,
        split_group_id=root,
        split="train",
        parent_ids=parents,
        surface_sha256=_sha(surface or json.dumps(payload, sort_keys=True)),
        alpha_sha256=_sha(f"{name}:alpha"),
        canonical_ast_sha256=_sha(f"{name}:ast"),
        near_template_sha256=_sha(f"{name}:near"),
        payload=payload,
    )


def _stage_graph(
    output_dir: Path,
    *,
    invalid: bool = False,
    canonical_preference: bool = True,
) -> tuple[str, str, str]:
    root = _train_root()
    question = _node(
        "question",
        "question",
        root,
        {"prompt": PROMPT},
        surface=PROMPT,
    )
    canonical_surface = (
        'root = TextContent("untemplated prose")' if invalid else CANONICAL
    )
    canonical = _node(
        "canonical",
        "answer",
        root,
        {"openui": canonical_surface},
        parents=(question.artifact_id,),
        surface=canonical_surface,
    )
    equivalent_surface = (
        'root = TextContent("other untemplated prose")' if invalid else EQUIVALENT
    )
    equivalent = _node(
        "equivalent",
        "answer",
        root,
        {"openui": equivalent_surface},
        parents=(question.artifact_id,),
        surface=equivalent_surface,
    )
    pair = _node(
        "pair",
        "qa_pair",
        root,
        {
            "question_id": question.artifact_id,
            "accepted_answer_ids": [
                canonical.artifact_id,
                equivalent.artifact_id,
            ],
            "canonical_preference_answer_id": (
                canonical.artifact_id if canonical_preference else None
            ),
        },
        parents=(
            question.artifact_id,
            canonical.artifact_id,
            equivalent.artifact_id,
        ),
    )
    store = ArtifactGraphStore(output_dir)
    for node in (question, canonical, equivalent, pair):
        assert store.append(node).parent.name == "records"
    return pair.artifact_id, canonical.artifact_id, equivalent.artifact_id


def _write_plan(tmp_path: Path, destination: str) -> Path:
    plan = replace(
        SynthesisPlanV1.load(PLAN_PATH),
        plan_id="staged-materialization-fixture",
        destinations=(destination,),
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _config(plan_path: Path) -> TrainDataConfig:
    return TrainDataConfig(
        profile="permissive",
        source="staged",
        output_root=Path("out"),
        version="v1",
        synthesis_plan_path=plan_path,
        synthesizer="none",
        require_design_md=False,
        test_seed_path=None,
        include_frontier_artifacts=False,
        governance_artifacts=False,
        mixture_manifest=False,
    )


def test_no_plan_legacy_fixture_bytes_remain_pinned(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    seed = tmp_path / "seed.jsonl"
    seed.write_text(
        json.dumps(
            {
                "id": "legacy-fixture",
                "prompt": PROMPT,
                "openui": CANONICAL,
                "placeholders": [":hero.title", ":hero.body"],
                "split": "train",
                "source": "fixture",
                "meta": {"layout": "hero"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    build_train_data(
        TrainDataConfig(
            profile="permissive",
            seed_path=seed,
            rico_path=None,
            source="fixture",
            output_root=Path("out"),
            version="v1",
            synthesizer="none",
            require_design_md=False,
            test_seed_path=None,
            include_frontier_artifacts=False,
            governance_artifacts=False,
            mixture_manifest=False,
        )
    )

    assert _sha((Path("out/v1") / "records.jsonl").read_text()) == (
        "0831fdee53294f5fb41588f36b2fa3f6605ade4dd976ccbd9212f28f30ef2f6a"
    )


def test_staged_graph_uses_canonical_pipeline_and_rebuilds_deterministically(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = Path("out/v1")
    pair_id, canonical_id, equivalent_id = _stage_graph(output_dir)
    plan_path = _write_plan(tmp_path, output_dir.as_posix())

    first = build_train_data(_config(plan_path))
    records_bytes = (output_dir / "records.jsonl").read_bytes()
    card_bytes = (output_dir / "DATASET_CARD.md").read_bytes()
    second = build_train_data(_config(plan_path))

    rows = load_jsonl(output_dir / "records.jsonl")
    assert len(rows) == 2
    row = next(
        item
        for item in rows
        if item.meta["staged_sources"]["graph_node_id"] == canonical_id
    )
    assert {item.meta["staged_sources"]["graph_node_id"] for item in rows} == {
        canonical_id,
        equivalent_id,
    }
    assert row.id == f"staged-{pair_id}-{canonical_id}"
    assert row.meta["staged_sources"]["qa_pair_artifact_id"] == pair_id
    assert (
        row.meta["synthesis_plan"]["sha256"]
        == first["manifest"]["synthesis_plan"]["sha256"]
    )
    assert row.meta["staged_validation"]["integrity"]["passed"] is True
    assert first["manifest"]["artifact_graph"]["accepted_count"] == 2
    assert first["manifest"]["staged_materialization"] == {
        "accepted_count": 2,
        "rejected_count": 0,
        "preference_pair_count": 1,
    }
    assert (output_dir / "preference_pairs.jsonl").is_file()
    assert (output_dir / "records.jsonl").read_bytes() == records_bytes
    assert (output_dir / "DATASET_CARD.md").read_bytes() == card_bytes
    assert (
        second["manifest"]["content_fingerprint"]
        == first["manifest"]["content_fingerprint"]
    )


def test_invalid_staged_target_fails_closed_with_retained_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = Path("out/v1")
    pair_id, canonical_id, equivalent_id = _stage_graph(output_dir, invalid=True)
    plan_path = _write_plan(tmp_path, output_dir.as_posix())

    result = build_train_data(_config(plan_path))

    assert result["stats"]["record_count"] == 0
    rejected = [
        json.loads(line)
        for line in (output_dir / "rejected.jsonl").read_text().splitlines()
    ]
    assert any(
        row.get("id")
        in {
            f"staged-{pair_id}-{canonical_id}",
            f"staged-{pair_id}-{equivalent_id}",
        }
        and row["stage"] in {"normalize", "staged_validation"}
        for row in rejected
    )
    assert result["manifest"]["artifact_graph"]["quarantine_count"] == 2
    assert result["manifest"]["staged_materialization"]["rejected_count"] == 2


def test_qa_without_canonical_preference_still_materializes_answers(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = Path("out/v1")
    _stage_graph(output_dir, canonical_preference=False)
    plan_path = _write_plan(tmp_path, output_dir.as_posix())

    result = build_train_data(_config(plan_path))

    assert result["stats"]["record_count"] == 2
    assert result["manifest"]["staged_materialization"]["preference_pair_count"] == 0
    assert not (output_dir / "preference_pairs.jsonl").exists()
