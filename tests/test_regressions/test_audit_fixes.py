"""Regressions for high-risk findings from the main-branch audit."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gpu_multi_farm.models import FarmListResult, Offer
from gpu_multi_farm.registry import list_across_farms
from scripts import remote_train
from slm_training.harnesses.annotations import (
    AnnotationRecord,
    load_annotations,
    persist_annotation,
    utc_now_iso,
)
from slm_training.dsl import bridge_available
from slm_training.dsl.schema import ExampleRecord, load_jsonl, write_jsonl
from slm_training.harnesses.model_build import ModelBuildConfig, evaluate
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data
from slm_training.web.app import create_app


class _BlockingStdout:
    def readline(self) -> str:
        threading.Event().wait(1)
        return ""


class _BlockingProc:
    stdout = _BlockingStdout()
    args = ["node", "bridge.mjs", "--repl"]


@pytest.mark.parametrize(
    "reader",
    [
        pytest.param(
            __import__(
                "slm_training.bridge_utils", fromlist=["readline_with_timeout"]
            ).readline_with_timeout,
            id="bridge-utils",
        ),
    ],
)
def test_bridge_readline_deadline(reader) -> None:
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        reader(_BlockingProc(), 0.02, error_message="bridge timed out")
    assert time.monotonic() - started < 0.25


def test_prompt_variation_is_cross_process_stable() -> None:
    code = (
        "from slm_training.web.prompts import vary_prompt;"
        "print(vary_prompt('Hero card with button', salt=7))"
    )
    outputs = {
        subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        for _ in range(4)
    }
    assert len(outputs) == 1
    assert " utton " not in f" {next(iter(outputs)).lower()} "


def test_concurrent_annotation_promotions_do_not_lose_rows(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback.jsonl"
    human = tmp_path / "human.jsonl"
    pairs = tmp_path / "pairs.jsonl"
    barrier = threading.Barrier(2)

    def _persist(index: int) -> None:
        record = AnnotationRecord(
            id=f"r{index}",
            ts=utc_now_iso(),
            prompt=f"Prompt {index}",
            openui=(
                f"root = Stack([item{index}])\n"
                f'item{index} = TextContent(":item.{index}")'
            ),
            rating="up",
        )
        barrier.wait()
        persist_annotation(
            record,
            feedback_path=feedback,
            human_train_path=human,
            pairs_path=pairs,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(_persist, range(2)))

    assert len(load_annotations(feedback)) == 2
    assert len(load_jsonl(human)) == 2


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge deps missing")
def test_curriculum_never_imports_evaluation_fixtures(tmp_path: Path) -> None:
    seeds = tmp_path / "train.jsonl"
    write_jsonl(
        seeds,
        [
            ExampleRecord(
                id="train_only",
                prompt="A train-only button",
                openui='root = Stack([cta])\ncta = Button(":train.cta")',
                placeholders=[":train.cta"],
            )
        ],
    )
    result = build_train_data(
        TrainDataConfig(
            seed_path=seeds,
            human_annotations_path=None,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "out",
            version="curriculum",
            synthesizer="none",
            curriculum=True,
        )
    )
    records = load_jsonl(Path(result["output_dir"]) / "records.jsonl")
    ids = {record.id for record in records}
    assert "train_only" in ids
    assert all(
        record_id == "train_only" or record_id.startswith("curriculum_c_stress_")
        for record_id in ids
    )
    assert all(record.split == "train" for record in records)


def test_annotation_token_is_enforced(tmp_path: Path) -> None:
    app = create_app(
        checkpoint=tmp_path / "missing.pt",
        annotations_path=tmp_path / "feedback.jsonl",
        human_train_path=tmp_path / "human.jsonl",
        human_pairs_path=tmp_path / "pairs.jsonl",
        annotation_token="secret",
    )
    from fastapi.testclient import TestClient

    payload = {
        "prompt": "Button",
        "openui": 'root = Stack([cta])\ncta = Button(":cta")',
        "rating": "up",
    }
    client = TestClient(app)
    assert client.post("/api/annotate", json=payload).status_code == 401
    assert (
        client.post(
            "/api/annotate",
            json=payload,
            headers={"Authorization": "Bearer secret"},
        ).status_code
        == 200
    )
    assert client.get("/api/annotations/recent").status_code == 401
    assert (
        client.get(
            "/api/annotations/recent",
            headers={"Authorization": "Bearer secret"},
        ).status_code
        == 200
    )


def test_evaluation_requires_a_real_checkpoint(tmp_path: Path) -> None:
    suite = tmp_path / "test" / "suites" / "smoke"
    suite.mkdir(parents=True)
    write_jsonl(
        suite / "records.jsonl",
        [
            ExampleRecord(
                id="smoke",
                prompt="Button",
                openui='root = Stack([cta])\ncta = Button(":cta")',
                split="smoke",
            )
        ],
    )
    config = ModelBuildConfig(
        train_dir=tmp_path / "train",
        test_dir=tmp_path / "test",
        run_root=tmp_path / "runs",
        model_name="stub",
    )
    with pytest.raises(FileNotFoundError, match="evaluation checkpoint"):
        evaluate(config)


@pytest.mark.asyncio
async def test_multi_farm_listing_is_concurrent() -> None:
    class Client:
        timeout_s = 1.0

        def __init__(self, name: str) -> None:
            self.name = name

        async def list_offers(self, **_kwargs) -> FarmListResult:
            import asyncio

            await asyncio.sleep(0.1)
            return FarmListResult(farm=self.name)

    started = time.monotonic()
    result = await list_across_farms(
        {name: Client(name) for name in ("a", "b", "c")},
        gpu_type=None,
        max_price_per_hr=None,
    )
    assert set(result) == {"a", "b", "c"}
    assert time.monotonic() - started < 0.25


@pytest.mark.parametrize("run_id", [".", ".."])
def test_remote_train_rejects_dot_segments(run_id: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        remote_train.main(["--host", "example", "--run-id", run_id, "--dry-run"])
    assert exc_info.value.code == 2


def test_remote_train_fails_when_checkpoint_copy_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def _run(cmd: list[str], **_kwargs):
        calls.append(cmd)
        if cmd[0] == "ssh" and "pwd -P" in cmd[-1]:
            return subprocess.CompletedProcess(cmd, 0, "/home/audit/custom\n", "")
        if cmd[0] == "scp":
            return subprocess.CompletedProcess(cmd, 7)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(remote_train.subprocess, "run", _run)
    rc = remote_train.main(
        [
            "--host",
            "example",
            "--remote-dir",
            "~/custom",
            "--pull-dir",
            str(tmp_path),
        ]
    )
    assert rc == 7
    scp = next(cmd for cmd in calls if cmd[0] == "scp")
    assert "/home/audit/custom/outputs/runs/remote_run/checkpoints/" in scp[-2]


@pytest.mark.asyncio
async def test_vast_search_uses_current_flat_post_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gpu_multi_farm.farms import vast

    calls: list[tuple[str, dict]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "offers": [
                    {
                        "id": 1,
                        "gpu_name": "RTX 4090",
                        "dph_total": 0.4,
                        "rentable": True,
                    }
                ]
            }

    class Client:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, url: str, **kwargs):
            calls.append((url, kwargs["json"]))
            return Response()

    monkeypatch.setattr(vast.httpx, "AsyncClient", Client)
    result = await vast.VastClient("key").list_offers(gpu_type="4090")
    assert result.error is None
    assert result.offers
    assert calls[0][0].endswith("/bundles/")
    assert calls[0][1]["type"] == "ondemand"
    assert "q" not in calls[0][1]
    assert "gpu_name" not in calls[0][1]


@pytest.mark.asyncio
async def test_vast_launch_uses_put_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gpu_multi_farm.farms import vast

    async def list_offers(_self, **_kwargs) -> FarmListResult:
        return FarmListResult(
            farm="vast",
            offers=[Offer("vast", "42", "RTX 4090", 0.4)],
        )

    calls: list[tuple[str, dict]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"new_contract": 99}

    class Client:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def put(self, url: str, **kwargs):
            calls.append((url, kwargs["json"]))
            return Response()

    monkeypatch.setattr(vast.VastClient, "list_offers", list_offers)
    monkeypatch.setattr(vast.httpx, "AsyncClient", Client)
    result = await vast.VastClient("key").launch({"gpu_type": "4090"})
    assert result.pod_id == "99"
    assert calls[0][0] == f"{vast.VAST_BASE}/asks/42/"
    assert calls[0][1]["runtype"] == "ssh"


@pytest.mark.asyncio
async def test_lambda_gpu_count_is_not_reported_as_vram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gpu_multi_farm.farms import lambda_labs

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": {
                    "gpu_8x_a100": {
                        "instance_type": {
                            "name": "gpu_8x_a100",
                            "description": "8x NVIDIA A100",
                            "price_cents_per_hour": 1000,
                            "specs": {"gpus": 8},
                        },
                        "regions_with_capacity_available": [{"name": "us-test-1"}],
                    }
                }
            }

    class Client:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, _url: str, **_kwargs):
            return Response()

    monkeypatch.setattr(lambda_labs.httpx, "AsyncClient", Client)
    result = await lambda_labs.LambdaClient("key").list_offers()
    assert result.error is None
    assert result.offers[0].vram_gb is None
    assert result.offers[0].raw_ref["gpu_count"] == 8
