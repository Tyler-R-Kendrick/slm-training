"""Dashboard freshness: live local runs, bucket inventory, cache invalidation."""

from __future__ import annotations

import json

from slm_training.harnesses.model_build import checkpoint_bucket as bucket_mod
from slm_training.web.observability import Readers, _HF_JOBS_CACHE


def _write_minimal(root, run_id: str = "checkpoint-1") -> None:
    docs = root / "docs"
    design = docs / "design"
    design.mkdir(parents=True, exist_ok=True)
    (docs / "MODEL_CARD.md").write_text(
        f"""# Model card

## Current checkpoint roster

| Role | Run id | Kind | Location | Status |
| --- | --- | --- | --- | --- |
| Current checkpoint | `{run_id}` | TwoTower | `outputs/runs/{run_id}/last.pt` | candidate |

## Checkpoint history

| Date (UTC) | Run id | Bucket / path | Metric headline | Notes |
| --- | --- | --- | --- | --- |
| 2026-07-14 | `{run_id}` | `outputs/runs/{run_id}/` | pending | not evaluated |
""",
        encoding="utf-8",
    )
    (design / "quality-matrix-results.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "id": "E1",
                        "run_id": "experiment-1",
                        "pass": True,
                        "suites": {
                            "smoke": {
                                "n": 5,
                                "parse_rate": 1.0,
                                "placeholder_fidelity": 0.8,
                                "structural_similarity": 0.6,
                                "reward_score": 0.9,
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_local_run_checkpoints_merge_into_roster(tmp_path) -> None:
    _write_minimal(tmp_path)
    run_dir = tmp_path / "outputs" / "runs" / "live-train-42"
    run_dir.mkdir(parents=True)
    (run_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "run_id": "live-train-42",
                "context_backend": "hf",
                "finished_at": "2026-07-21T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "last.pt").write_bytes(b"ckpt")

    payload = Readers(tmp_path).checkpoints()
    by_id = {row["run_id"]: row for row in payload["checkpoints"]}
    assert "live-train-42" in by_id
    assert by_id["live-train-42"]["provenance"] == "live"
    assert by_id["live-train-42"]["source"] == "local_runs"
    assert payload["sources"]["local_runs"] >= 1
    assert payload["provenance"] == "live"


def test_bucket_inventory_parses_tree_and_merges(tmp_path, monkeypatch) -> None:
    _write_minimal(tmp_path)
    monkeypatch.delenv("SLM_DISABLE_REMOTE_INVENTORY", raising=False)
    bucket_mod._BUCKET_LIST_CACHE.update({"key": "", "ts": 0.0, "payload": None})

    def fake_fetch(url: str, *, timeout_s: float = 12.0, token: str | None = None):
        if url.rstrip("/").endswith("/TKendrick/OpenUI"):
            return {
                "id": "TKendrick/OpenUI",
                "updatedAt": "2026-07-19T17:25:05.151Z",
                "size": 100,
                "totalFiles": 2,
            }
        return [
            {
                "type": "file",
                "path": "checkpoints/bucket-run-9/last.pt",
                "size": 10,
                "mtime": "2026-07-19T10:00:00Z",
            },
            {
                "type": "file",
                "path": "checkpoints/bucket-run-9/train_summary.json",
                "size": 20,
                "mtime": "2026-07-19T11:00:00Z",
            },
        ]

    inventory = bucket_mod.list_bucket_checkpoint_runs(fetch=fake_fetch, ttl_s=0)
    assert inventory["ok"] is True
    assert inventory["count"] == 1
    assert inventory["runs"][0]["run_id"] == "bucket-run-9"
    assert inventory["runs"][0]["mtime"] == "2026-07-19T11:00:00Z"

    import slm_training.web.observability as obs

    monkeypatch.setattr(obs, "list_bucket_checkpoint_runs", lambda **_: inventory)
    payload = Readers(tmp_path).checkpoints()
    by_id = {row["run_id"]: row for row in payload["checkpoints"]}
    assert "bucket-run-9" in by_id
    assert by_id["bucket-run-9"]["provenance"] == "bucket"
    assert payload["bucket"]["ok"] is True
    assert payload["provenance"] == "bucket"

    # Model-card rows already present still upgrade when the same run is remote.
    _write_minimal(tmp_path, run_id="bucket-run-9")
    upgraded = Readers(tmp_path).checkpoints()
    card_row = next(row for row in upgraded["checkpoints"] if row["run_id"] == "bucket-run-9")
    assert card_row["source"] == "model_card"
    assert card_row["provenance"] == "bucket"
    assert upgraded["provenance"] == "bucket"


def test_research_scoreboard_invalidates_on_non_iter_json(tmp_path) -> None:
    _write_minimal(tmp_path)
    design = tmp_path / "docs" / "design"
    (design / "custom-campaign-results.json").write_text(
        json.dumps(
            {
                "run_id": "custom-1",
                "date_utc": "2026-07-21",
                "campaign": "custom campaign",
                "ship_gates": {"pass": False},
                "evaluation": {
                    "suites": {
                        "smoke": {
                            "n": 3,
                            "parse_rate": 0.5,
                            "placeholder_fidelity": 0.5,
                            "structural_similarity": 0.5,
                            "reward_score": 0.5,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    readers = Readers(tmp_path)
    first = readers.scoreboard("research")
    assert any(row.get("run_id") == "custom-1" for row in first["results"])

    (design / "custom-campaign-results.json").write_text(
        json.dumps(
            {
                "run_id": "custom-2",
                "date_utc": "2026-07-21",
                "campaign": "custom campaign v2",
                "ship_gates": {"pass": True},
                "evaluation": {
                    "suites": {
                        "smoke": {
                            "n": 3,
                            "parse_rate": 1.0,
                            "placeholder_fidelity": 1.0,
                            "structural_similarity": 1.0,
                            "reward_score": 1.0,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    second = readers.scoreboard("research")
    assert any(row.get("run_id") == "custom-2" for row in second["results"])
    assert not any(row.get("run_id") == "custom-1" for row in second["results"])


def test_insights_cache_invalidates_when_experiments_change(tmp_path) -> None:
    _write_minimal(tmp_path)
    readers = Readers(tmp_path)
    first = readers.performance_insights()
    assert first["cache"]["cached"] is False

    design = tmp_path / "docs" / "design"
    payload = json.loads((design / "quality-matrix-results.json").read_text(encoding="utf-8"))
    payload["results"].append(
        {
            "id": "E2",
            "run_id": "experiment-2",
            "pass": False,
            "suites": {
                "smoke": {
                    "n": 5,
                    "parse_rate": 0.2,
                    "placeholder_fidelity": 0.2,
                    "structural_similarity": 0.2,
                    "reward_score": 0.2,
                }
            },
        }
    )
    (design / "quality-matrix-results.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    changed = Readers(tmp_path).performance_insights()
    assert changed["cache"]["cached"] is False
    assert changed["reference_fingerprint"] != first["reference_fingerprint"]
    assert changed["stats"]["experiments"] >= 2


def test_system_freshness_includes_experiment_date(tmp_path) -> None:
    _write_minimal(tmp_path)
    design = tmp_path / "docs" / "design"
    (design / "iter-e999-20260721.json").write_text(
        json.dumps(
            {
                "run_id": "e999",
                "date_utc": "2026-07-21T18:00:00Z",
                "campaign": "freshness probe",
                "ship_gates": {"pass": True},
                "evaluation": {
                    "suites": {
                        "smoke": {
                            "n": 2,
                            "parse_rate": 1.0,
                            "placeholder_fidelity": 1.0,
                            "structural_similarity": 1.0,
                            "reward_score": 1.0,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    _HF_JOBS_CACHE.update({"ts": 0.0, "payload": None})
    system = Readers(tmp_path).system()
    assert system["outputs_present"] is False
    assert system["freshness"]["newest_experiment_date"] == "2026-07-21T18:00:00Z"
    assert system["freshness"]["experiment_count"] >= 1


def test_list_bucket_respects_disable_env(monkeypatch) -> None:
    monkeypatch.setenv("SLM_DISABLE_REMOTE_INVENTORY", "1")
    bucket_mod._BUCKET_LIST_CACHE.update({"key": "", "ts": 0.0, "payload": None})
    payload = bucket_mod.list_bucket_checkpoint_runs(ttl_s=0)
    assert payload["ok"] is False
    assert "disabled" in (payload.get("error") or "")
