"""SLM-291: evidence-bundle census CLI (deterministic, no-write)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.verify_evidence_bundles import main
from slm_training.harnesses.model_build.evidence_bundle_census import (
    build_census,
    census_model_card,
    render_markdown,
)

_CARD = """# Model card

## Current checkpoint roster

| Role | Run id | Kind | Location | Status |
| --- | --- | --- | --- | --- |
| Demo | `demo_run` | Fixture wiring | `src/slm_training/resources/checkpoints/playground_demo/last.pt` (git) | Demo only — **not** a quality or ship claim |
| Lost run | `lost_run` | CPU scratch | `/tmp/overnight/outputs/runs/lost_run/checkpoints/last.pt` (local) | 200 steps — **not promotable or ship** |
| Champion | `champ_run` | Full HF train | `outputs/runs/champ_run/checkpoints/last.pt` (local) | promoted champion, ship_candidate |
| Bucket run | `bucket_run` | Full HF train | `hf://buckets/TKendrick/OpenUI/checkpoints/bucket_run/last.pt` | promoted champion, verified |

## Other section
"""


def _write_card(tmp_path: Path, text: str = _CARD) -> Path:
    card = tmp_path / "MODEL_CARD.md"
    card.write_text(text, encoding="utf-8")
    return card


def test_model_card_census_classification(tmp_path: Path) -> None:
    rows = census_model_card(_write_card(tmp_path))
    by_run = {row["run_id"]: row for row in rows}
    assert by_run["demo_run"]["durability"] == "repo_local"
    assert not by_run["demo_run"]["claims_durable"]
    assert by_run["lost_run"]["durability"] == "ephemeral_local"
    assert not by_run["lost_run"]["claims_durable"]
    assert by_run["champ_run"]["claims_durable"]
    assert by_run["champ_run"]["verdict"] == "fail"
    assert any("non-durable" in f for f in by_run["champ_run"]["failures"])
    assert by_run["bucket_run"]["claims_durable"]
    assert by_run["bucket_run"]["verdict"] == "pass"


def test_census_verifies_committed_bundles(tmp_path: Path) -> None:
    design = tmp_path / "design"
    design.mkdir()
    payload = {
        "schema": "evidence_bundle/v1",
        "run_id": "r1",
        "claim_class": "diagnostic",
        "suite_hashes": {},
        "checkpoint": None,
        "raw_envelopes": [],
        "seeds": [0],
    }
    (design / "iter-x.json").write_text(json.dumps(payload), encoding="utf-8")
    census = build_census(
        model_card=_write_card(tmp_path), design_dir=design, artifact_root=tmp_path
    )
    assert census["schema"] == "evidence_bundle_census/v1"
    assert census["summary"]["evidence_bundles"] == 1
    assert census["evidence_bundles"][0]["verdict"] == "pass"
    # One roster failure: the local champion.
    assert census["summary"]["failures"] == 1
    assert len(census["census_sha256"]) == 64


def test_cli_exit_codes_and_output(tmp_path: Path) -> None:
    card = _write_card(tmp_path)
    design = tmp_path / "design"
    design.mkdir()
    out = tmp_path / "census.json"
    rc = main(
        [
            "--model-card",
            str(card),
            "--design-dir",
            str(design),
            "--artifact-root",
            str(tmp_path),
            "--output",
            str(out),
        ]
    )
    assert rc == 1
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["summary"]["failures"] == 1

    clean_card = _write_card(
        tmp_path,
        _CARD.replace(
            "| Champion | `champ_run` | Full HF train | `outputs/runs/champ_run/checkpoints/last.pt` (local) | promoted champion, ship_candidate |",
            "| Champion | `champ_run` | Full HF train | `hf://buckets/TKendrick/OpenUI/checkpoints/champ_run/last.pt` | promoted champion, ship_candidate |",
        ),
    )
    rc = main(
        [
            "--model-card",
            str(clean_card),
            "--design-dir",
            str(design),
            "--artifact-root",
            str(tmp_path),
            "--format",
            "markdown",
        ]
    )
    assert rc == 0


def test_render_markdown_lists_failures(tmp_path: Path) -> None:
    census = build_census(
        model_card=_write_card(tmp_path),
        design_dir=tmp_path / "empty",
        artifact_root=tmp_path,
    )
    text = render_markdown(census)
    assert "champ_run" in text
    assert "non-durable" in text


def test_census_is_deterministic(tmp_path: Path) -> None:
    card = _write_card(tmp_path)
    design = tmp_path / "design"
    design.mkdir()
    first = build_census(model_card=card, design_dir=design, artifact_root=tmp_path)
    second = build_census(model_card=card, design_dir=design, artifact_root=tmp_path)
    assert first["census_sha256"] == second["census_sha256"]
    assert hashlib.sha256(
        json.dumps(first, sort_keys=True).encode()
    ).hexdigest() == hashlib.sha256(json.dumps(second, sort_keys=True).encode()).hexdigest()
