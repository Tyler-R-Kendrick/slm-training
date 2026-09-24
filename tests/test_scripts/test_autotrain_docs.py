"""A materialized cycle carries the ledger its successor gate will rebuild."""

import json

import pytest

from scripts.autotrain_docs import with_evidence_ledger
from slm_training.autoresearch.evidence_ledger import build_ledger


def test_document_bundle_ledger_matches_independent_successor_rebuild(tmp_path):
    design = tmp_path / "docs/design"
    design.mkdir(parents=True)
    old = design / "old.json"
    old.write_text('{}\n')
    files = {"docs/design/new-results.md": "# Diagnostic\n", "docs/design/new-results.json": '{}\n'}
    bundle = with_evidence_ledger(tmp_path, files)
    ledger = "src/slm_training/resources/experiments/autotrain_climb/evidence_ledger.v1.json"
    assert set(bundle) == {*files, ledger}
    assert not (design / "new-results.json").exists()
    assert json.loads(bundle[ledger])["file_counts"]["scanned"] == 2
    (design / "new-results.json").write_text(files["docs/design/new-results.json"])
    assert json.loads(bundle[ledger]) == build_ledger(design)
    # An overlay replacing an existing JSON is counted once, not as a new row.
    assert build_ledger(design, replacements={old: '[]'})["file_counts"]["scanned"] == 2
    with pytest.raises(ValueError, match="below the design directory"):
        build_ledger(design, replacements={tmp_path / "outside.json": '{}'})
