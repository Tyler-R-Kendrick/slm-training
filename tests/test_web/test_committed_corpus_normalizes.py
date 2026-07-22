"""Every committed docs/design record must normalize or carry a typed reason.

This locks in the fix for the silent-drop regime where the dashboard rendered
12 of 378 committed records: discovery must account for every file, and the
row count must not regress below the corpus floor.
"""

from __future__ import annotations

from pathlib import Path

from slm_training.web.observability import Readers

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TYPED_REASONS = {
    "unreadable",
    "no_metric_blocks",
    "canonical_missing_suites",
}

# Observed at introduction: 257 normalized rows of 378 files. The floor leaves
# headroom for genuinely non-experiment additions without allowing a reader
# regression back toward the 12-row regime.
_MIN_ROWS = 200


def test_committed_corpus_fully_accounted_for() -> None:
    readers = Readers(_REPO_ROOT)
    rows = readers._research_results()
    unparsed = readers.last_unparsed
    total = len(list((_REPO_ROOT / "docs" / "design").glob("*.json")))
    assert len(rows) + len(unparsed) == total, "silent drop: file neither parsed nor rejected"
    assert len(rows) >= _MIN_ROWS, f"reader regression: only {len(rows)} rows normalized"
    assert all(entry["reason"] in _TYPED_REASONS for entry in unparsed)
    # Every row keeps provenance back to its committed file and dialect.
    assert all(row["source"].startswith("docs/design/") for row in rows)
    assert all(row.get("source_schema") for row in rows)


def test_newest_honest_boards_are_visible() -> None:
    # E292-E295 were the audit's flagship invisible records (non iter-* names
    # plus honest_evaluation short-key nesting). They must render.
    readers = Readers(_REPO_ROOT)
    sources = {row["source"] for row in readers._research_results()}
    for name in (
        "choice-loss-suite-results-iter-e292-20260717.json",
        "choice-component-plan-results-iter-e293-20260717.json",
        "choice-plan-control-results-iter-e294-20260717.json",
        "choice-design-dropout-results-iter-e295-20260717.json",
        "iter-e864-e865-opaque-marker-validity-20260722.json",
        "iter-e866-e867-semantic-contrast-opaque-slots-20260722.json",
        "iter-e868-e878-hard-tail-warmstart-20260722.json",
        "iter-e879-e885-vocab-union-matched-eval-20260722.json",
        "iter-e886-e888-recovered-baseline-20260722.json",
        "iter-e889-e890-hard-tail-current-policy-20260722.json",
        "iter-e891-e893-balanced-replay-20260722.json",
        "iter-e894-e896-low-replay-20260722.json",
        "iter-e900-e901-focused-role-continuation-20260722.json",
        "iter-e902-e903-focused-role-retention-20260722.json",
        "iter-e904-e907-e891-canvas-cap-20260722.json",
        "iter-e908-e915-typed-array-item-margin-20260722.json",
        "iter-e916-e920-schema-component-types-20260722.json",
        "iter-e921-e922-schema-inline-items-20260722.json",
        "iter-e923-e925-closed-array-arity-20260722.json",
        "iter-e926-e927-direct-component-types-20260722.json",
    ):
        assert f"docs/design/{name}" in sources, name
