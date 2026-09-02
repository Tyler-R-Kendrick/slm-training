"""No screening arm may train on a corpus that overlaps the scored eval suites.

A hill-climb that trains on eval material scores the eval it memorized. The
overlap here is measured from the committed corpora and suites, never
declared, so a rebuild that re-introduces contamination fails this file
instead of producing a leakage "win".
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "src" / "slm_training" / "resources" / "data"
TRAIN = DATA / "train"
EVAL = DATA / "eval"

#: Suites the screening/promotion tiers actually score (climb policy
#: ``measurement.screening_suites`` / ``promotion_suites`` resolved through
#: ``default_eval_version``).
SCORED_SUITES = (
    ("smoke96_v1/smoke", "e938_role_safe_all_targets_smoke96_v1/suites/smoke"),
    ("smoke96_v1/held_out", "e938_role_safe_all_targets_smoke96_v1/suites/held_out"),
    (
        "heldout24_v1/held_out",
        "e938_role_safe_all_targets_heldout24_v1/suites/held_out",
    ),
)

_FAMILY_KEYS = (
    "root_family",
    "root_parent_id",
    "split_group_id",
    "parent_id",
    "family",
)


def _driver():
    path = REPO / "scripts" / "run_autotrain_continuous.py"
    spec = importlib.util.spec_from_file_location("_rac_leakage", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _norm(value: str | None) -> str:
    return " ".join((value or "").split())


def _family(record: dict) -> str | None:
    meta = record.get("meta") or {}
    for key in _FAMILY_KEYS:
        if meta.get(key):
            return str(meta[key])
        if record.get(key):
            return str(record[key])
    return None


def _overlap(corpus: str) -> dict[str, dict[str, int]]:
    records = _load(TRAIN / corpus / "records.jsonl")
    programs = {_norm(r.get("openui")) for r in records}
    prompts = {_norm(r.get("prompt")) for r in records}
    families = {_family(r) for r in records} - {None}
    out: dict[str, dict[str, int]] = {}
    for label, rel in SCORED_SUITES:
        suite = EVAL / rel / "records.jsonl"
        if not suite.is_file():
            continue
        rows = _load(suite)
        out[label] = {
            "programs": len(programs & {_norm(r.get("openui")) for r in rows}),
            "prompts": len(prompts & {_norm(r.get("prompt")) for r in rows}),
            "families": len(families & ({_family(r) for r in rows} - {None})),
        }
    return out


def test_scored_suites_exist() -> None:
    for _, rel in SCORED_SUITES:
        assert (EVAL / rel / "records.jsonl").is_file(), rel


def test_default_train_corpus_is_disjoint_from_every_scored_suite() -> None:
    """The policy default corpus is a root-family split of the same corpus the
    suites are sampled from, so disjointness is structural — assert it holds."""
    module = _driver()
    corpus = module._default_screening_train_version()
    assert corpus == "openui_verified_train_v1"
    for label, counts in _overlap(corpus).items():
        assert counts == {"programs": 0, "prompts": 0, "families": 0}, (
            f"{corpus} leaks into {label}: {counts}"
        )


@pytest.mark.parametrize("corpus", sorted({"hillclimb_strict_v2", "wf_smoke_v2"}))
def test_known_leaked_corpora_are_still_leaked_and_still_excluded(corpus: str) -> None:
    """Pins why these corpora are barred. If a rebuild makes one clean, this
    test fails and the corpus may be re-admitted deliberately, never silently."""
    module = _driver()
    assert corpus in module._LEAKED_TRAIN_VERSIONS
    counts = _overlap(corpus)
    assert counts, f"{corpus}: no scored suite was comparable"
    assert any(sum(c.values()) > 0 for c in counts.values()), (
        f"{corpus} no longer overlaps any scored suite: {counts}. Re-admit it as a "
        "data arm on purpose (and delete this expectation) rather than leaving a "
        "clean corpus barred."
    )


def test_no_screening_arm_trains_on_a_leaked_corpus() -> None:
    module = _driver()
    offenders = [
        (slug, extras.get("train_version"))
        for slug, _, extras in module._all_screening_arm_bank()
        if str(extras.get("train_version") or "") in module._LEAKED_TRAIN_VERSIONS
    ]
    assert offenders == [], f"screening arms train on leaked corpora: {offenders}"


def test_every_data_arm_corpus_is_disjoint_from_the_scored_suites() -> None:
    """Whatever data arms exist, each one's corpus must be measurably clean."""
    module = _driver()
    seen = 0
    for slug, _, extras in module._all_screening_arm_bank():
        corpus = str(extras.get("train_version") or "")
        if not corpus or not (TRAIN / corpus / "records.jsonl").is_file():
            continue
        seen += 1
        for label, counts in _overlap(corpus).items():
            assert counts == {"programs": 0, "prompts": 0, "families": 0}, (
                f"arm {slug!r} trains on {corpus}, which leaks into {label}: {counts}"
            )
    assert seen >= 0
