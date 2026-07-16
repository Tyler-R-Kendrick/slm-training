from __future__ import annotations

import json

from scripts.train_model import resolve_published_train_version


def test_published_train_version_resolves_canonical_mixture(tmp_path) -> None:
    version_dir = tmp_path / "v2"
    version_dir.mkdir()
    mixture = version_dir / "mixture.json"
    mixture.write_text(json.dumps({"mixture_id": "v2"}), encoding="utf-8")

    train_dir, resolved_mixture = resolve_published_train_version(
        "v2", root=tmp_path
    )

    assert train_dir == version_dir
    assert resolved_mixture == mixture


def test_published_train_version_allows_corpus_without_mixture(tmp_path) -> None:
    version_dir = tmp_path / "v2"
    version_dir.mkdir()

    train_dir, resolved_mixture = resolve_published_train_version(
        "v2", root=tmp_path
    )

    assert train_dir == version_dir
    assert resolved_mixture is None
