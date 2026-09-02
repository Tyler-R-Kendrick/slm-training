"""Decode invariant I3: deterministic speculative ranking over symbol tables.

Ranking is a lever; legality is not. Every test here checks one of those two
halves — that the scorer orders sensibly and reproducibly, and that it can
never add a candidate the completion forest did not already prove legal.
"""

from __future__ import annotations

import json

import pytest

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    CompletionPath,
)
from slm_training.dsl.grammar.fastpath.speculative_rank import (
    NGRAM_TABLE_SCHEMA,
    NgramTableV1,
    SpeculativeRankerV1,
    build_ngram_table,
    load_ranker,
    speculative_span,
)


def _table(order: int = 3) -> NgramTableV1:
    # "1 2 3" is the corpus habit; "1 2 9" appears once.
    return build_ngram_table([[1, 2, 3]] * 8 + [[1, 2, 9]], order=order)


def _forest(paths: list[list[int]], coverage: str = "complete") -> CompletionForest:
    return CompletionForest(
        paths=tuple(CompletionPath(token_ids=tuple(p), kind="action") for p in paths),
        coverage=coverage,  # type: ignore[arg-type]
    )


def test_ngram_table_prefers_the_corpus_continuation() -> None:
    table = _table()
    assert table.log_prob([1, 2], 3) > table.log_prob([1, 2], 9)


def test_unseen_continuation_scores_finite_not_negative_infinity() -> None:
    """A total order needs finite scores, or margins stop being comparable."""
    table = _table()
    score = table.log_prob([1, 2], 4242)
    assert score == pytest.approx(score)  # not nan
    assert score > float("-inf")


def test_backoff_reaches_the_unigram_level() -> None:
    table = _table()
    # Context never observed: backoff must still separate a frequent token
    # from one that never appears at all.
    assert table.log_prob([777, 888], 3) > table.log_prob([777, 888], 4242)


def test_table_round_trips_through_json() -> None:
    table = _table()
    restored = NgramTableV1.from_dict(json.loads(json.dumps(table.to_dict())))
    assert restored.order == table.order
    assert restored.corpus_fingerprint == table.corpus_fingerprint
    assert restored.log_prob([1, 2], 3) == pytest.approx(table.log_prob([1, 2], 3))


def test_table_write_and_load_round_trip(tmp_path) -> None:
    table = _table()
    path = table.write(tmp_path / "nested" / "table.json")
    assert json.loads(path.read_text())["schema"] == NGRAM_TABLE_SCHEMA
    assert NgramTableV1.load(path).corpus_fingerprint == table.corpus_fingerprint


def test_ranker_orders_legal_paths_without_changing_membership() -> None:
    ranker = SpeculativeRankerV1(table=_table())
    paths = (
        CompletionPath(token_ids=(9,), kind="a"),
        CompletionPath(token_ids=(3,), kind="b"),
    )
    choice = ranker.choose([1, 2], paths)
    assert choice is not None
    assert choice.best_index == 1  # token 3 is the corpus habit
    scored = ranker.score_paths([1, 2], paths)
    assert set(scored) == {(9,), (3,)}


def test_ranking_is_reproducible_regardless_of_input_order() -> None:
    ranker = SpeculativeRankerV1(table=_table())
    a = CompletionPath(token_ids=(3,), kind="a")
    b = CompletionPath(token_ids=(9,), kind="b")
    forward = ranker.choose([1, 2], (a, b))
    reverse = ranker.choose([1, 2], (b, a))
    assert forward is not None and reverse is not None
    assert forward.scores[forward.best_index] == pytest.approx(
        reverse.scores[reverse.best_index]
    )


def test_zero_margin_never_commits_without_the_model() -> None:
    """The safe default: supply an ordering, let the model decide."""
    ranker = SpeculativeRankerV1(table=_table(), margin=0.0)
    choice = ranker.choose(
        [1, 2],
        (
            CompletionPath(token_ids=(3,), kind="a"),
            CompletionPath(token_ids=(9,), kind="b"),
        ),
    )
    assert choice is not None
    assert not choice.confident


def test_confident_margin_commits() -> None:
    ranker = SpeculativeRankerV1(table=_table(), margin=0.5)
    choice = ranker.choose(
        [1, 2],
        (
            CompletionPath(token_ids=(3,), kind="a"),
            CompletionPath(token_ids=(9,), kind="b"),
        ),
    )
    assert choice is not None
    assert choice.confident
    assert choice.margin >= 0.5


def test_span_commits_forced_singletons_with_no_ranker() -> None:
    """A singleton domain needs no scorer and no forward — I2 inside a span."""
    sequence = {0: [[7]], 1: [[8]], 2: [[9]]}

    def build(prefix: list[int]) -> CompletionForest:
        return _forest(sequence[len(prefix)])

    span = speculative_span([], build, max_tokens=3)
    assert span.token_ids == (7, 8, 9)
    assert span.forced_tokens == 3
    assert span.ranked_tokens == 0


def test_span_stops_at_a_branch_point_without_a_ranker() -> None:
    def build(prefix: list[int]) -> CompletionForest:
        return _forest([[7]]) if not prefix else _forest([[8], [9]])

    span = speculative_span([], build, max_tokens=4)
    assert span.token_ids == (7,)
    assert span.stop_reason == "no_ranker"


def test_span_refuses_to_speculate_past_an_incomplete_proof() -> None:
    """A partial completion forest is not a proof; speculation must stop."""

    def build(prefix: list[int]) -> CompletionForest:
        return _forest([[7]]) if not prefix else _forest([[8]], coverage="partial")

    span = speculative_span(
        [], build, max_tokens=4, ranker=SpeculativeRankerV1(_table())
    )
    assert span.token_ids == (7,)
    assert span.stop_reason == "incomplete_coverage"


def test_span_stops_on_an_empty_domain_rather_than_inventing_a_token() -> None:
    def build(prefix: list[int]) -> CompletionForest:
        return _forest([[7]]) if not prefix else _forest([])

    span = speculative_span([], build, max_tokens=4)
    assert span.token_ids == (7,)
    assert span.stop_reason == "empty_domain"


def test_span_only_ever_emits_tokens_the_forest_offered() -> None:
    """The legality half: nothing in a span came from outside the domain."""
    domains = {0: [[1]], 1: [[2]], 2: [[3], [9]], 3: [[5]]}

    def build(prefix: list[int]) -> CompletionForest:
        return _forest(domains[len(prefix)])

    span = speculative_span(
        [], build, ranker=SpeculativeRankerV1(_table(), margin=0.1), max_tokens=4
    )
    emitted = list(span.token_ids)
    for index, token in enumerate(emitted):
        offered = {path[0] for path in domains[index]}
        assert token in offered


def test_span_respects_its_token_budget() -> None:
    def build(prefix: list[int]) -> CompletionForest:
        return _forest([[1, 1, 1]])

    span = speculative_span([], build, max_tokens=2)
    assert span.token_ids == ()
    assert span.stop_reason == "budget"


def test_load_ranker_falls_back_to_the_committed_table() -> None:
    """An unnamed table is the committed default, not "no ranker"."""
    ranker = load_ranker(None)
    assert ranker is not None
    assert ranker.table.sequences > 0


def test_load_ranker_rejects_an_empty_table(tmp_path) -> None:
    """A run must be able to name the table that ranked it."""
    path = tmp_path / "empty.json"
    build_ngram_table([], order=2).write(path)
    with pytest.raises(ValueError, match="empty"):
        load_ranker(path)


def test_unsupported_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="schema"):
        NgramTableV1.from_dict({"schema": "something/v9", "order": 2, "counts": []})


def test_committed_table_is_loadable_and_train_only() -> None:
    """I3 is reachable without a build step: the default table ships."""
    from slm_training.dsl.grammar.fastpath.speculative_rank import (
        COMMITTED_NGRAM_TABLE,
    )

    assert COMMITTED_NGRAM_TABLE.is_file()
    ranker = load_ranker(None)
    assert ranker is not None
    assert ranker.table.order >= 2
    assert ranker.table.sequences > 0
    assert ranker.table.corpus_fingerprint


def test_committed_table_matches_its_builder() -> None:
    """A stale artifact would silently rank against a corpus nobody has."""
    from scripts.build_speculative_ngram_table import main as build

    assert build(["--check"]) == 0


def test_ranker_beats_uniform_on_known_continuations() -> None:
    """Better-than-uniform: the corpus continuation outranks the rare sibling."""
    ranker = SpeculativeRankerV1(table=_table())
    paths = (
        CompletionPath(token_ids=(9,), kind="rare"),
        CompletionPath(token_ids=(3,), kind="habit"),
    )
    choice = ranker.choose([1, 2], paths)
    assert choice is not None
    scores = choice.scores
    assert scores[choice.best_index] > sum(scores) / len(scores)
    assert paths[choice.best_index].token_ids == (3,)


def test_domain_restricted_backoff_does_not_let_leaf_unigrams_win() -> None:
    """c527 failure mode: TextContent unigrams beat a rare-but-contextual container."""
    table = build_ngram_table(
        [[1, 2, 7]] * 4 + [[9]] * 40,  # 9 is a frequent leaf; 7 follows 1,2
        order=3,
    )
    ranker = SpeculativeRankerV1(table=table)
    choice = ranker.choose(
        [1, 2],
        (
            CompletionPath(token_ids=(9,), kind="leaf"),
            CompletionPath(token_ids=(7,), kind="container"),
        ),
    )
    assert choice is not None
    assert choice.order[0] == 1  # token 7


def test_builder_rejects_eval_manifest(tmp_path) -> None:
    from scripts.build_speculative_ngram_table import main as build

    eval_dir = tmp_path / "eval_suite"
    eval_dir.mkdir()
    (eval_dir / "manifest.json").write_text(
        json.dumps({"kind": "eval", "version": "leak"}), encoding="utf-8"
    )
    (eval_dir / "records.jsonl").write_text(
        json.dumps(
            {
                "id": "smoke_x",
                "prompt": "p",
                "openui": 'root = TextContent(":slot_0")',
                "split": "smoke",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="eval"):
        build(["--records", str(eval_dir), "--output", str(tmp_path / "out.json")])


def test_builder_is_deterministic_from_a_fixed_manifest(tmp_path) -> None:
    from scripts.build_speculative_ngram_table import main as build

    src = tmp_path / "train"
    src.mkdir()
    (src / "manifest.json").write_text(
        json.dumps({"kind": "train", "version": "fixture"}), encoding="utf-8"
    )
    program = 'root = Stack([v0])\nv0 = TextContent(":slot_0")\n'
    line = json.dumps(
        {
            "id": "train_a",
            "prompt": "p",
            "openui": program,
            "split": "train",
        }
    )
    (src / "records.jsonl").write_text(line + "\n" + line + "\n", encoding="utf-8")
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    assert build(["--records", str(src), "--output", str(a), "--order", "3"]) == 0
    assert build(["--records", str(src), "--output", str(b), "--order", "3"]) == 0
    assert json.loads(a.read_text()) == json.loads(b.read_text())


def test_committed_table_ranks_real_branch_points_confidently() -> None:
    """The point of I3: pick the next-most-likely legal symbol, no forward."""
    from slm_training.dsl.grammar.fastpath.compiler_draft import (
        build_completion_forest,
    )
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    ranker = load_ranker(None, margin=0.5)
    assert ranker is not None

    prefix = list(tok.encode("root = ", add_special=False))
    forest = build_completion_forest(tok, prefix, remaining_tokens=32)
    paths = tuple(path for path in forest.paths if path.token_ids)
    assert forest.coverage == "complete"
    assert len(paths) > 1  # a genuine branch point, not a forced singleton

    choice = ranker.choose(prefix, paths)
    assert choice is not None
    assert choice.confident
    # The chosen path is one the forest offered — ranking never widens.
    assert paths[choice.best_index] in paths


# The two branch points `docs/design/decode-invariants.md` (§ I3, "The
# committed table") quotes. These are measured from the committed artifact, not
# typed: `iter-s2-ngram-table-provenance-20260902.md` records the run. A drift
# here means the doc, the artifact, or the builder changed and the other two
# did not follow.
DOCUMENTED_BRANCH_POINTS = {
    "root = ": (27, "Stack(", 15.000),
    "root = Stack([": (26, "b1", 1.738),
}


def test_committed_table_pins_the_documented_branch_points() -> None:
    from slm_training.dsl.grammar.fastpath.compiler_draft import (
        build_completion_forest,
    )
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    ranker = load_ranker(None, margin=0.5)
    assert ranker is not None
    for text, (n_candidates, pick, margin) in DOCUMENTED_BRANCH_POINTS.items():
        prefix = list(tok.encode(text, add_special=False))
        forest = build_completion_forest(tok, prefix, remaining_tokens=32)
        paths = tuple(path for path in forest.paths if path.token_ids)
        assert forest.coverage == "complete", text
        assert len(paths) == n_candidates, text
        choice = ranker.choose(prefix, paths)
        assert choice is not None and choice.confident, text
        assert tok.decode(list(paths[choice.best_index].token_ids)) == pick, text
        assert choice.margin == pytest.approx(margin, abs=0.01), text


def test_committed_table_records_the_certified_train_bucket() -> None:
    """The artifact names the corpus it was built from, and that corpus is the
    certified TRAIN bucket -- the numbers the doc quotes follow from it."""
    from scripts.build_speculative_ngram_table import CERTIFIED_TRAIN_BUCKET
    from slm_training.dsl.grammar.fastpath.speculative_rank import (
        COMMITTED_NGRAM_TABLE,
    )

    payload = json.loads(COMMITTED_NGRAM_TABLE.read_text(encoding="utf-8"))
    manifest = json.loads(
        (CERTIFIED_TRAIN_BUCKET.parent / "manifest.json").read_text(encoding="utf-8")
    )
    certified_sha = next(
        item["sha256"]
        for item in manifest["artifacts"]
        if item["path"] == "records.jsonl"
    )
    assert manifest["kind"] == "train"
    assert payload["source"] == {
        "dataset_id": "openui_verified_train_v1",
        "manifest_content_fingerprint": manifest["content_fingerprint"],
        "manifest_kind": "train",
        "records": "src/slm_training/resources/data/train/openui_verified_train_v1/records.jsonl",
        "records_sha256": certified_sha,
        "records_total": manifest["record_count"],
    }
    assert payload["order"] == 3
    assert (payload["sequences"], payload["tokens"], len(payload["counts"])) == (
        1054,
        54434,
        493,
    )
    # The loader tolerates the provenance block: the ranker sees the same table.
    assert (
        NgramTableV1.load(COMMITTED_NGRAM_TABLE).corpus_fingerprint
        == (payload["corpus_fingerprint"])
    )


def test_check_fails_when_the_recorded_source_drifts_from_the_manifest(
    tmp_path,
) -> None:
    """`--check` binds the artifact to the corpus manifest, not only to a rebuild."""
    from scripts.build_speculative_ngram_table import main as build

    src = tmp_path / "train"
    src.mkdir()
    manifest = {"kind": "train", "version": "fixture", "content_fingerprint": "a" * 64}
    (src / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    line = json.dumps(
        {
            "id": "train_a",
            "prompt": "p",
            "openui": 'root = Stack([v0])\nv0 = TextContent(":slot_0")\n',
            "split": "train",
        }
    )
    (src / "records.jsonl").write_text(line + "\n", encoding="utf-8")
    out = tmp_path / "table.json"
    records = src / "records.jsonl"
    assert build(["--records", str(records), "--output", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["source"]["dataset_id"] == "fixture"
    assert payload["source"]["manifest_content_fingerprint"] == "a" * 64
    assert build(["--records", str(records), "--output", str(out), "--check"]) == 0

    # The corpus is re-certified under a new manifest fingerprint: the same
    # counts no longer prove the artifact came from the corpus the docs claim.
    manifest["content_fingerprint"] = "b" * 64
    (src / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert build(["--records", str(records), "--output", str(out), "--check"]) == 1

    # And a tampered records file is refused outright when the manifest
    # certifies its sha256.
    manifest["content_fingerprint"] = "a" * 64
    manifest["artifacts"] = [{"path": "records.jsonl", "sha256": "0" * 64}]
    (src / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit):
        build(["--records", str(records), "--output", str(out), "--check"])
