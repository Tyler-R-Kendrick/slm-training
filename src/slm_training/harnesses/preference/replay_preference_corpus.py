"""SLM-418 (DSH5-10) seventh slice: write the bounded replay-preference corpus
to real ``PreferencePair`` corpus files, using this repo's existing
conversion functions and corpus root -- no second corpus shape or shadow
path.

The sixth slice (v7) added :func:`~slm_training.dsl.operators.
replay_preference.preference_pair_from_replay_row` and
:func:`~slm_training.dsl.operators.replay_preference.
preference_pairs_from_trace`, which convert an already-extracted
``OperatorReplayPreferenceRowV1`` into this repo's existing ``PreferencePair``
shape, but never wrote a pair to a corpus file or fed one into the real
``slm preference train``/``train-local`` harness (see that slice's own
"Explicitly still not attempted" note). This module closes exactly that gap
for the bounded synthetic corpus
(:func:`~slm_training.harnesses.preference.
replay_preference_context_view_variants.synthesize_bounded_session_corpus`):

* :func:`replay_preference_pairs_for_split` converts every row in every
  session of one split (``"train"`` or ``"held_out"``) into a
  ``PreferencePair``, reusing ``preference_pairs_from_trace`` for every
  session that carries a real ``ConversationTraceV1`` and
  ``preference_pair_from_replay_row`` directly for ``merge_success`` (whose
  row is grounded on a ``BranchEditV1`` tip, not any one trace) -- the exact
  same trace/direct-path convention the sixth slice's own tests exercise, now
  applied uniformly across a whole corpus instead of one hand-built trace.
* :func:`write_replay_preference_corpus` writes those pairs to this repo's
  existing preference-pairs corpus root (``outputs/data/preference/`` -- the
  same root ``slm preference build-pairs --out`` already writes into) via the
  existing ``write_pairs`` writer. No second corpus tree is invented.

Held-out pairs are written to their own file and are never mixed into the
train file -- this repo's isolation/provenance law ("never fit training data
to holdouts"). See docs/design/dsh5-10-replay-preference-rows.md's "Seventh
slice" for what real training run and held-out measurement this corpus feeds,
and for what is still explicitly out of scope (the DSH3-selected
``TypedOperatorPolicyScorer``, the four-baseline comparison, CAP0/CAP1/CAP2
retention).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from slm_training.dsl.operators.replay_preference import (
    preference_pair_from_replay_row,
    preference_pairs_from_trace,
)
from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.preference import PreferencePair, write_pairs
from slm_training.harnesses.preference.local_decisions import Split
from slm_training.harnesses.preference.replay_preference_context_view_variants import (
    ReplayPreferenceSessionV1,
    synthesize_bounded_session_corpus,
)

#: Canonical preference-pairs corpus root this repo already uses (the same
#: root ``slm preference build-pairs --out`` writes into) -- never a second
#: corpus tree.
DEFAULT_TRAIN_OUT_PATH = Path("outputs/data/preference/replay_preference_train_pairs.jsonl")
DEFAULT_HELD_OUT_OUT_PATH = Path(
    "outputs/data/preference/replay_preference_held_out_pairs.jsonl"
)


def replay_preference_pairs_for_split(
    sessions: Sequence[ReplayPreferenceSessionV1], split: Split
) -> tuple[tuple[PreferencePair, ...], tuple[dict, ...]]:
    """Convert every row of every ``split`` session into a ``PreferencePair``.

    Returns ``(pairs, skipped)``. Sessions carrying a real
    ``ConversationTraceV1`` (every relation but ``merge_success``) go through
    ``preference_pairs_from_trace`` unchanged; a session with no trace (only
    ``merge_success`` today) converts each row directly via
    ``preference_pair_from_replay_row``, honestly skipping -- never
    fabricating a prompt for -- a row whose ``input_state_id`` is missing
    from that session's own ``state_lookup``. This mirrors, at corpus scale,
    the exact skip/direct-path convention the sixth slice's tests already
    prove at single-trace scale.
    """
    pairs: list[PreferencePair] = []
    skipped: list[dict] = []
    for session in sessions:
        if session.split != split:
            continue
        if session.trace is not None:
            session_pairs, session_skipped = preference_pairs_from_trace(
                session.trace, session.report.rows
            )
            pairs.extend(session_pairs)
            skipped.extend(dict(entry) for entry in session_skipped)
            continue
        for row in session.report.rows:
            node = session.state_lookup.get(row.input_state_id)
            if node is None:
                skipped.append(
                    {
                        "row": row.to_dict(),
                        "reason": (
                            f"input_state_id {row.input_state_id!r} not found in "
                            f"session {session.group_id!r}'s state_lookup"
                        ),
                    }
                )
                continue
            pairs.append(
                preference_pair_from_replay_row(row, input_state_source=node.state.source)
            )
    return tuple(pairs), tuple(skipped)


@dataclass(frozen=True)
class ReplayPreferenceCorpusBuildReportV1:
    """Honest counts for one written replay-preference corpus split file."""

    split: Split
    out_path: str
    pair_count: int
    skipped_count: int
    session_count: int
    row_count: int
    skipped: tuple[dict, ...]
    version_stamp: dict
    schema: str = "replay_preference_corpus_build_report/v1"

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "split": self.split,
            "out_path": self.out_path,
            "pair_count": self.pair_count,
            "skipped_count": self.skipped_count,
            "session_count": self.session_count,
            "row_count": self.row_count,
            "skipped": list(self.skipped),
            "version_stamp": self.version_stamp,
        }


def write_replay_preference_corpus(
    out_path: Path | str,
    split: Split,
    *,
    sessions: Sequence[ReplayPreferenceSessionV1] | None = None,
) -> ReplayPreferenceCorpusBuildReportV1:
    """Build the bounded synthetic corpus and write one ``split`` to ``out_path``.

    Writes into this repo's existing preference-pairs corpus root via the
    existing ``write_pairs`` writer -- never a second corpus tree. Raises
    ``ValueError`` (fails closed) rather than writing an empty corpus file
    when a split produces zero pairs.
    """
    resolved_sessions = (
        tuple(sessions) if sessions is not None else synthesize_bounded_session_corpus()
    )
    pairs, skipped = replay_preference_pairs_for_split(resolved_sessions, split)
    if not pairs:
        raise ValueError(
            f"replay-preference corpus build produced zero {split!r} pairs; "
            "refusing to write an empty corpus file"
        )
    out = Path(out_path)
    write_pairs(out, list(pairs))
    split_sessions = [session for session in resolved_sessions if session.split == split]
    return ReplayPreferenceCorpusBuildReportV1(
        split=split,
        out_path=str(out),
        pair_count=len(pairs),
        skipped_count=len(skipped),
        session_count=len(split_sessions),
        row_count=sum(len(session.report.rows) for session in split_sessions),
        skipped=skipped,
        version_stamp=build_version_stamp("harness.preference.replay_preference_corpus"),
    )
