#!/usr/bin/env python3
"""VAR3-04 (SLM-433): real argument-bound corpus for turn-disposition training.

VAR3-03 (``run_slm433_turn_disposition_training.py``) trained a real
``TurnDispositionClassifierV1`` end to end but reported ``no_difference``:
its two-operator competing library had zero argument slots, so every
non-forced row shared one feature vector and one gold label -- there was no
per-row distinguishing content for the classifier to learn beyond a
constant policy. That doc's own "What a real capability run would require"
section named the fix: "a corpus whose ambiguous turns carry genuine
per-row distinguishing content (e.g. operators with argument slots binding
to different reference-table candidates per turn...)".

This script is that corpus, built from already-canonical machinery rather
than new fixture operators: :func:`build_symbolic_operator_corpus`
(DSH3-24/SLM-399, already used by ``run_dsh3_28_typed_operator_policy.py``
and others) applies the real, production ``openui`` local operator library
-- genuine ``node``/``role``/``value`` argument slots -- to real admitted
train-root documents (``openui_verified_v1``) and a disjoint held-out
record set (``e763_symbol_only_eval_r2_20260722/suites/held_out``). Its
``on_collapsed_trace`` callback hands each real 2-turn
``ConversationTraceV1``/``CollapsedInstructionV1``/authority-resolver
triple straight into VAR3-03's own, unmodified
:func:`build_turn_disposition_rows`, :class:`TurnDispositionClassifierV1`,
:func:`train_turn_disposition_head`, and :func:`evaluate_turn_disposition_arms`
-- no new corpus-labeling or scoring code, only a new, richer source of
real rows.

Same honesty posture as VAR3-03: ``claim_class: "wiring"`` unless the held-out
result actually shows a directional benefit, in which case that is reported
exactly as measured, never manufactured. No promotion or ship claim either
way.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from slm_training.data.flow.turn_disposition_corpus import (
    TurnDispositionRowV1,
    assert_leakage_safe_split,
    build_turn_disposition_rows,
    corpus_action_frequency,
)
from slm_training.dsl.schema import load_jsonl
from slm_training.harnesses.experiments.turn_disposition_training import (
    TurnDispositionClassifierV1,
    TurnDispositionExampleV1,
    evaluate_turn_disposition_arms,
    train_turn_disposition_head,
)
from slm_training.harnesses.train_data.operator_corpus import (
    OperatorCorpusConfig,
    build_symbolic_operator_corpus,
)
from slm_training.levers import (
    MAX_RUN_MINUTES,
    TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
    TURN_DISPOSITION_CLARIFY_TOP_K,
)
from slm_training.models.operator_feature_encoder import OperatorFeatureVocabularyV1
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/design/var3-04-turn-disposition-real-corpus-20260727.json"
TRAIN_SOURCE = ROOT / "src/slm_training/resources/data/train/openui_verified_v1/records.jsonl"
DEV_SOURCE = (
    ROOT
    / "src/slm_training/resources/data/eval/e763_symbol_only_eval_r2_20260722"
    / "suites/held_out/records.jsonl"
)
SEED = 433_04
N_TRAIN_ROOTS = 8
N_DEV_ROOTS = 4
DIM = 8
STEPS = 30
LEARNING_RATE = 0.05
MAX_COMBINATIONS_PER_OPERATOR = 32


def _admit_raw_traces(
    *,
    source: Path,
    target_count: int,
    work_dir: Path,
    stamp: dict[str, Any],
) -> tuple[list[tuple[Any, Any, Any]], list[str]]:
    """Collect ``target_count`` real ``(trace, collapse, authority_resolver)``
    triples from ``source``'s admitted document roots, one
    :func:`build_symbolic_operator_corpus` call per candidate root.

    Not every admitted-document root's 2-turn extension survives replay +
    symbolic collapse (a small, real minority do not -- confirmed by probing
    this repo's own ``openui_verified_v1`` roots before writing this
    function); a root that fails is skipped, never retried or patched, and
    the next candidate (in deterministic sorted-id order) is tried instead.
    This keeps the corpus itself and its later frequency/training passes
    honest and reproducible: the expensive real build happens exactly once
    per admitted root, and skips are visible in the returned ``skipped`` list
    rather than silently absorbed by ``build_symbolic_operator_corpus``'s own
    fail-closed batch behavior (which raises on the first failing root in a
    multi-root batch call).
    """
    records = load_jsonl(source)
    documents = sorted(
        (record for record in records if record.target_kind == "document"),
        key=lambda record: record.id,
    )
    raw: list[tuple[Any, Any, Any]] = []
    skipped: list[str] = []
    for record in documents:
        if len(raw) >= target_count:
            break
        captured: list[tuple[Any, Any, Any]] = []

        def collect(trace, collapse, authority_resolver, _captured=captured) -> None:
            _captured.append((trace, collapse, authority_resolver))

        try:
            build_symbolic_operator_corpus(
                records=[record],
                output_dir=work_dir / record.id,
                version="var3-04",
                version_stamp=stamp,
                config=OperatorCorpusConfig(
                    max_roots=1,
                    max_combinations_per_operator=MAX_COMBINATIONS_PER_OPERATOR,
                    sibling_forks=False,
                ),
                on_collapsed_trace=collect,
            )
        except Exception:  # noqa: BLE001 -- a real per-root replay/collapse failure, not a bug
            skipped.append(record.id)
            continue
        raw.extend(captured)
    if len(raw) < target_count:
        raise ValueError(
            f"only {len(raw)} of {target_count} requested roots admitted from {source}"
        )
    return raw, skipped


def _rows_from_raw(
    raw: list[tuple[Any, Any, Any]],
    *,
    action_frequency: Any,
    split: str,
) -> tuple[list[TurnDispositionRowV1], list[dict[str, Any]]]:
    """Re-derive rows from already-built real traces -- no corpus rebuild."""
    rows: list[TurnDispositionRowV1] = []
    rejected: list[dict[str, Any]] = []
    for trace, collapse, authority_resolver in raw:
        built, dropped = build_turn_disposition_rows(
            trace=trace,
            collapse=collapse,
            authority_resolver=authority_resolver,
            action_frequency=action_frequency,
            split=split,
            max_combinations_per_operator=MAX_COMBINATIONS_PER_OPERATOR,
            clarify_margin_threshold=TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
            clarify_top_k=TURN_DISPOSITION_CLARIFY_TOP_K,
        )
        rows.extend(built)
        rejected.extend(dropped)
    return rows, rejected


def run_training(
    *,
    n_train_roots: int = N_TRAIN_ROOTS,
    n_dev_roots: int = N_DEV_ROOTS,
    dim: int = DIM,
    steps: int = STEPS,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    stamp = build_version_stamp(
        "config.levers",
        "dsl.operators.turn_disposition",
        "data.flow.turn_disposition_corpus",
        "harness.experiments.turn_disposition_training",
        "harness.train_data",
        "model.turn_disposition_head",
        "model.operator_feature_encoder",
    )

    with tempfile.TemporaryDirectory(prefix="var3-04-corpus-") as tmp:
        work_dir = Path(tmp)

        # Real corpus build happens exactly once per admitted root, for
        # both splits. Train and dev roots come from wholly distinct
        # source-record files (no shared root state between them), so
        # trace-level leakage is structurally impossible here -- verified
        # below regardless.
        train_raw, train_skipped_roots = _admit_raw_traces(
            source=TRAIN_SOURCE,
            target_count=n_train_roots,
            work_dir=work_dir / "train",
            stamp=stamp,
        )
        dev_raw, dev_skipped_roots = _admit_raw_traces(
            source=DEV_SOURCE,
            target_count=n_dev_roots,
            work_dir=work_dir / "dev",
            stamp=stamp,
        )

    # Pass 1: derive train rows with an unweighted (empty) frequency table
    # -- identical two-pass structure to VAR3-03: the frequency statistic
    # itself must come only from historically accepted actions, so it
    # cannot be computed before any row exists. Cheap: re-derives rows from
    # the already-built real traces above, no corpus rebuild.
    train_rows, train_rejected = _rows_from_raw(train_raw, action_frequency={}, split="train")
    action_frequency = corpus_action_frequency(
        {row.trace_id: (row.accepted_action_serialized,) for row in train_rows}
    )

    # Pass 2: re-derive train rows using the train-only frequency table,
    # then derive dev rows using that same train-derived statistic --
    # never dev's own history.
    train_rows, train_rejected = _rows_from_raw(
        train_raw, action_frequency=action_frequency, split="train"
    )
    dev_rows, dev_rejected = _rows_from_raw(dev_raw, action_frequency=action_frequency, split="dev")

    assert_leakage_safe_split((*train_rows, *dev_rows))
    if not train_rows or not dev_rows:
        raise ValueError("real corpus produced no rows for at least one split")

    train_examples = [TurnDispositionExampleV1.from_row(row) for row in train_rows]
    dev_examples = [TurnDispositionExampleV1.from_row(row) for row in dev_rows]

    vocabulary = OperatorFeatureVocabularyV1.from_inputs(
        [example.view for example in train_examples]
    )
    classifier = TurnDispositionClassifierV1(vocabulary, dim=dim)
    loss_history = train_turn_disposition_head(
        classifier, train_examples, steps=steps, learning_rate=learning_rate
    )

    train_arms = evaluate_turn_disposition_arms(classifier, train_examples)
    dev_arms = evaluate_turn_disposition_arms(classifier, dev_examples)

    # Real evidence this corpus is not the VAR3-03 single-feature-vector
    # degenerate case: distinct (value_type, ref_kind) reference-row
    # signatures actually observed across train rows.
    distinct_reference_signatures = {
        (row.ref_kind.value, row.value_type)
        for example in train_examples
        for row in example.view.reference_rows
    }
    distinct_action_operator_ids = {
        action.operator_id for example in train_examples for action in example.view.action_rows
    }

    baseline_composite = dev_arms["disposition_off"]["composite_penalized_error_rate"]
    trained_composite = dev_arms["disposition_trained"]["composite_penalized_error_rate"]
    if trained_composite < baseline_composite:
        disposition = "held_out_improvement"
        note = (
            "The trained classifier's held-out composite_penalized_error_rate "
            f"({trained_composite:.4f}) is lower than the always-emit baselines' "
            f"({baseline_composite:.4f}) on this real, argument-bound corpus. "
            "Directional only at this scale (n_dev={} rows) -- see claim_class."
        ).format(len(dev_rows))
    elif trained_composite > baseline_composite:
        disposition = "held_out_regression"
        note = (
            "The trained classifier's held-out composite_penalized_error_rate "
            f"({trained_composite:.4f}) is worse than the always-emit baselines' "
            f"({baseline_composite:.4f}); a legitimate, honestly-reported "
            "negative result, not reframed."
        )
    else:
        disposition = "no_difference"
        note = (
            "The trained classifier's held-out predictions score identically to "
            "the always-emit baselines, even though this corpus's reference rows "
            f"carry {len(distinct_reference_signatures)} distinct (ref_kind, "
            "value_type) signatures and its action rows carry "
            f"{len(distinct_action_operator_ids)} distinct operator ids across "
            "real argument-bound applications -- so per-row distinguishing "
            "content genuinely exists here, unlike VAR3-03's degenerate case. "
            "A no_difference result under real per-row variety is still a "
            "legitimate negative at this held-out scale (n_dev="
            f"{len(dev_rows)} rows), not reframed as inconclusive."
        )

    return {
        "schema": "var3_04_turn_disposition_real_corpus/v1",
        "honesty": "fixture_or_scratch",
        "claim_class": "wiring",
        "disposition": disposition,
        "note": note,
        "seed": seed,
        "clarify_margin_threshold": TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
        "clarify_top_k": TURN_DISPOSITION_CLARIFY_TOP_K,
        "max_wall_minutes": float(MAX_RUN_MINUTES),
        "recipe": {
            "n_train_roots": n_train_roots,
            "n_dev_roots": n_dev_roots,
            "dim": dim,
            "steps": steps,
            "learning_rate": learning_rate,
        },
        "sources": {
            "train_source": str(TRAIN_SOURCE.relative_to(ROOT)),
            "dev_source": str(DEV_SOURCE.relative_to(ROOT)),
        },
        "corpus": {
            "train_roots_admitted": len(train_raw),
            "train_roots_skipped": train_skipped_roots,
            "dev_roots_admitted": len(dev_raw),
            "dev_roots_skipped": dev_skipped_roots,
            "train_rows": len(train_rows),
            "train_rejected": len(train_rejected),
            "dev_rows": len(dev_rows),
            "dev_rejected": len(dev_rejected),
            "train_forced_singleton_rows": sum(
                1 for row in train_rows if row.target.forced_singleton
            ),
            "dev_forced_singleton_rows": sum(
                1 for row in dev_rows if row.target.forced_singleton
            ),
            "distinct_reference_signatures": len(distinct_reference_signatures),
            "distinct_action_operator_ids": len(distinct_action_operator_ids),
        },
        "final_train_loss": loss_history[-1] if loss_history else None,
        "arms": {
            "train": train_arms,
            "held_out": dev_arms,
        },
        "version_stamp": stamp,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-train-roots", type=int, default=N_TRAIN_ROOTS)
    parser.add_argument("--n-dev-roots", type=int, default=N_DEV_ROOTS)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    report = run_training(
        n_train_roots=args.n_train_roots,
        n_dev_roots=args.n_dev_roots,
        dim=args.dim,
        steps=args.steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
