#!/usr/bin/env python3
"""VAR3-03 (SLM-433): train a real TurnDispositionHead and measure held-out benefit.

VAR3-02 shipped ``turn_disposition_losses``/``TurnDispositionReportV1`` as
untrained, synthetic-tensor-only surface. This script is the first run that
actually trains a classifier over it (``harnesses.experiments.
turn_disposition_training.TurnDispositionClassifierV1``) on a real corpus of
executed DSL operator applications (never hand-authored disposition labels
or synthetic scores) and reports its held-out benefit against two matched
always-emit baselines:

* ``disposition_off`` -- always attempts to emit (VAR3-02's own baseline).
* ``derived_only`` -- the free ``out_of_scope``/``answer``/forced-singleton
  rules only; ``clarify`` never fires.
* ``disposition_trained`` -- this script's trained classifier.

The corpus is a small, deterministic set of short conversation traces built
from executed operator applications (not hand-labeled), split disjointly by
trace so no source conversation ever appears in both train and dev (checked
by ``assert_leakage_safe_split``).

``claim_class: "wiring"``, not ``"capability"``: this fixture's competing
operators carry no per-row distinguishing argument content (no argument
slots at all), so every non-forced row shares one feature vector and one
gold label -- there is no genuine per-row disambiguation signal for the
classifier to learn beyond a constant policy. The run demonstrates the
wiring end to end (a real trained classifier, matched always-emit
baselines, a real trace-level held-out split, unchanged
``score_disposition_predictions`` scoring) but its ``disposition`` field
reports the honest outcome -- typically ``no_difference`` at this scale --
rather than a directional capability claim. Honesty label:
``fixture_or_scratch``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from slm_training.data.flow.turn_disposition_corpus import (
    assert_leakage_safe_split,
    build_turn_disposition_rows,
    corpus_action_frequency,
)
from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    CompilerCoverage,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    RegisteredOperatorV1,
    append_operator_turn,
    branch_fingerprint,
    build_reference_table,
    collapse_conversation_trace,
    create_conversation_trace,
)
from slm_training.dsl.operators.contracts import _fingerprint
from slm_training.dsl.pack import get_pack
from slm_training.harnesses.experiments.turn_disposition_training import (
    TurnDispositionClassifierV1,
    TurnDispositionExampleV1,
    evaluate_turn_disposition_arms,
    train_turn_disposition_head,
)
from slm_training.levers import (
    MAX_RUN_MINUTES,
    TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
    TURN_DISPOSITION_CLARIFY_TOP_K,
)
from slm_training.models.operator_feature_encoder import OperatorFeatureVocabularyV1
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/design/var3-03-turn-disposition-corpus-training-20260727.json"
SEED = 433
N_TRAIN_TRACES = 8
N_DEV_TRACES = 4
DIM = 8
STEPS = 30
LEARNING_RATE = 0.05


def _sha(value: str) -> str:
    return _fingerprint({"schema": "slm433_fixture/v1", "value": value})


def _source_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _provenance(state: OperatorStateV1, *, trace_label: str) -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.slm433_disposition_training_fixture",
        compiler_version="v1",
        source_artifact_digest=_source_digest(state.source),
        request_id=f"request-{trace_label}",
    )


def _table(state: OperatorStateV1, branch: str, *, seed: int, trace_label: str):
    return build_reference_table(
        request_id=f"request-{trace_label}",
        state_digest=state.state_digest,
        branch_digest=branch,
        descriptors=(),
        seed=seed,
    )


def _declaration(operator_id: str) -> AstOperatorV1:
    return AstOperatorV1(
        operator_id=operator_id,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(),
        preconditions=(),
        effect_signature=(),
        locality="node",
        cost=1.0,
    )


def _competing_library() -> OperatorLibraryV1:
    """Two operators, both unconditionally legal at every state -- every
    step of a trace built from this library has >= 2 real live candidates,
    exercising the scored clarify-vs-emit path on genuinely ambiguous data."""

    def execute_set_x(_state, _arguments):
        return OperatorMutationV1(
            source='root = TextContent(":set.x")',
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    def execute_set_y(_state, _arguments):
        return OperatorMutationV1(
            source='root = TextContent(":set.y")',
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    return OperatorLibraryV1(
        (
            RegisteredOperatorV1(_declaration("openui.set_x"), execute_set_x),
            RegisteredOperatorV1(_declaration("openui.set_y"), execute_set_y),
        )
    )


def _sequential_library() -> OperatorLibraryV1:
    """B requires A's effect, so exactly one operator is ever live at each
    step -- every row is a real forced-singleton emit (I1 bypass), giving
    the corpus genuine structural variety alongside the always-ambiguous
    competing library rather than one homogeneous case shape."""

    def execute_a(state, _arguments):
        if ":hero.title" not in state.source:
            raise OperatorRejectedError("fixture.a_precondition_failed")
        return OperatorMutationV1(
            source=state.source.replace(":hero.title", ":hero.body"),
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    def execute_b(state, _arguments):
        if ":hero.body" not in state.source:
            raise OperatorRejectedError("fixture.b_precondition_failed")
        return OperatorMutationV1(
            source=state.source.replace(":hero.body", ":hero.footer"),
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    return OperatorLibraryV1(
        (
            RegisteredOperatorV1(_declaration("openui.set_a"), execute_a),
            RegisteredOperatorV1(_declaration("openui.set_b"), execute_b),
        )
    )


def _build_trace_rows(*, trace_label: str, index: int, action_frequency, split: str):
    """Build one 2-turn trace and its disposition rows.

    Alternates between the always-ambiguous competing library (>= 2 live
    candidates every step) and the sequential-precondition library (exactly
    one live candidate every step) by trace index, so the corpus has real
    structural variety rather than one homogeneous case shape.
    ``trace_label`` seeds every id-bearing field (source digest, branch,
    reference-table seeds) so train and dev traces are distinct, real,
    deterministic conversations -- never the same source replayed twice.
    """
    if index % 2 == 0:
        library = _competing_library()
        first, second = "openui.set_x", "openui.set_y"
    else:
        library = _sequential_library()
        first, second = "openui.set_a", "openui.set_b"
    pack = replace(get_pack("openui"), operator_library=library)
    root_state = OperatorStateV1.from_source(pack, 'root = TextContent(":hero.title")')
    branch = branch_fingerprint(root_state.state_digest, _sha(f"branch:{trace_label}"))
    root_table = _table(root_state, branch, seed=index * 10, trace_label=trace_label)
    trace = create_conversation_trace(
        pack=pack,
        root_state=root_state,
        root_reference_table=root_table,
        provenance=_provenance(root_state, trace_label=trace_label),
    )

    def append(trace, operator_id, *, seed):
        state = trace.current.state
        result = library.apply(
            pack, state, operator_id, (), _provenance(state, trace_label=trace_label)
        )
        assert result.succeeded, result.application.rejection
        output_table = _table(result.state, trace.current.branch_digest, seed=seed, trace_label=trace_label)
        return append_operator_turn(
            trace,
            pack=pack,
            library=library,
            application=result.application,
            output_reference_table=output_table,
        )

    trace = append(trace, first, seed=index * 10 + 1)
    trace = append(trace, second, seed=index * 10 + 2)

    def authority_resolver(_node):
        return pack, library

    decision = collapse_conversation_trace(
        pack=pack, authority_resolver=authority_resolver, trace=trace
    )
    if decision.rejection is not None:
        raise AssertionError(f"fixture collapse rejected: {decision.rejection}")

    rows, rejected = build_turn_disposition_rows(
        trace=trace,
        collapse=decision.collapse,
        authority_resolver=authority_resolver,
        action_frequency=action_frequency,
        split=split,
        clarify_margin_threshold=TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
        clarify_top_k=TURN_DISPOSITION_CLARIFY_TOP_K,
    )
    return rows, rejected


def run_training(
    *,
    n_train_traces: int = N_TRAIN_TRACES,
    n_dev_traces: int = N_DEV_TRACES,
    dim: int = DIM,
    steps: int = STEPS,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
) -> dict[str, Any]:
    torch.manual_seed(seed)

    # Pass 1: build train rows with an unweighted (empty) frequency table --
    # the frequency statistic itself must come only from historically
    # accepted actions, so it cannot be computed before any row exists.
    train_rows: list = []
    train_rejected: list = []
    for index in range(n_train_traces):
        rows, rejected = _build_trace_rows(
            trace_label=f"train-{index}", index=index, action_frequency={}, split="train"
        )
        train_rows.extend(rows)
        train_rejected.extend(rejected)

    action_frequency = corpus_action_frequency(
        {row.trace_id: (row.accepted_action_serialized,) for row in train_rows}
    )

    # Pass 2: rebuild train rows using the train-only frequency table (the
    # score provider a trained margin classifier's baseline would actually
    # see), then build dev rows from wholly distinct source traces using
    # that same train-derived statistic -- never dev's own history.
    train_rows = []
    train_rejected = []
    for index in range(n_train_traces):
        rows, rejected = _build_trace_rows(
            trace_label=f"train-{index}",
            index=index,
            action_frequency=action_frequency,
            split="train",
        )
        train_rows.extend(rows)
        train_rejected.extend(rejected)

    dev_rows: list = []
    dev_rejected: list = []
    for index in range(n_dev_traces):
        rows, rejected = _build_trace_rows(
            trace_label=f"dev-{index}",
            index=index,
            action_frequency=action_frequency,
            split="dev",
        )
        dev_rows.extend(rows)
        dev_rejected.extend(rejected)

    assert_leakage_safe_split((*train_rows, *dev_rows))

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

    baseline_composite = dev_arms["disposition_off"]["composite_penalized_error_rate"]
    trained_composite = dev_arms["disposition_trained"]["composite_penalized_error_rate"]
    if trained_composite < baseline_composite:
        disposition = "held_out_improvement"
        note = (
            "The trained classifier's held-out composite_penalized_error_rate "
            f"({trained_composite:.4f}) is lower than the always-emit baselines' "
            f"({baseline_composite:.4f}). Directional only at this scale -- see "
            "claim_class."
        )
    elif trained_composite > baseline_composite:
        disposition = "held_out_regression"
        note = (
            "The trained classifier's held-out composite_penalized_error_rate "
            f"({trained_composite:.4f}) is worse than the always-emit baselines' "
            f"({baseline_composite:.4f}); a legitimate, honestly-reported negative "
            "result, not reframed."
        )
    else:
        disposition = "no_difference"
        note = (
            "The trained classifier's held-out predictions score identically to "
            "the always-emit baselines. This corpus's competing operators carry "
            "no per-row distinguishing argument content, so every non-forced row "
            "shares one feature vector and one gold label; the classifier "
            "necessarily collapses to a constant policy rather than learning a "
            "genuine per-row disambiguation. A no_difference result here is "
            "expected and legitimate, not reframed as inconclusive."
        )

    return {
        "schema": "var3_03_turn_disposition_training/v1",
        "honesty": "fixture_or_scratch",
        "claim_class": "wiring",
        "disposition": disposition,
        "note": note,
        "seed": seed,
        "clarify_margin_threshold": TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
        "clarify_top_k": TURN_DISPOSITION_CLARIFY_TOP_K,
        "max_wall_minutes": float(MAX_RUN_MINUTES),
        "recipe": {
            "n_train_traces": n_train_traces,
            "n_dev_traces": n_dev_traces,
            "dim": dim,
            "steps": steps,
            "learning_rate": learning_rate,
        },
        "corpus": {
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
        },
        "final_train_loss": loss_history[-1] if loss_history else None,
        "arms": {
            "train": train_arms,
            "held_out": dev_arms,
        },
        "version_stamp": build_version_stamp(
            "config.levers",
            "dsl.operators.turn_disposition",
            "data.flow.operator_policy_corpus",
            "data.flow.turn_disposition_corpus",
            "harness.experiments.turn_disposition_training",
            "evals.cap2_operator",
            "model.turn_disposition_head",
            "model.operator_feature_encoder",
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-train-traces", type=int, default=N_TRAIN_TRACES)
    parser.add_argument("--n-dev-traces", type=int, default=N_DEV_TRACES)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    report = run_training(
        n_train_traces=args.n_train_traces,
        n_dev_traces=args.n_dev_traces,
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
