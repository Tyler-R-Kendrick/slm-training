"""Prove a fresh OpenUI reference context can continue a branch merge locally."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.dsl.operators import (
    ApplicationProvenanceV1,
    ActionEffectV1,
    AstOperatorV1,
    BranchEditV1,
    CompilerCoverage,
    ConversationStateNodeV1,
    EffectDeltaKind,
    EffectDeltaV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RefKind,
    ReferenceResolutionError,
    RegisteredOperatorV1,
    append_operator_turn,
    branch_fingerprint,
    build_openui_topology_operator_context,
    build_openui_topology_operator_library,
    clone_reference_table_for_branch,
    create_conversation_trace,
    enumerate_operator_legal_set,
    merge_conversation_branches,
    merged_continuation_node,
    replay_branch_merge,
    replay_conversation_trace,
)
from slm_training.dsl.pack import DslPack, get_pack
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.versioning import build_version_stamp

SOURCE = (
    'root = Card([Stack([TextContent(":source.left"), '
    'TextContent(":source.right")]), Stack([])], "clear")'
)
VALUES = ("clear", "column", "row", "updated", "final")
REQUEST_ID = "slm395-audit"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(state: OperatorStateV1) -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.post_merge_continuation_audit",
        compiler_version="v1",
        source_artifact_digest=_sha(state.source),
        request_id=REQUEST_ID,
    )


def _context(
    pack: DslPack, state: OperatorStateV1, branch_digest: str, seed: int
):
    return build_openui_topology_operator_context(
        pack,
        state,
        request_id=REQUEST_ID,
        branch_digest=branch_digest,
        seed=seed,
        values=VALUES,
    )


def _continuation_table(
    pack: DslPack,
    state: OperatorStateV1,
    branch_digest: str,
    seed: int,
):
    """Canonical compiler rebuild; no branch table is cloned or patched."""
    return _context(pack, state, branch_digest, seed).local.reference_table


def _fixture_branch_edit(
    *,
    base_pack: DslPack,
    base_node: ConversationStateNodeV1,
    branch_digest: str,
    seed: int,
    replacement: tuple[str, str],
    target_index: int,
) -> tuple[BranchEditV1, tuple[DslPack, OperatorLibraryV1]]:
    # A branch starts from the one permitted fork transform. The fixture
    # executor has no arguments, so it needs no private compiler payloads.
    table = clone_reference_table_for_branch(
        base_node.reference_table, branch_digest=branch_digest, seed=seed
    )
    target = tuple(
        entry.ref
        for entry in table.entries
        if entry.descriptor.ref_kind is RefKind.NODE
    )[target_index]
    declaration = AstOperatorV1(
        operator_id=f"openui.fixture_branch_{target_index}",
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(),
        preconditions=(),
        effect_signature=(EffectDeltaKind.PROPERTY,),
        locality="node",
        cost=1.0,
    )

    def execute(
        state: OperatorStateV1, _arguments: tuple[Any, ...]
    ) -> OperatorMutationV1:
        return OperatorMutationV1(
            source=state.source.replace(*replacement),
            effect=ActionEffectV1(
                property_deltas=(
                    EffectDeltaV1(
                        kind=EffectDeltaKind.PROPERTY,
                        target=target,
                        before=replacement[0],
                        after=replacement[1],
                    ),
                ),
                compiler_coverage=CompilerCoverage.EXACT,
            ),
        )

    library = OperatorLibraryV1((RegisteredOperatorV1(declaration, execute),))
    pack = replace(base_pack, operator_library=library)
    input_node = ConversationStateNodeV1(
        parent_state_id=base_node.state_id,
        branch_digest=branch_digest,
        state=base_node.state,
        reference_table=table,
    )
    applied = library.apply(
        pack,
        input_node.state,
        declaration.operator_id,
        (),
        _provenance(input_node.state),
    )
    if not applied.succeeded or applied.state is None:
        raise RuntimeError("fixture branch edit did not apply")
    output = _context(pack, applied.state, branch_digest, seed + 1)
    return (
        BranchEditV1(
            input_node=input_node,
            output_node=ConversationStateNodeV1(
                parent_state_id=input_node.state_id,
                branch_digest=branch_digest,
                state=applied.state,
                reference_table=output.local.reference_table,
            ),
            application=applied.application,
        ),
        (pack, library),
    )


def _portable(value: Any, output_dir: Path) -> Any:
    prefix = str(output_dir.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "output-dir://" + value[len(prefix) :].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    prefixes = (
        (str(output_dir.resolve()), "output-dir://"),
        (str(Path(__file__).resolve().parents[1]), "repo-root://"),
    )
    for path in (output_dir / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for raw, portable in prefixes:
            text = text.replace(quote(raw, safe=""), quote(portable, safe=""))
            text = text.replace(raw, portable)
        path.write_text(text, encoding="utf-8")


def _write_summary(output_dir: Path, report: dict[str, Any]) -> None:
    audit = report["audit"]
    lines = [
        "# SLM-395 post-merge continuation audit",
        "",
        "Status: fixture wiring evidence passed; not a model or ship claim.",
        "",
        "The audit creates two branch-local reference clones and applies exact",
        "fixture edits to find one supported disjoint merge locally, then rebuilds the merged",
        "reference table through the canonical topology context. It enumerates a",
        "fresh legal set, appends one follow-up turn, and replays that trace.",
        "",
        "## Exact local result",
        "",
        f"- branch candidates considered: `{audit['branch_candidate_pairs']}`",
        f"- selected merge: `{audit['merge_id']}`",
        f"- fresh table fingerprint: `{audit['reference_table_fingerprint']}`",
        f"- continuation trace turns: `{audit['trace_turns']}`",
        f"- legal-set coverage: `{audit['followup_legal_set_coverage']}`",
        "",
        "Old branch tables are explicitly refused as stale or cross-branch. A",
        "different allocation seed preserves semantic membership while changing",
        "the opaque table fingerprint. Legacy `branch_merge/v1` artifacts remain",
        "terminal: only `branch_merge_continuation/v1` can start a new trace.",
        "",
        "This fixture proves wiring and replay only; it is not a training,",
        "evaluation, checkpoint, or ship-grade result.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-combinations", type=int, default=64)
    parser.add_argument("--candidate-limit", type=int, default=8)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_pack = get_pack("openui")
    base_state = OperatorStateV1.from_source(base_pack, SOURCE)
    root_branch = branch_fingerprint(base_state.state_digest, _sha("slm395-root"))
    root_context = _context(base_pack, base_state, root_branch, 395)
    base_node = ConversationStateNodeV1(
        parent_state_id=None,
        branch_digest=root_branch,
        state=base_state,
        reference_table=root_context.local.reference_table,
    )
    left_branch = branch_fingerprint(base_state.state_digest, _sha("slm395-left"))
    right_branch = branch_fingerprint(base_state.state_digest, _sha("slm395-right"))
    left, left_authority = _fixture_branch_edit(
        base_pack=base_pack,
        base_node=base_node,
        branch_digest=left_branch,
        seed=396,
        replacement=(":source.left", ":source.left_merged"),
        target_index=0,
    )
    right, right_authority = _fixture_branch_edit(
        base_pack=base_pack,
        base_node=base_node,
        branch_digest=right_branch,
        seed=397,
        replacement=(":source.right", ":source.right_merged"),
        target_index=1,
    )
    authorities = {
        left.input_node.state_id: left_authority,
        right.input_node.state_id: right_authority,
    }
    decision = merge_conversation_branches(
        pack=base_pack,
        base=base_node,
        left=left,
        right=right,
        authority_resolver=lambda node: authorities[node.state_id],
        reference_table_builder=_continuation_table,
    )
    if not decision.succeeded:
        raise RuntimeError("fixture branch merge was rejected")
    node = merged_continuation_node(decision)
    assert decision.continuation is not None and decision.merge is not None

    stale_codes = []
    old_ref = left.input_node.reference_table.entries[0].ref
    for state_digest, branch_digest in (
        (node.state.state_digest, left.input_node.branch_digest),
        (left.input_node.state.state_digest, node.branch_digest),
    ):
        try:
            left.input_node.reference_table.resolve(
                old_ref,
                state_digest=state_digest,
                branch_digest=branch_digest,
                expected_kind=old_ref.KIND,
            )
        except ReferenceResolutionError as exc:
            stale_codes.append(exc.code)
    alternate = _continuation_table(
        base_pack,
        node.state,
        node.branch_digest,
        decision.continuation.allocation_seed + 1,
    )

    followup_context = _context(
        base_pack,
        node.state,
        node.branch_digest,
        decision.continuation.allocation_seed,
    )
    if followup_context.local.reference_table != node.reference_table:
        raise RuntimeError("canonical continuation rebuild differs from merge table")
    followup_library = build_openui_topology_operator_library(followup_context)
    followup_pack = replace(base_pack, operator_library=followup_library)
    provenance = _provenance(node.state)
    legal = enumerate_operator_legal_set(
        pack=followup_pack,
        library=followup_library,
        state=node.state,
        reference_table=node.reference_table,
        provenance=provenance,
        max_combinations_per_operator=args.max_combinations,
    )
    action = next(iter(legal.operator_actions), None)
    if action is None:
        raise RuntimeError("fresh merged context has no legal follow-up")
    applied = followup_library.apply(
        followup_pack, node.state, action.operator_id, action.arguments, provenance
    )
    if not applied.succeeded or applied.state is None:
        raise RuntimeError("selected legal follow-up did not apply")
    output_table = _continuation_table(
        followup_pack,
        applied.state,
        node.branch_digest,
        decision.continuation.allocation_seed + 1,
    )
    trace = create_conversation_trace(
        pack=followup_pack,
        root_state=node.state,
        root_reference_table=node.reference_table,
        provenance=provenance,
    )
    trace = append_operator_turn(
        trace,
        pack=followup_pack,
        library=followup_library,
        application=applied.application,
        output_reference_table=output_table,
    )
    replayed = replay_conversation_trace(
        pack=followup_pack, library=followup_library, trace=trace
    )
    replay_branch_merge(
        pack=base_pack,
        base=base_node,
        left=left,
        right=right,
        authority_resolver=lambda item: authorities[item.state_id],
        reference_table_builder=_continuation_table,
        recorded=decision,
    )
    legacy = merge_conversation_branches(
        pack=base_pack,
        base=base_node,
        left=left,
        right=right,
        authority_resolver=lambda item: authorities[item.state_id],
    )
    try:
        merged_continuation_node(legacy)
    except ValueError as exc:
        legacy_error = str(exc)
    else:
        legacy_error = "accepted"

    stamp = build_version_stamp("dsl.operators.merge", "evals.agentv")
    audit = {
        "schema": "PostMergeContinuationReportV1",
        "run": {
            "device": "cpu",
            "steps": 0,
            "context_backend": "none",
            "matrix_set": "local_post_merge_continuation",
            "honesty": "fixture_wiring_only",
            "ship_claim": False,
        },
        "branch_candidate_pairs": 1,
        "merge_id": decision.merge.merge_id,
        "reference_table_fingerprint": node.reference_table.fingerprint,
        "allocation_seed": decision.continuation.allocation_seed,
        "fresh_table_matches_canonical_rebuild": True,
        "permuted_semantic_membership_matches": {
            entry.descriptor.semantic_fingerprint for entry in alternate.entries
        }
        == {
            entry.descriptor.semantic_fingerprint
            for entry in node.reference_table.entries
        },
        "old_table_refusal_codes": stale_codes,
        "legacy_terminal_error": legacy_error,
        "followup_legal_set_coverage": legal.coverage.value,
        "followup_action": action.semantic_id,
        "trace_turns": len(trace.turns),
        "replayed_state_digest": replayed.state.state_digest,
    }
    cases = [
        {
            "id": "fresh-canonical-table",
            "criteria": (
                "Merged state receives a fresh canonical topology reference table."
            ),
            "pass": audit["fresh_table_matches_canonical_rebuild"],
            "result": {"fingerprint": audit["reference_table_fingerprint"]},
        },
        {
            "id": "old-table-refusal-and-permutation",
            "criteria": (
                "Old branch tables refuse stale/cross-branch use; allocation "
                "permutation preserves membership."
            ),
            "pass": (
                set(audit["old_table_refusal_codes"])
                == {"ref.stale_state", "ref.cross_branch"}
                and audit["permuted_semantic_membership_matches"]
            ),
            "result": {"codes": audit["old_table_refusal_codes"]},
        },
        {
            "id": "followup-trace-replay-and-legacy-terminal",
            "criteria": (
                "Fresh legal follow-up replays through a whole trace and legacy "
                "merge artifacts stay terminal."
            ),
            "pass": audit["trace_turns"] == 1
            and audit["replayed_state_digest"] == applied.state.state_digest
            and audit["legacy_terminal_error"]
            == "merge.continuation_unsupported_legacy",
            "result": {"followup_action": audit["followup_action"]},
        },
    ]
    agentv = publish_agentv_evaluation(
        output_dir,
        name="slm395-post-merge-continuation",
        claim="fixture_post_merge_continuation_not_ship",
        cases=cases,
        version_stamp=stamp,
    )
    _rewrite_agentv_paths(output_dir)
    report = {
        "schema": "PostMergeContinuationReportV1",
        "audit": audit,
        "agentv": _portable(agentv, output_dir),
        "version_stamp": stamp,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary(output_dir, report)
    print(
        json.dumps(
            {"report": str(output_dir / "report.json"), **audit}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
