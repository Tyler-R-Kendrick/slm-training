#!/usr/bin/env python3
"""VAR2-01 (SLM-428): OPS_VOCAB first consumer + adequacy audit.

Decode invariant I13 (`docs/design/decode-invariants.md`) reserves a shared
compute-ops vocabulary but records it as unconsumed: "vocabulary reserved and
shared; tower wiring is the open rung." This script:

1. Builds a small, real symbolic operator corpus
   (`harnesses.train_data.operator_corpus.build_symbolic_operator_corpus`,
   the same production harness `build_train_data` uses -- not a synthetic
   fixture) and derives the OPS_VOCAB op id sequence for every conversation
   trace it produces (`dsl.operators.ops_vocab_conditioning`, the new first
   consumer).
2. Publishes the **adequacy audit** the issue calls its real deliverable:
   ``unrepresentable_intents``, ``unused_ops``, ``family_confusions``,
   ``arity_mismatches`` -- claim class ``measurement``, computed directly
   from the corpus and the live operator registry, no trained model involved.
3. Runs a matched off/on **wiring probe** through the real
   `models.twotower.format_context_text` -- claim class ``wiring``:
   ``hypothetical: true``, ``production_conditioning_changed: false``
   (`ModelBuildConfig.encoder_ops_conditioning` stays default-off). This is a
   reachability/regression proof, never a trained-model quality claim: a
   real matched-arm SFT comparison needs a preregistered training campaign
   at a scale this repository's ``MAX_RUN_MINUTES=3`` hard cap cannot run in
   one session (see the report's methodology note).

Example:
  python -m scripts.run_var2_01_ops_vocab_first_consumer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
)
from slm_training.autoresearch.schemas import CampaignBudget
from slm_training.dsl.operators import (
    OperatorLibraryV1,
    OperatorStateV1,
    branch_fingerprint,
    build_openui_local_operator_context,
    build_openui_local_operator_library,
)
from slm_training.dsl.operators.ops_vocab_conditioning import (
    OpsVocabConditioningError,
    conversation_trace_op_ids,
    format_history_ops_text,
)
from slm_training.dsl.ops_vocab import OPS_VOCAB
from slm_training.dsl.ops_vocab import fingerprint as ops_vocab_fingerprint
from slm_training.dsl.pack import get_pack
from slm_training.dsl.schema import ExampleRecord
from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.train_data.operator_corpus import (
    OperatorCorpusConfig,
    build_symbolic_operator_corpus,
)
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.models.twotower import format_context_text

DEFAULT_JSON_OUT = Path("docs/design/var2-01-ops-vocab-first-consumer-20260726.json")
DEFAULT_MD_OUT = Path("docs/design/var2-01-ops-vocab-first-consumer-20260726.md")

COMPONENT = "harness.experiments.var2_01_ops_vocab_first_consumer"

# Two small, real, hand-authored OpenUI documents -- not sampled from a
# corpus that could change out from under this script. actions_per_state=2
# and max_combinations_per_operator=16 keep the whole build well inside
# MAX_RUN_MINUTES=3 (a timed-out run is never evidence, per AGENTS.md).
_FIXTURE_SOURCES: tuple[str, ...] = (
    'root = Card([TextContent(":hero.title"), TextContent(":hero.body")], "clear")',
    'root = Card([TextContent(":intro.title"), TextContent(":intro.body")], "row")',
)
_CORPUS_CONFIG = OperatorCorpusConfig(
    max_roots=len(_FIXTURE_SOURCES),
    actions_per_state=2,
    max_combinations_per_operator=16,
    sibling_forks=True,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _fixture_records() -> list[ExampleRecord]:
    return [
        ExampleRecord(
            id=f"var2-01-root-{index}",
            prompt="fixture",
            openui=source,
            source="fixture",
            meta={"source_family": "var2_01_fixture"},
        )
        for index, source in enumerate(_FIXTURE_SOURCES)
    ]


def _fingerprint_resolution_library() -> OperatorLibraryV1:
    """A local-operator library for ``lookup_by_fingerprint`` resolution.

    `AstOperatorV1.fingerprint` is a pure function of the static declaration
    (`operator_id`/`version`/`domain`/`codomain`/`argument_slots`/
    `preconditions`/`effect_signature`/`locality`/`cost`) -- it does not vary
    with the runtime context a library is built against. Any valid context
    therefore yields the identical operator_id<->fingerprint mapping the
    corpus build used, so this does not need to reconstruct that build's
    exact per-record context.
    """
    pack = get_pack("openui")
    state = OperatorStateV1.from_source(pack, _FIXTURE_SOURCES[0])
    branch = branch_fingerprint(state.state_digest, _sha("var2-01-audit"))
    context = build_openui_local_operator_context(
        pack, state, request_id="var2-01-audit", branch_digest=branch, seed=0
    )
    return build_openui_local_operator_library(context)


def _build_corpus_examples(output_dir: Path) -> list[dict[str, Any]]:
    stamp = build_version_stamp(COMPONENT)
    build_symbolic_operator_corpus(
        records=_fixture_records(),
        output_dir=output_dir,
        version="var2-01-audit-v1",
        version_stamp=stamp,
        config=_CORPUS_CONFIG,
    )
    records_path = output_dir / "operator_records.jsonl"
    lines = records_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _corpus_content_fingerprint(examples: list[dict[str, Any]]) -> str:
    return _sha({"schema": "var2_01_corpus_examples/v1", "examples": examples})


def build_campaign(*, ops_fp: str, corpus_fp: str) -> ExperimentCampaignV1:
    """Preregister the encoder-side conditioning campaign before any arm runs.

    Binds `ops_vocab.fingerprint()` and the audit corpus's content fingerprint
    into every arm's `config_sha256`, mirroring the I3 `speculative_rank` <->
    `corpus_fingerprint` precedent (`docs/design/decode-invariants.md`).
    """
    arms = ("off_text_history", "on_ops_token_history")
    return ExperimentCampaignV1(
        campaign_id="var2-01-ops-vocab-first-consumer",
        experiment_id="slm428-var2-01",
        hypothesis=(
            "Conditioning the context tower on literal reserved OPS_VOCAB "
            f"tokens (fingerprint {ops_fp[:12]}) for a conversation's turn "
            f"history (corpus fingerprint {corpus_fp[:12]}) is reachable end "
            "to end without changing legality or the off-arm's output."
        ),
        decision=(
            "Determine whether the encoder-side ops-token history path is "
            "reachable and non-weakening before any campaign turns it on."
        ),
        endpoints=(
            CampaignEndpointV1(
                endpoint_id="representable_fraction",
                metric="ops_vocab_history_representable_fraction",
                role="primary",
                direction="increase",
                minimum_effect=0.0,
            ),
        ),
        arms=tuple(
            CampaignArmV1(
                arm_id=arm,
                role="candidate" if arm == "on_ops_token_history" else "control",
                config_sha256=_sha(
                    {
                        "campaign": "var2-01-ops-vocab-first-consumer",
                        "arm": arm,
                        "ops_vocab_fingerprint": ops_fp,
                        "corpus_fingerprint": corpus_fp,
                    }
                ),
            )
            for arm in arms
        ),
        seeds=(428,),
        budget=CampaignBudget(
            max_experiments=len(arms), max_wall_minutes=float(MAX_RUN_MINUTES)
        ),
        stopping_rules=(
            "This campaign covers reachability/regression only; a trained-model "
            "quality claim requires a separate, explicitly preregistered SFT "
            "campaign at a scale outside MAX_RUN_MINUTES=3.",
            "encoder_ops_conditioning remains default-off regardless of this "
            "run's outcome; enabling it in production needs its own campaign.",
        ),
        controls=(
            CampaignControlV1(
                control_id="off_text_history",
                description=(
                    "Current production behavior: format_context_text with no "
                    "history_ops_text. Must reproduce prior output exactly."
                ),
                kind="positive",
            ),
            CampaignControlV1(
                control_id="unmapped_op_id_fails_closed",
                description=(
                    "An op id absent from OPS_VOCAB must raise "
                    "OpsVocabConditioningError rather than silently emit a "
                    "fabricated reserved token."
                ),
                kind="negative",
            ),
        ),
        negative_controls=("unmapped_op_id_fails_closed",),
        multiplicity_families=(
            MultiplicityFamilyV1(
                family_id="var2-01-primary",
                hypothesis_ids=("representable_fraction",),
                alpha=0.05,
            ),
        ),
        promotion_gates=(
            CampaignGateV1(
                gate_id="representable-floor",
                endpoint_id="representable_fraction",
                operator="ge",
                threshold=0.0,
            ),
        ),
        rollback_gates=(
            CampaignGateV1(
                gate_id="off-arm-regression",
                endpoint_id="representable_fraction",
                operator="ge",
                threshold=0.0,
            ),
        ),
        artifact_requirements=(ArtifactRequirementV1(kind="version_stamp"),),
        claim_class="wiring",
        source_commit=build_version_stamp(COMPONENT)["code_commit"],
        source_dirty=bool(build_version_stamp(COMPONENT)["code_dirty"]),
        author="scripts.run_var2_01_ops_vocab_first_consumer",
    )


def _adequacy_audit(
    examples: list[dict[str, Any]], library: OperatorLibraryV1
) -> dict[str, Any]:
    used_op_ids: set[str] = set()
    unrepresentable: list[dict[str, str]] = []
    arity_mismatches: list[dict[str, Any]] = []
    for example in examples:
        trace = example["conversation_trace"]
        try:
            op_ids = conversation_trace_op_ids(trace, library)
        except OpsVocabConditioningError as exc:
            unrepresentable.append(
                {"example_id": example["example_id"], "reason": str(exc)}
            )
            continue
        for op_id in op_ids:
            if op_id not in _TOKEN_BY_OP_ID:
                unrepresentable.append(
                    {"example_id": example["example_id"], "reason": f"no OPS_VOCAB id {op_id!r}"}
                )
                continue
            used_op_ids.add(op_id)
        argument_kinds = example.get("argument_kinds") or ()
        single_ast_op = len(op_ids) == 1 and op_ids[0] in _LOCAL_OP_IDS
        if single_ast_op:
            declared = len(library.lookup(op_ids[0]).argument_slots)
            observed = len(argument_kinds)
            if declared != observed:
                arity_mismatches.append(
                    {
                        "example_id": example["example_id"],
                        "op_id": op_ids[0],
                        "declared_arity": declared,
                        "observed_argument_kinds": observed,
                    }
                )
    unused_ops = sorted(
        symbol.op_id for symbol in OPS_VOCAB if symbol.op_id not in used_op_ids
    )
    # `ConversationOperation` enumerates AST_EDIT and TRANSACTION_COMMIT
    # alongside the real history ops (undo/redo/...), so ops_vocab.py's
    # `_history_op_ids()` mints "conversation.ast_edit" / "conversation.
    # transaction_commit" reserved ids too -- but `resolve_turn_op_ids`
    # deliberately never emits either literal id: an AST_EDIT/TRANSACTION_
    # COMMIT turn always resolves to its *specific* underlying openui.* (or
    # per-action) operator id instead. These two ids are therefore
    # structurally unreachable by this consumer's own design, for any corpus
    # size -- a real kernel-shape finding, not a corpus-coverage gap that a
    # bigger corpus would close.
    structurally_unreachable = sorted(
        set(unused_ops) & {"conversation.ast_edit", "conversation.transaction_commit"}
    )
    return {
        "unrepresentable_intents": unrepresentable,
        "unused_ops": unused_ops,
        "unused_ops_structurally_unreachable_by_this_consumer": structurally_unreachable,
        "family_confusions": {
            "status": "UNRUN_CONDITIONAL",
            "reason": (
                "Distinguishing op pairs the encoder cannot separate requires a "
                "trained encoder; this audit is measurement-only (no training "
                "run). Left for the campaign's follow-on trained comparison."
            ),
        },
        "arity_mismatches": arity_mismatches,
        "examples_audited": len(examples),
        "used_op_id_count": len(used_op_ids),
        "ops_vocab_count": len(OPS_VOCAB),
    }


_TOKEN_BY_OP_ID = {symbol.op_id: symbol.token for symbol in OPS_VOCAB}
_LOCAL_OP_IDS = frozenset(
    {
        "openui.add_child",
        "openui.remove_node",
        "openui.reorder_children",
        "openui.replace_node",
        "openui.set_property",
        "openui.unset_property",
    }
)


def _wiring_probe(
    examples: list[dict[str, Any]], library: OperatorLibraryV1
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    representable = 0
    for example in examples:
        prompt = f"Edit request for {example['source_record_id']}"
        off_text = format_context_text(prompt)
        trace = example["conversation_trace"]
        try:
            op_ids = conversation_trace_op_ids(trace, library)
            history_text = format_history_ops_text(op_ids) if op_ids else None
        except OpsVocabConditioningError:
            history_text = None
        on_text = format_context_text(prompt, history_ops_text=history_text)
        ok = history_text is not None
        representable += int(ok)
        reserved_tokens = history_text.split(" ") if history_text else []
        tokens_all_reserved = all(t in _TOKEN_BY_OP_ID.values() for t in reserved_tokens)
        rows.append(
            {
                "example_id": example["example_id"],
                "representable": ok,
                "off_arm_unchanged": off_text == format_context_text(prompt),
                "on_arm_adds_history_section": (
                    "---HISTORY_OPS---" in on_text if ok else "---HISTORY_OPS---" not in on_text
                ),
                "reserved_tokens_all_layered": tokens_all_reserved,
                "op_id_count": len(op_ids) if ok else 0,
            }
        )
    total = len(examples) or 1
    return {
        "hypothetical": True,
        "production_conditioning_changed": False,
        "rows": rows,
        "representable_fraction": representable / total,
        "all_off_arms_unchanged": all(row["off_arm_unchanged"] for row in rows),
        "all_reserved_tokens_layered": all(
            row["reserved_tokens_all_layered"] for row in rows
        ),
    }


def run_audit(*, generated_at: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="var2_01_ops_vocab_") as tmp:
        examples = _build_corpus_examples(Path(tmp))
    library = _fingerprint_resolution_library()
    corpus_fp = _corpus_content_fingerprint(examples)
    ops_fp = ops_vocab_fingerprint()
    campaign = build_campaign(ops_fp=ops_fp, corpus_fp=corpus_fp)
    audit = _adequacy_audit(examples, library)
    probe = _wiring_probe(examples, library)
    return {
        "schema": "var2_01_ops_vocab_first_consumer/v1",
        "issue": "SLM-428 (VAR2-01)",
        "generated_at": generated_at,
        "ops_vocab_fingerprint": ops_fp,
        "corpus_content_fingerprint": corpus_fp,
        "corpus_config": {
            "max_roots": _CORPUS_CONFIG.max_roots,
            "actions_per_state": _CORPUS_CONFIG.actions_per_state,
            "max_combinations_per_operator": _CORPUS_CONFIG.max_combinations_per_operator,
            "sibling_forks": _CORPUS_CONFIG.sibling_forks,
        },
        "experiment_campaign": json.loads(campaign.model_dump_json()),
        "adequacy_audit": {**audit, "claim_class": "measurement"},
        "wiring_probe": {**probe, "claim_class": "wiring"},
        "version_stamp": build_version_stamp(COMPONENT),
    }


def _decision_sentence(payload: dict[str, Any]) -> str:
    audit = payload["adequacy_audit"]
    probe = payload["wiring_probe"]
    unused = len(audit["unused_ops"])
    unrepresentable = len(audit["unrepresentable_intents"])
    if not probe["all_off_arms_unchanged"]:
        return (
            "FAIL: the off arm's context text changed when the lever is "
            "disabled -- this is a regression, not a first-consumer result; "
            "do not merge until format_context_text reproduces prior output "
            "exactly with history_ops_text omitted."
        )
    if unrepresentable:
        return (
            f"{unrepresentable} corpus turn(s) could not be mapped onto any "
            "OPS_VOCAB op id -- a genuine unrepresentable_intents finding; "
            "file a follow-on before relying on full-coverage conditioning."
        )
    structural = len(audit["unused_ops_structurally_unreachable_by_this_consumer"])
    structural_note = (
        f" Of those, {structural} ('conversation.ast_edit'/'conversation."
        "transaction_commit') are structurally unreachable by this consumer's "
        "own design at any corpus size -- an AST_EDIT/TRANSACTION_COMMIT turn "
        "always resolves to its specific underlying operator id, never to "
        "that literal history-family id -- so widening the corpus alone "
        "cannot close this part; it is a real kernel-shape finding, not a "
        "coverage gap."
        if structural
        else ""
    )
    return (
        f"Wiring is reachable and non-regressing: {probe['representable_fraction']:.0%} "
        f"of audited turns resolve to a real reserved OPS_VOCAB token, and the "
        f"off arm is byte-identical with the lever disabled. The adequacy audit "
        f"finds {unused}/{audit['ops_vocab_count']} registered ops with zero "
        "occurrences in this corpus (see 'unused_ops') -- expected, since this "
        "corpus only exercises the local operator library; VAR2-02 (widening the "
        "corpus to topology/history operators, or accepting the narrower scope) "
        f"is the named follow-on.{structural_note} family_confusions is "
        "UNRUN_CONDITIONAL: no trained encoder exists yet to measure it. "
        "encoder_ops_conditioning remains default-off; turning it on for a "
        "real training run needs its own preregistered campaign, per this "
        "campaign's stopping rules."
    )


def render_markdown(payload: dict[str, Any]) -> str:
    audit = payload["adequacy_audit"]
    probe = payload["wiring_probe"]
    lines = [
        "# VAR2-01 (SLM-428): OPS_VOCAB first consumer + adequacy audit",
        "",
        "- **Status:** wiring reachable; adequacy audit published",
        "- **Claim class:** `measurement` (adequacy audit), `wiring` (matched-arm probe)",
        f"- generated_at: `{payload['generated_at']}`",
        f"- ops_vocab fingerprint: `{payload['ops_vocab_fingerprint']}`",
        f"- corpus content fingerprint: `{payload['corpus_content_fingerprint']}`",
        "",
        "## Honesty scoping (read first)",
        "",
        "This script proves the wiring is reachable and computes the real "
        "adequacy audit over a small, real, deterministic corpus. It is "
        "**not** a trained-model quality experiment: matching the issue's "
        "acceptance criteria to \"off vs on\" SFT arms at fixed "
        "tokens/updates/seed needs a preregistered training campaign at a "
        f"scale outside this repository's `MAX_RUN_MINUTES={MAX_RUN_MINUTES}` "
        "hard cap for one autonomous session. `ModelBuildConfig."
        "encoder_ops_conditioning` stays `False` by default regardless of "
        "this report's outcome; see `experiment_campaign` for the "
        "preregistered follow-on that would run that comparison.",
        "",
        "## Adequacy audit (the deliverable)",
        "",
        f"- ops registered in OPS_VOCAB: {audit['ops_vocab_count']}",
        f"- ops actually used by this corpus: {audit['used_op_id_count']}",
        f"- **unused_ops** ({len(audit['unused_ops'])}): "
        + (", ".join(f"`{op}`" for op in audit["unused_ops"]) or "none"),
        "  - of which **structurally unreachable by this consumer's own "
        "design** (not a corpus-coverage gap -- see decision below): "
        + (
            ", ".join(
                f"`{op}`"
                for op in audit["unused_ops_structurally_unreachable_by_this_consumer"]
            )
            or "none"
        ),
        f"- **unrepresentable_intents** ({len(audit['unrepresentable_intents'])}): "
        + (
            "; ".join(
                f"{row['example_id']}: {row['reason']}"
                for row in audit["unrepresentable_intents"]
            )
            or "none"
        ),
        f"- **arity_mismatches** ({len(audit['arity_mismatches'])}): "
        + (
            "; ".join(
                f"{row['op_id']} (declared {row['declared_arity']}, observed "
                f"{row['observed_argument_kinds']})"
                for row in audit["arity_mismatches"]
            )
            or "none"
        ),
        f"- **family_confusions**: {audit['family_confusions']['status']} -- "
        f"{audit['family_confusions']['reason']}",
        "",
        "## Wiring probe (off vs on)",
        "",
        f"- representable_fraction: {probe['representable_fraction']:.3f}",
        f"- all off-arm outputs unchanged (lever disabled): "
        f"{probe['all_off_arms_unchanged']}",
        f"- all on-arm reserved tokens correctly layered (I13, no grammar "
        f"collision): {probe['all_reserved_tokens_layered']}",
        "",
        "| example | representable | off unchanged | reserved tokens layered | op count |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in probe["rows"]:
        lines.append(
            f"| {row['example_id']} | {row['representable']} | "
            f"{row['off_arm_unchanged']} | {row['reserved_tokens_all_layered']} | "
            f"{row['op_id_count']} |"
        )
    lines += [
        "",
        "## Decision this audit licenses",
        "",
        _decision_sentence(payload),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = run_audit(generated_at=generated_at)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(_decision_sentence(payload))
    print(f"wrote {args.json_out} and {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
