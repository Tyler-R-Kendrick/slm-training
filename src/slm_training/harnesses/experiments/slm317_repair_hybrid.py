"""SLM-317 (LAR2-06): do-no-harm AR→repair hybrid commit machinery.

A valid-state repair arm is only useful if it never damages a correct AR
program. This module provides the deterministic, metamorphism-invariant
commit rule for the AR→repair hybrid screen:

- :func:`hard_evidence` scores one final-program source on the LAR1-02
  unified ladder: official parse (``dsl.parser.validate``) → output contract
  (``output_contract_violations``) → ``meaningful_program_v1`` verdict. The
  ladder rank is the ONLY hard authority.
- :func:`commit` applies the preregistered do-no-harm rule:

  a. hard evidence improves → COMMIT (``hard_improvement``);
  b. hard evidence regresses → RETAIN source (``hard_regression``);
  c. hard evidence unchanged and a calibrated learned score (model value or
     oracle distance improvement) improves → COMMIT
     (``soft_improvement_no_hard_regression``);
  d. otherwise RETAIN (``no_improvement``); a missing/empty candidate is an
     ABSTAIN (``candidate_unavailable``), never a silent commit.

- :func:`oracle_commit` is the non-promotable upper bound: commit exactly
  when the candidate's hard evidence is strictly better.
- Metamorphic transforms (:func:`alpha_rename`, :func:`reorder_statements`,
  :func:`normalize_formatting`, :func:`ast_roundtrip`) are
  semantics-preserving source rewrites used by the invariance tests: hard
  evidence and commit decisions must be identical across them.
- :func:`repair_decode` runs the branch's value-guided beam repair starting
  from an arbitrary valid source state (not the minimal seed), fail-closed
  via ``TreeEditSpace.apply``.

The module is torch-free at import time; ``repair_decode``/``state_value``
import torch lazily through the model they are given.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from slm_training.dsl.language_contract import output_contract_violations
from slm_training.dsl.parser import validate
from slm_training.harnesses.model_build.eval_runner import (
    meaningful_program_v1_report,
)
from slm_training.models.tree_edit_diffusion import (
    ACTION_STOP,
    Statement,
    parse_statements,
    render_statements,
)

EXPERIMENT_ID = "slm317-repair-hybrid"
COMMIT_DECISION_SCHEMA = "slm317_commit_decision/v1"

# Commit actions.
ACTION_COMMIT = "commit"
ACTION_RETAIN = "retain"
ACTION_ABSTAIN = "abstain"

# Preregistered reason codes (the full closed set).
REASON_HARD_IMPROVEMENT = "hard_improvement"
REASON_HARD_REGRESSION = "hard_regression"
REASON_SOFT_IMPROVEMENT = "soft_improvement_no_hard_regression"
REASON_NO_IMPROVEMENT = "no_improvement"
REASON_CANDIDATE_UNAVAILABLE = "candidate_unavailable"
REASON_ORACLE_HARD_IMPROVEMENT = "oracle_hard_improvement"
REASON_ORACLE_NO_GAIN = "oracle_no_gain"

# Dispositions (exactly one per screen run).
DISPOSITION_POSITIVE = "repair_positive"
DISPOSITION_NEGATIVE = "repair_negative"
DISPOSITION_INCONCLUSIVE = "inconclusive"


# --------------------------------------------------------------------------- #
# Hard evidence (LAR1-02 ladder)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HardEvidence:
    """Deterministic verifier/contract evidence for one final program."""

    parse_ok: bool
    contract_ok: bool
    v1_verdict: bool
    v1_reason_codes: tuple[str, ...]
    contract_violations: tuple[str, ...]

    @property
    def rank(self) -> int:
        """Ordered ladder: 0 unparseable < 1 parsed < 2 contract-clean < 3 v1-valid."""
        if not self.parse_ok:
            return 0
        if not self.contract_ok:
            return 1
        if not self.v1_verdict:
            return 2
        return 3

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["v1_reason_codes"] = list(self.v1_reason_codes)
        data["contract_violations"] = list(self.contract_violations)
        data["rank"] = self.rank
        return data


def hard_evidence(source: str | None) -> HardEvidence:
    """Score one program source on the deterministic ladder. Never raises."""
    if not source:
        return HardEvidence(False, False, False, ("no_program",), ())
    parse_ok = False
    try:
        validate(source)
        parse_ok = True
    except Exception:  # noqa: BLE001 - parse fact, never raised
        parse_ok = False
    violations: tuple[str, ...] = ()
    if parse_ok:
        violations = tuple(output_contract_violations(source))
    contract_ok = parse_ok and not violations
    report = meaningful_program_v1_report(source, gold=None)
    return HardEvidence(
        parse_ok=parse_ok,
        contract_ok=contract_ok,
        v1_verdict=bool(report["verdict"]),
        v1_reason_codes=tuple(report.get("reason_codes", ())),
        contract_violations=violations,
    )


# --------------------------------------------------------------------------- #
# Commit rule
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CommitDecision:
    """Durable per-example commit record (source + candidate + exact reason)."""

    action: str  # commit | retain | abstain
    reason: str
    source: str
    candidate: str
    final: str
    source_evidence: HardEvidence
    candidate_evidence: HardEvidence
    learned_score_source: float | None = None
    learned_score_candidate: float | None = None
    record_id: str = ""
    arm_id: str = ""
    seed: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COMMIT_DECISION_SCHEMA,
            "action": self.action,
            "reason": self.reason,
            "source": self.source,
            "candidate": self.candidate,
            "final": self.final,
            "source_evidence": self.source_evidence.to_dict(),
            "candidate_evidence": self.candidate_evidence.to_dict(),
            "learned_score_source": self.learned_score_source,
            "learned_score_candidate": self.learned_score_candidate,
            "record_id": self.record_id,
            "arm_id": self.arm_id,
            "seed": self.seed,
            "notes": list(self.notes),
        }


def commit(
    source: str,
    candidate: str | None,
    *,
    evidence: dict[str, Any] | None = None,
    record_id: str = "",
    arm_id: str = "",
    seed: int = 0,
) -> CommitDecision:
    """Do-no-harm commit rule (deterministic; see module docstring).

    ``evidence`` may carry ``learned_score_source`` /
    ``learned_score_candidate`` (model value or oracle distance improvement;
    higher = better) and ``learned_margin`` (strict improvement margin,
    default 0.0). Learned scores are only consulted when hard evidence is
    unchanged, and a soft commit is forbidden whenever hard evidence regresses.
    """
    evidence = evidence or {}
    src_ev = hard_evidence(source)
    if candidate is None or not candidate.strip():
        return CommitDecision(
            action=ACTION_ABSTAIN,
            reason=REASON_CANDIDATE_UNAVAILABLE,
            source=source,
            candidate="",
            final=source,
            source_evidence=src_ev,
            candidate_evidence=hard_evidence(None),
            record_id=record_id,
            arm_id=arm_id,
            seed=seed,
        )
    cand_ev = hard_evidence(candidate)
    soft_s = evidence.get("learned_score_source")
    soft_c = evidence.get("learned_score_candidate")
    margin = float(evidence.get("learned_margin", 0.0))

    if cand_ev.rank > src_ev.rank:
        action, reason, final = ACTION_COMMIT, REASON_HARD_IMPROVEMENT, candidate
    elif cand_ev.rank < src_ev.rank:
        action, reason, final = ACTION_RETAIN, REASON_HARD_REGRESSION, source
    elif (
        soft_s is not None
        and soft_c is not None
        and float(soft_c) > float(soft_s) + margin
    ):
        action, reason, final = ACTION_COMMIT, REASON_SOFT_IMPROVEMENT, candidate
    else:
        action, reason, final = ACTION_RETAIN, REASON_NO_IMPROVEMENT, source
    return CommitDecision(
        action=action,
        reason=reason,
        source=source,
        candidate=candidate,
        final=final,
        source_evidence=src_ev,
        candidate_evidence=cand_ev,
        learned_score_source=None if soft_s is None else float(soft_s),
        learned_score_candidate=None if soft_c is None else float(soft_c),
        record_id=record_id,
        arm_id=arm_id,
        seed=seed,
    )


def oracle_commit(
    source: str,
    candidate: str | None,
    *,
    record_id: str = "",
    arm_id: str = "",
    seed: int = 0,
) -> CommitDecision:
    """Non-promotable upper bound: commit exactly on strict hard improvement."""
    if candidate is None or not candidate.strip():
        return commit(
            source, candidate, record_id=record_id, arm_id=arm_id, seed=seed
        )
    src_ev = hard_evidence(source)
    cand_ev = hard_evidence(candidate)
    better = cand_ev.rank > src_ev.rank
    return CommitDecision(
        action=ACTION_COMMIT if better else ACTION_RETAIN,
        reason=REASON_ORACLE_HARD_IMPROVEMENT if better else REASON_ORACLE_NO_GAIN,
        source=source,
        candidate=candidate,
        final=candidate if better else source,
        source_evidence=src_ev,
        candidate_evidence=cand_ev,
        record_id=record_id,
        arm_id=arm_id,
        seed=seed,
        notes=["oracle selector; non-promotable upper bound"],
    )


# --------------------------------------------------------------------------- #
# Metamorphic transforms (semantics-preserving; used by invariance tests)
# --------------------------------------------------------------------------- #


def alpha_rename(source: str, *, prefix: str = "m") -> str:
    """Consistently rename non-root node identifiers (references included)."""
    statements = parse_statements(source)
    if statements is None:
        return source
    mapping: dict[str, str] = {}
    for stmt in statements:
        if stmt.name != "root":
            mapping[stmt.name] = f"{prefix}{len(mapping)}"
    renamed = [
        Statement(
            name=mapping.get(stmt.name, stmt.name),
            comp=stmt.comp,
            children=[mapping.get(c, c) for c in stmt.children],
            rest=stmt.rest,
            has_list=stmt.has_list,
        )
        for stmt in statements
    ]
    return render_statements(renamed)


def reorder_statements(source: str) -> str:
    """Move the root statement last (declarations first); semantics preserved."""
    statements = parse_statements(source)
    if statements is None:
        return source
    roots = [s for s in statements if s.name == "root"]
    others = [s for s in statements if s.name != "root"]
    return render_statements(others + roots)


def normalize_formatting(source: str) -> str:
    """Re-render the structural form (canonical spacing/newlines)."""
    statements = parse_statements(source)
    if statements is None:
        return source.strip()
    return render_statements(statements)


def ast_roundtrip(source: str) -> str:
    """Equivalent AST serialization via the official parser's serialized form."""
    try:
        program = validate(source)
    except Exception:  # noqa: BLE001 - unparseable input round-trips to itself
        return source
    return program.serialized or source.strip()


# --------------------------------------------------------------------------- #
# Repair decode from an arbitrary valid source state
# --------------------------------------------------------------------------- #


def repair_decode(
    model: Any,
    source: str,
    inventory: list[str],
    prompt: str,
) -> tuple[str, dict[str, Any]]:
    """Value-guided beam repair seeded from ``source`` (not the minimal seed).

    Mirrors ``TreeEditDiffusionModel._decode_one`` (same fail-closed
    ``TreeEditSpace.apply``, same value re-scoring, same STOP handling) with
    the beam initialized on the caller's valid state. An unparseable or
    invalid source fails closed: returns ``("", {"failure": ...})``.
    """
    import torch  # lazy: module import stays torch-free

    statements = parse_statements(source)
    if statements is None:
        return "", {"failure": "source_unparseable", "kind": "tree_edit_repair"}
    from slm_training.models.tree_edit_diffusion import _is_valid

    if not _is_valid(source):
        return "", {"failure": "source_invalid", "kind": "tree_edit_repair"}

    model.eval()
    ctx, ctx_pad = model._encode_context(
        [model._format_context(prompt, slot_contract=list(inventory))]
    )
    cfg = model.config
    beam: list[tuple[float, list[Statement], bool]] = [(0.0, statements, False)]
    evidence: dict[str, Any] = {"steps": 0, "expansions": 0, "kind": "tree_edit_repair"}
    with torch.no_grad():
        for _ in range(cfg.max_search_steps):
            live = [entry for entry in beam if not entry[2]]
            if not live:
                break
            sources = [render_statements(s) for _, s, _ in live]
            out = model.policy(
                model._state_batch(sources),
                model.tokenizer.pad_id,
                ctx.expand(len(sources), -1, -1),
                ctx_pad.expand(len(sources), -1),
            )
            next_beam = [entry for entry in beam if entry[2]]
            seen: set[str] = {
                render_statements(s) for _, s, frozen in next_beam if frozen
            }
            for row, (_, stmts, _) in enumerate(live):
                candidates = model._enumerate_edits(out, row, len(stmts), len(inventory))
                expanded = 0
                for score, edit in candidates:
                    if expanded >= cfg.expand_per_state:
                        break
                    if edit.action == ACTION_STOP:
                        text = render_statements(stmts)
                        if text in seen:
                            continue
                        seen.add(text)
                        next_beam.append((float(out["value"][row]), stmts, True))
                        expanded += 1
                        continue
                    child = model.space.apply(stmts, edit, inventory, reason=[])
                    if child is None:
                        continue
                    text = render_statements(child)
                    if text in seen:
                        continue
                    seen.add(text)
                    next_beam.append((float(out["value"][row]), child, False))
                    expanded += 1
                    evidence["expansions"] += 1
            if not next_beam:
                break
            unfrozen = [entry for entry in next_beam if not entry[2]]
            if unfrozen:
                rescore = model.policy(
                    model._state_batch([render_statements(s) for _, s, _ in unfrozen]),
                    model.tokenizer.pad_id,
                    ctx.expand(len(unfrozen), -1, -1),
                    ctx_pad.expand(len(unfrozen), -1),
                )
                rescored = [
                    (float(rescore["value"][i]), entry[1], False)
                    for i, entry in enumerate(unfrozen)
                ]
            else:
                rescored = []
            frozen = [entry for entry in next_beam if entry[2]]
            beam = sorted(frozen + rescored, key=lambda e: e[0], reverse=True)[
                : cfg.beam_width
            ]
            evidence["steps"] += 1
            if all(entry[2] for entry in beam):
                break
    best = max(beam, key=lambda e: e[0])
    evidence["value"] = float(best[0])
    evidence["frozen"] = bool(best[2])
    return render_statements(best[1]), evidence


def state_value(model: Any, source: str, inventory: list[str], prompt: str) -> float | None:
    """Calibrated learned score (policy value head) for one state; None if unparseable."""
    import torch  # lazy

    if parse_statements(source) is None:
        return None
    model.eval()
    ctx, ctx_pad = model._encode_context(
        [model._format_context(prompt, slot_contract=list(inventory))]
    )
    with torch.no_grad():
        out = model.policy(
            model._state_batch([source]),
            model.tokenizer.pad_id,
            ctx,
            ctx_pad,
        )
    return float(out["value"][0])
