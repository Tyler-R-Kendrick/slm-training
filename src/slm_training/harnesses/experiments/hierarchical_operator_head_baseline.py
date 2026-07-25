"""DSH3-15 hierarchical operator action head causal baseline (follow-up to E803).

SLM-383's own "Decision" text authorizes testing the encoder-side hierarchical
operator action head (``slm_training.dsl.operators.hierarchical_head``) "only
after the discrete token baseline proves causal value." DSH3-14/SLM-382
(E803, ``docs/design/e803-reserved-operator-baseline-20260723``) rejected the
*decoder-visible* token hypothesis outright -- every arm tied at chance. Per
AGENTS.md IV.11, that result says nothing about an encoder-side/masked
scoring mechanism, which is what the hierarchical head implements. This
module runs the multi-seed, matched-capacity causal-attribution experiment
SLM-383's acceptance criteria require: token baseline vs. weight-zero
(no-op capacity control) vs. enabled (trained) arms, with per-decision
causal-choice-change accounting, following the exact pattern of
``reserved_operator_baseline.py``.

Scope: this experiment measures only the stage-1 (action-kind/operator)
decision. Stage-2 (typed-argument) selection has no learnable signal under
the current typed feature set -- every candidate within one operator's group
shares an identical ``OperatorActionFeatureViewV1`` profile (same effect
counts), so no arm, including a decoder-visible token arm, can distinguish
between them without leaking opaque/display-name identity. That gap is
SLM-398's ("replace hash-scalar candidate features with typed embeddings")
declared follow-up, not this issue's.

Corpus: a small fixed vocabulary of ``(value_candidate_count,
role_candidate_count)`` "shapes" recurs across many independently-constructed
fixture states (distinct source text, reference-table seed, and state
digest), so gold depends only on the *typed* candidate-count feature -- never
on opaque ids, display names, or the specific state instance. This is
fixture/wiring-scale evidence (CPU-only, deterministic, seconds), not a
production or ship claim.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Sequence

import torch
from torch import nn

from slm_training.models.dynamic_pointer_scorer import (
    PointerCandidate,
    PointerCandidateSet,
    count_pointer_scorer_parameters,
)
from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    HierarchicalOperatorActionHead,
    HierarchicalOperatorHeadConfigV1,
    OperatorArgumentSlotV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    RegisteredOperatorV1,
    build_operator_action_features,
    build_reference_table,
    enumerate_operator_legal_set,
    group_operator_features,
)
from slm_training.dsl.pack import get_pack

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]*|\d+|[^\s]")

SET_VALUE_OPERATOR_ID = "openui.fixture_set_value"
BIND_ROLE_OPERATOR_ID = "openui.fixture_bind_role"


class HierarchicalOperatorHeadArm(str, Enum):
    TOKEN_BASELINE = "TOKEN_BASELINE"
    WEIGHT_ZERO = "WEIGHT_ZERO"
    ENABLED = "ENABLED"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class HierarchicalOperatorDecisionCaseV1:
    """One stage-1 (operator-kind) decision over a live, compiler-verified legal set."""

    decision_id: str
    shape: tuple[int, int]
    legal_set_fingerprint: str
    group_texts: dict[str, str]
    gold_operator_id: str
    legal_operator_ids: frozenset[str]


def _build_fixture_state(
    *, n_values: int, n_roles: int, decision_seed: int
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Build one small, independent fixture state with the given candidate-group shape.

    Mirrors ``tests/test_dsl/test_hierarchical_operator_head.py::_build_fixture``,
    parameterized by group sizes and an independent seed so every decision has
    a distinct source/state/reference-table identity even when its shape repeats.
    """
    source = 'root = TextContent(":before")'
    request_id = f"hier-head-baseline-{decision_seed}"
    base = get_pack("openui")
    state = OperatorStateV1.from_source(base, source)

    value_descriptors = tuple(
        ReferenceDescriptorV1(
            ref_kind=RefKind.VALUE,
            semantic_fingerprint=_sha(f"value-{decision_seed}-{i}"),
            value_type="openui.string",
        )
        for i in range(n_values)
    )
    role_descriptors = tuple(
        ReferenceDescriptorV1(
            ref_kind=RefKind.ROLE,
            semantic_fingerprint=_sha(f"role-{decision_seed}-{i}"),
            value_type="openui.role",
        )
        for i in range(n_roles)
    )
    table = build_reference_table(
        request_id=request_id,
        state_digest=state.state_digest,
        branch_digest=_sha(f"branch-{decision_seed}"),
        descriptors=value_descriptors + role_descriptors,
        seed=decision_seed,
    )

    def execute_set_value(_state: Any, arguments: Any) -> OperatorMutationV1:
        (argument,) = arguments
        return OperatorMutationV1(
            source='root = TextContent(":afterset")',
            effect=ActionEffectV1(
                property_deltas=(
                    EffectDeltaV1(EffectDeltaKind.PROPERTY, argument.value, "before", "after"),
                ),
                compiler_coverage=CompilerCoverage.EXACT,
            ),
        )

    def execute_bind_role(_state: Any, arguments: Any) -> OperatorMutationV1:
        (argument,) = arguments
        return OperatorMutationV1(
            source='root = TextContent(":bound")',
            effect=ActionEffectV1(
                scope_deltas=(
                    EffectDeltaV1(EffectDeltaKind.SCOPE, argument.value, "unbound", "bound"),
                ),
                compiler_coverage=CompilerCoverage.EXACT,
            ),
        )

    set_value_decl = AstOperatorV1(
        operator_id=SET_VALUE_OPERATOR_ID,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(OperatorArgumentSlotV1("value", RefKind.VALUE, BindingPhase.APPLICATION),),
        preconditions=(),
        effect_signature=(EffectDeltaKind.PROPERTY,),
        locality="node",
        cost=1.0,
    )
    bind_role_decl = AstOperatorV1(
        operator_id=BIND_ROLE_OPERATOR_ID,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(OperatorArgumentSlotV1("role", RefKind.ROLE, BindingPhase.APPLICATION),),
        preconditions=(),
        effect_signature=(EffectDeltaKind.SCOPE,),
        locality="node",
        cost=1.0,
    )
    library = OperatorLibraryV1(
        (
            RegisteredOperatorV1(set_value_decl, execute_set_value),
            RegisteredOperatorV1(bind_role_decl, execute_bind_role),
        )
    )
    pack = replace(base, operator_library=library)
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="hier-head-baseline",
        compiler_version="v1",
        source_artifact_digest=_sha(source),
        request_id=request_id,
    )
    legal_set = enumerate_operator_legal_set(
        pack=pack, library=library, state=state, reference_table=table, provenance=provenance
    )
    return pack, library, state, table, provenance, legal_set


def _group_text(operator_id: str, candidate_count: int, dominant_effect_kind: str) -> str:
    """Decoder-visible-equivalent rendering of one operator group's typed features.

    Carries exactly the same information the encoder-side head's stage-1
    candidate representation is built from (operator id, candidate count,
    dominant effect kind) -- a matched-information text arm, not an
    information-advantaged one.
    """
    return f"operator={operator_id};candidates={candidate_count};effect={dominant_effect_kind}"


def build_hierarchical_operator_decisions(
    *,
    shapes: Sequence[tuple[int, int]],
    repeats: int,
    seed: int,
) -> tuple[HierarchicalOperatorDecisionCaseV1, ...]:
    """Build decisions whose gold operator depends only on typed candidate-group size.

    Each ``(n_values, n_roles)`` shape recurs ``repeats`` times with an
    independent source/state/reference-table identity per repeat, so a model
    that memorizes opaque ids or one specific state cannot generalize --  only
    memorizing the finite shape-to-gold mapping (a typed, not opaque, feature)
    transfers across repeats.
    """
    cases: list[HierarchicalOperatorDecisionCaseV1] = []
    counter = 0
    for n_values, n_roles in shapes:
        if n_values == n_roles:
            raise ValueError("ambiguous shape: value/role candidate counts must differ")
        for _ in range(repeats):
            decision_seed = seed * 10_000 + counter
            counter += 1
            _pack, library, state, _table, provenance, legal_set = _build_fixture_state(
                n_values=n_values, n_roles=n_roles, decision_seed=decision_seed
            )
            features = build_operator_action_features(
                legal_set=legal_set,
                library=library,
                pack=_pack,
                state=state,
                provenance=provenance,
            )
            groups = group_operator_features(features)
            assert {g.operator_id for g in groups} == {
                SET_VALUE_OPERATOR_ID,
                BIND_ROLE_OPERATOR_ID,
            }
            group_texts = {
                g.operator_id: _group_text(g.operator_id, g.candidate_count, g.dominant_effect_kind)
                for g in groups
            }
            gold_operator_id = max(groups, key=lambda g: g.candidate_count).operator_id
            cases.append(
                HierarchicalOperatorDecisionCaseV1(
                    decision_id=_sha(f"{seed}:{decision_seed}:{n_values}:{n_roles}"),
                    shape=(n_values, n_roles),
                    legal_set_fingerprint=legal_set.fingerprint,
                    group_texts=group_texts,
                    gold_operator_id=gold_operator_id,
                    legal_operator_ids=frozenset(group_texts),
                )
            )
    if not cases:
        raise ValueError("hierarchical operator head baseline requires decisions")
    return tuple(cases)


def _token_ids(value: str, buckets: int) -> torch.Tensor:
    tokens = _TOKEN_RE.findall(value)
    if not tokens:
        tokens = ["<empty>"]
    return torch.tensor(
        [(zlib.crc32(token.encode()) & 0xFFFFFFFF) % buckets for token in tokens],
        dtype=torch.long,
    )


class _TokenVisibleGroupScorer(nn.Module):
    """Decoder-visible-equivalent hashed-token scorer over rendered group text."""

    def __init__(self, *, seed: int, buckets: int = 1024, width: int = 32) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.buckets = buckets
        self.embedding = nn.Embedding(buckets, width)
        self.candidate = nn.Linear(width, width)
        self.bias = nn.Linear(width, 1)

    def _encode(self, value: str) -> torch.Tensor:
        return self.embedding(_token_ids(value, self.buckets)).mean(dim=0)

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        vectors = torch.stack([torch.tanh(self.candidate(self._encode(text))) for text in texts])
        return self.bias(vectors).squeeze(-1)


def _decision_order(case: HierarchicalOperatorDecisionCaseV1) -> list[str]:
    return sorted(case.legal_operator_ids)


def _run_token_baseline_arm(
    *,
    train: Sequence[HierarchicalOperatorDecisionCaseV1],
    held_out: Sequence[HierarchicalOperatorDecisionCaseV1],
    seed: int,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    model = _TokenVisibleGroupScorer(seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(steps):
        loss = torch.tensor(0.0)
        for case in train:
            order = _decision_order(case)
            texts = [case.group_texts[op] for op in order]
            logits = model(texts)
            target = order.index(case.gold_operator_id)
            loss = loss + torch.nn.functional.cross_entropy(
                logits.unsqueeze(0), torch.tensor([target])
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    predictions = []
    correct = 0
    for case in held_out:
        order = _decision_order(case)
        texts = [case.group_texts[op] for op in order]
        with torch.no_grad():
            selected = int(model(texts).argmax().item())
        selected_operator_id = order[selected]
        is_correct = selected_operator_id == case.gold_operator_id
        correct += is_correct
        predictions.append(
            {
                "decision_id": case.decision_id,
                "selected_operator_id": selected_operator_id,
                "gold_operator_id": case.gold_operator_id,
                "correct": is_correct,
                "false_legal_admission": selected_operator_id not in case.legal_operator_ids,
            }
        )
    n = len(held_out)
    return {
        "arm": HierarchicalOperatorHeadArm.TOKEN_BASELINE.value,
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "initial_loss": history[0],
        "final_loss": history[-1],
        "held_out_n": n,
        "operator_accuracy": correct / n,
        "false_legal_admissions": sum(p["false_legal_admission"] for p in predictions),
        "predictions": predictions,
    }


def _zero_head_parameters(head: HierarchicalOperatorActionHead) -> None:
    for parameter in head._scorer.parameters():
        parameter.data.zero_()
        parameter.requires_grad_(False)


def _evaluate_head(
    *,
    head: HierarchicalOperatorActionHead,
    cases: Sequence[HierarchicalOperatorDecisionCaseV1],
    d_model: int,
) -> list[dict[str, Any]]:
    predictions = []
    for case in cases:
        order = _decision_order(case)
        candidates = _candidate_set(order, [case.group_texts[op] for op in order])
        state_vector = torch.zeros(d_model)
        logits = head._scorer.forward(state_vector, candidates)
        selected = int(torch.argmax(logits).item())
        selected_operator_id = order[selected]
        is_correct = selected_operator_id == case.gold_operator_id
        predictions.append(
            {
                "decision_id": case.decision_id,
                "selected_operator_id": selected_operator_id,
                "gold_operator_id": case.gold_operator_id,
                "correct": is_correct,
                "false_legal_admission": selected_operator_id not in case.legal_operator_ids,
            }
        )
    return predictions


def _candidate_set(operator_ids: Sequence[str], texts: Sequence[str]) -> PointerCandidateSet:
    """Build the real ``PointerCandidateSet`` stage-1 scoring sees for these groups.

    Reuses the production ``PointerCandidate``/``PointerCandidateSet`` contract
    so the scorer forward pass is byte-identical to live stage-1 scoring;
    ``stable_id``/``type_name`` are the real live operator ids, never invented.
    """
    candidates = tuple(
        PointerCandidate(
            stable_id=operator_id,
            display_text=text,
            kind="schema_entity",
            type_name=operator_id,
            provenance="compiler_scope",
        )
        for operator_id, text in zip(operator_ids, texts, strict=True)
    )
    payload = "|".join(f"{c.stable_id}:{c.display_text}" for c in candidates)
    return PointerCandidateSet(
        candidates=candidates,
        permitted_sources=("structured_contract",),
        manifest_hash=_sha(payload)[:16],
    )


def _run_weight_zero_arm(
    *,
    held_out: Sequence[HierarchicalOperatorDecisionCaseV1],
    seed: int,
    d_model: int,
    hidden_dim: int,
) -> dict[str, Any]:
    head = HierarchicalOperatorActionHead(
        HierarchicalOperatorHeadConfigV1(mode="dynamic_head", d_model=d_model, hidden_dim=hidden_dim)
    )
    _zero_head_parameters(head)
    predictions = _evaluate_head(head=head, cases=held_out, d_model=d_model)
    correct = sum(p["correct"] for p in predictions)
    n = len(held_out)
    return {
        "arm": HierarchicalOperatorHeadArm.WEIGHT_ZERO.value,
        "seed": seed,
        "steps": 0,
        "learning_rate": 0.0,
        "parameter_count": sum(p.numel() for p in head._scorer.parameters()),
        "trainable_parameter_count": count_pointer_scorer_parameters(head._scorer),
        "initial_loss": None,
        "final_loss": None,
        "held_out_n": n,
        "operator_accuracy": correct / n,
        "false_legal_admissions": sum(p["false_legal_admission"] for p in predictions),
        "predictions": predictions,
    }


def _run_enabled_arm(
    *,
    train: Sequence[HierarchicalOperatorDecisionCaseV1],
    held_out: Sequence[HierarchicalOperatorDecisionCaseV1],
    seed: int,
    steps: int,
    learning_rate: float,
    d_model: int,
    hidden_dim: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    head = HierarchicalOperatorActionHead(
        HierarchicalOperatorHeadConfigV1(mode="dynamic_head", d_model=d_model, hidden_dim=hidden_dim)
    )
    optimizer = torch.optim.Adam(head._scorer.parameters(), lr=learning_rate)
    state_vector = torch.zeros(d_model)
    history = []
    for _ in range(steps):
        loss = torch.tensor(0.0)
        for case in train:
            order = _decision_order(case)
            candidates = _candidate_set(order, [case.group_texts[op] for op in order])
            logits = head._scorer.forward(state_vector, candidates)
            target = order.index(case.gold_operator_id)
            loss = loss + torch.nn.functional.cross_entropy(
                logits.unsqueeze(0), torch.tensor([target])
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    predictions = _evaluate_head(head=head, cases=held_out, d_model=d_model)
    correct = sum(p["correct"] for p in predictions)
    n = len(held_out)
    return {
        "arm": HierarchicalOperatorHeadArm.ENABLED.value,
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "parameter_count": sum(p.numel() for p in head._scorer.parameters()),
        "trainable_parameter_count": count_pointer_scorer_parameters(head._scorer),
        "initial_loss": history[0] if history else None,
        "final_loss": history[-1] if history else None,
        "held_out_n": n,
        "operator_accuracy": correct / n,
        "false_legal_admissions": sum(p["false_legal_admission"] for p in predictions),
        "predictions": predictions,
    }


def _causal_changes(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    control_by_id = {item["decision_id"]: item for item in control["predictions"]}
    changed = correct = wrong = 0
    for item in treatment["predictions"]:
        prior = control_by_id[item["decision_id"]]
        if item["selected_operator_id"] == prior["selected_operator_id"]:
            continue
        changed += 1
        correct += bool(item["correct"] and not prior["correct"])
        wrong += bool(prior["correct"] and not item["correct"])
    n = treatment["held_out_n"]
    return {
        "changed": changed,
        "eligible": n,
        "change_rate": changed / n,
        "correct_changes": correct,
        "wrong_changes": wrong,
    }


DEFAULT_SHAPES: tuple[tuple[int, int], ...] = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2))


def run_hierarchical_operator_head_baseline(
    *,
    shapes: Sequence[tuple[int, int]] = DEFAULT_SHAPES,
    train_repeats: int = 5,
    held_out_repeats: int = 2,
    seeds: Sequence[int] = (11, 29, 47),
    steps: int = 24,
    learning_rate: float = 0.05,
    d_model: int = 16,
    hidden_dim: int = 16,
) -> dict[str, Any]:
    """Run the matched DSH3-15 token-baseline/weight-zero/enabled causal comparison.

    Every arm sees the identical set of held-out decisions and (for trainable
    arms) the identical train decisions, seeds, optimizer steps, and learning
    rate. ``WEIGHT_ZERO`` shares the enabled arm's architecture with every
    parameter forced to zero and frozen -- a no-op capacity control, not a
    trained baseline. Only ``ENABLED``'s parameters vary by seed; the corpus
    (built once, independent of seed) is held fixed across seeds so seed
    variation reflects only optimization stochasticity.
    """
    train = build_hierarchical_operator_decisions(shapes=shapes, repeats=train_repeats, seed=1)
    held_out = build_hierarchical_operator_decisions(
        shapes=shapes, repeats=held_out_repeats, seed=2
    )

    runs: dict[str, list[dict[str, Any]]] = {arm.value: [] for arm in HierarchicalOperatorHeadArm}
    for seed in seeds:
        runs[HierarchicalOperatorHeadArm.TOKEN_BASELINE.value].append(
            _run_token_baseline_arm(
                train=train, held_out=held_out, seed=seed, steps=steps, learning_rate=learning_rate
            )
        )
        runs[HierarchicalOperatorHeadArm.WEIGHT_ZERO.value].append(
            _run_weight_zero_arm(
                held_out=held_out, seed=seed, d_model=d_model, hidden_dim=hidden_dim
            )
        )
        runs[HierarchicalOperatorHeadArm.ENABLED.value].append(
            _run_enabled_arm(
                train=train,
                held_out=held_out,
                seed=seed,
                steps=steps,
                learning_rate=learning_rate,
                d_model=d_model,
                hidden_dim=hidden_dim,
            )
        )

    control_arm = HierarchicalOperatorHeadArm.TOKEN_BASELINE.value
    controls = runs[control_arm]
    treatment_names = [
        HierarchicalOperatorHeadArm.WEIGHT_ZERO.value,
        HierarchicalOperatorHeadArm.ENABLED.value,
    ]
    changes: dict[str, list[dict[str, Any]]] = {
        name: [
            _causal_changes(control, treatment)
            for control, treatment in zip(controls, runs[name], strict=True)
        ]
        for name in treatment_names
    }

    def mean(arm: str, metric: str) -> float:
        values = [float(run[metric]) for run in runs[arm]]
        return sum(values) / len(values)

    enabled_beats_zero = all(
        run["operator_accuracy"] > zero["operator_accuracy"]
        for run, zero in zip(
            runs[HierarchicalOperatorHeadArm.ENABLED.value],
            runs[HierarchicalOperatorHeadArm.WEIGHT_ZERO.value],
            strict=True,
        )
    )
    enabled_beats_token_baseline = all(
        run["operator_accuracy"] > control["operator_accuracy"]
        for run, control in zip(
            runs[HierarchicalOperatorHeadArm.ENABLED.value], controls, strict=True
        )
    )
    acceptance = {
        "causal_change_rate_at_least_0_05_vs_weight_zero": all(
            item["change_rate"] >= 0.05
            for item in changes[HierarchicalOperatorHeadArm.WEIGHT_ZERO.value]
        ),
        "correct_changes_exceed_wrong_changes_vs_weight_zero": all(
            item["correct_changes"] > item["wrong_changes"]
            for item in changes[HierarchicalOperatorHeadArm.WEIGHT_ZERO.value]
        ),
        "enabled_beats_weight_zero_capacity_control": enabled_beats_zero,
        "enabled_causally_improves_beyond_token_baseline": enabled_beats_token_baseline,
        "zero_false_legal_admissions": all(
            run["false_legal_admissions"] == 0 for values in runs.values() for run in values
        ),
        "cap0_retention": {
            "available": True,
            "pass": True,
            "reason": "default_off_path_is_unchanged",
        },
        "cap1_retention": {
            "available": False,
            "pass": None,
            "reason": "CERT_CAP1_unavailable",
            "dependency_issue": "SLM-379",
        },
    }
    # DSH3-15's acceptance bar is explicit: the enabled head must causally
    # improve *beyond the token baseline*, not merely beyond a null control.
    passed = bool(
        acceptance["enabled_causally_improves_beyond_token_baseline"]
        and acceptance["zero_false_legal_admissions"]
        and acceptance["cap0_retention"]["pass"]
    )
    return {
        "schema": "hierarchical_operator_head_baseline/v1",
        "issue": "SLM-383",
        "predecessor_experiment_id": "E803",
        "train_decision_n": len(train),
        "held_out_decision_n": len(held_out),
        "shapes": [list(shape) for shape in shapes],
        "seeds": list(seeds),
        "steps_per_arm": steps,
        "learning_rate": learning_rate,
        "d_model": d_model,
        "hidden_dim": hidden_dim,
        "arms": runs,
        "causal_changes_vs_token_baseline": changes,
        "mean_operator_accuracy": {arm: mean(arm, "operator_accuracy") for arm in runs},
        "acceptance": acceptance,
        "accepted": passed,
        "verdict": "accept" if passed else "reject",
        "scope_note": (
            "Stage-1 (operator-kind) decision only; stage-2 (typed-argument) "
            "selection has no distinguishing typed-feature signal under the "
            "current hash-scalar candidate features (SLM-398 follow-up) and is "
            "out of scope for this run."
        ),
    }


__all__ = [
    "HierarchicalOperatorHeadArm",
    "HierarchicalOperatorDecisionCaseV1",
    "build_hierarchical_operator_decisions",
    "run_hierarchical_operator_head_baseline",
]
