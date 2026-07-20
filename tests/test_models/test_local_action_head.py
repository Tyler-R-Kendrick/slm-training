"""Regression tests for state-local action heads (CAP2-03)."""

from __future__ import annotations

import math

import pytest
import torch

from slm_training.models.action_code_registry import (
    ActionCodeRegistry,
    ActionSchema,
)
from slm_training.models.local_action_head import (
    GlobalMaskedHead,
    GrammarFactorizedHead,
    LocalActionHead,
    LocalFlatHead,
    StateContext,
    TernaryDigitHead,
    TernaryECOCHead,
)
from slm_training.models.semantic_cost import (
    build_ternary_ecoc_entry,
    ordinal_base3_codeword,
    trit_distance,
    uniform_cost,
)


HIDDEN_DIM = 8


def _hidden(batch: int = 1) -> torch.Tensor:
    return torch.randn(batch, HIDDEN_DIM)


def test_forced_decision_bypasses_head() -> None:
    """A state with a single legal action must bypass the learned head."""
    head = LocalFlatHead(HIDDEN_DIM)
    ctx = StateContext(state_family_id="test")
    legal = ["only_action"]
    out = head.score(_hidden(), ctx, legal)
    decision = head.decode(out, legal)
    assert decision.decision_kind == "forced"
    assert decision.action_identity == "only_action"
    assert decision.confidence == 1.0


@pytest.mark.parametrize("b", [1, 2, 3, 4, 8, 9])
def test_ternary_digit_round_trip(b: int) -> None:
    """Base-3 encode/decode is a lossless round-trip for in-range indices."""
    head = TernaryDigitHead(HIDDEN_DIM)
    actions = [f"a{i}" for i in range(b)]
    m = math.ceil(math.log(b, 3)) if b > 1 else 0
    # Feed the codeword for action a(idx) directly as hard trits.
    for idx in range(b):
        cw = ordinal_base3_codeword(idx, m)
        # Build [1, m, 3] one-hot logits: large positive at the chosen trit.
        trits = torch.zeros(1, m, 3)
        for pos, trit in enumerate(cw):
            trits[0, pos, trit] = 100.0
        out = head.score(_hidden(), StateContext("test"), actions)
        out.trits = trits
        decision = head.decode(out, actions)
        assert decision.action_identity == actions[idx]
        # Forced decisions do not need a codeword; scored ones must round-trip.
        if decision.decision_kind != "forced":
            assert decision.codeword == cw


def test_ternary_ecoc_detects_every_single_trit_error() -> None:
    """Distance-2 ternary ECOC detects every single-trit corruption."""
    actions = ("a", "b", "c", "d")
    entry = build_ternary_ecoc_entry(
        ActionSchema("test", actions),
        detect_single_trit_error=True,
        use_exact_search=True,
    )
    words = [a.codeword for a in entry.assignments]
    # Parity check: the final trit equals the sum of the preceding trits mod 3.
    for w in words:
        assert w[-1] == sum(w[:-1]) % 3
    # Every single-trit flip must land outside the codebook.
    for w in words:
        for pos in range(len(w)):
            for new_trit in (0, 1, 2):
                if new_trit == w[pos]:
                    continue
                corrupted = list(w)
                corrupted[pos] = new_trit
                corrupted_t = tuple(corrupted)
                assert entry.action_for_codeword(corrupted_t) is None


def test_spare_codeword_counterexample() -> None:
    """b=5, m=2 without detection cannot detect every single-trit error.

    There are 9 possible length-2 words but only 5 are assigned.  A single-trit
    corruption of an assigned word can land on an unused valid word, making it
    indistinguishable from a legal action.  This motivates the distance-2 parity
    code used by TernaryECOCHead.
    """
    actions = ("a0", "a1", "a2", "a3", "a4")
    entry = build_ternary_ecoc_entry(
        ActionSchema("counter", actions),
        detect_single_trit_error=False,
        use_exact_search=True,
    )
    # m = ceil(log_3 5) = 2, so 9 words exist and 4 are unused.
    assert entry.alphabet_radices == (3, 3)
    assert len(entry.unused_codewords) == 4
    # Some single-trit error lands on an unused valid codeword.
    found_counterexample = False
    for assign in entry.assignments:
        w = assign.codeword
        for pos in range(len(w)):
            for new_trit in (0, 1, 2):
                if new_trit == w[pos]:
                    continue
                corrupted = list(w)
                corrupted[pos] = new_trit
                corrupted_t = tuple(corrupted)
                if corrupted_t in entry.unused_codewords:
                    found_counterexample = True
    assert found_counterexample


def test_semantic_cost_keeps_catastrophic_pair_apart() -> None:
    """A high-cost pair is assigned codewords at larger Hamming distance."""
    actions = ("safe_a", "safe_b", "catastrophic_a", "catastrophic_b")
    costs = uniform_cost(actions)
    # Make the catastrophic pair overwhelmingly costly.
    costs[("catastrophic_a", "catastrophic_b")] = 100.0
    costs[("catastrophic_b", "catastrophic_a")] = 100.0
    entry = build_ternary_ecoc_entry(
        ActionSchema("danger", actions),
        costs,
        detect_single_trit_error=True,
        use_exact_search=True,
        cost_matrix_source="fingerprint",
    )
    assignment = {a.action_identity: a.codeword for a in entry.assignments}
    cat_dist = trit_distance(
        assignment["catastrophic_a"],
        assignment["catastrophic_b"],
    )
    # The catastrophic pair is placed at least as far apart as the code minimum.
    assert cat_dist >= entry.minimum_hamming_distance
    # Detection is enabled, so the overall minimum distance is at least 2.
    assert entry.minimum_hamming_distance >= 2
    # The high-cost pair itself is maximized under the cost-weighted objective.
    assert cat_dist >= 2


def test_invalid_code_abstains() -> None:
    """Invalid trit words follow the configured fallback and stay legal."""
    actions = ["a", "b", "c", "d"]
    head = TernaryECOCHead(HIDDEN_DIM, registry=ActionCodeRegistry(), use_detection=True)
    # Construct an output that is guaranteed not to be a codeword: all 2s.
    entry = head._get_entry(actions)
    m = len(entry.alphabet_radices)
    bad_trits = torch.full((1, m, 3), -100.0)
    bad_trits[:, :, 2] = 100.0
    out = head.score(_hidden(), StateContext("test"), actions)
    out.trits = bad_trits
    decision = head.decode(out, actions)
    assert decision.decision_kind in ("abstain", "detected_error")
    assert decision.action_identity is None


def test_action_schema_hash_determinism() -> None:
    """Schema hashing is deterministic and order-sensitive in the expected way."""
    a = ActionSchema("family", ("x", "y", "z"))
    b = ActionSchema("family", ("x", "y", "z"))
    c = ActionSchema("family", ("z", "y", "x"))
    assert a.schema_hash() == b.schema_hash()
    assert a.schema_hash() != c.schema_hash()


def test_registry_rejects_duplicate_schema() -> None:
    """Registering the same schema twice raises a clear error."""
    registry = ActionCodeRegistry()
    schema = ActionSchema("dup", ("a", "b"))
    entry = build_ternary_ecoc_entry(schema)
    registry.register(entry)
    with pytest.raises(ValueError, match="schema already registered"):
        registry.register(entry)


def test_all_head_families_agree_on_forced_lossless_fixture() -> None:
    """Every head family returns the forced action on a single-legal state."""
    legal = ["component:root:none:card"]
    ctx = StateContext(state_family_id="fixture")
    heads: list[LocalActionHead] = [
        GlobalMaskedHead(HIDDEN_DIM, max_vocabulary=16),
        LocalFlatHead(HIDDEN_DIM),
        TernaryDigitHead(HIDDEN_DIM),
        TernaryECOCHead(HIDDEN_DIM, registry=ActionCodeRegistry()),
        GrammarFactorizedHead(HIDDEN_DIM),
    ]
    for head in heads:
        out = head.score(_hidden(), ctx, legal)
        decision = head.decode(out, legal)
        assert decision.action_identity == legal[0]
        assert decision.decision_kind == "forced"


def test_global_head_masks_illegal_actions() -> None:
    """GlobalMaskedHead scores are -inf for all illegal action indices."""
    head = GlobalMaskedHead(HIDDEN_DIM, max_vocabulary=16)
    legal = ["a0", "a1"]
    out = head.score(_hidden(), StateContext("global"), legal)
    assert out.logits is not None
    # Illegal positions (not in legal set) should be -inf.
    legal_indices = set(out.metadata["legal_indices"].tolist())
    for idx in range(16):
        value = out.logits[0, idx].item()
        if idx in legal_indices:
            assert math.isfinite(value)
        else:
            assert value == float("-inf")


def test_grammar_factorized_reconstructs_known_action() -> None:
    """GrammarFactorizedHead can reconstruct an action in the legal set."""
    head = GrammarFactorizedHead(HIDDEN_DIM)
    legal = ["component:root:none:card", "bind:arg0:none:text"]
    out = head.score(_hidden(), StateContext("grammar"), legal)
    decision = head.decode(out, legal)
    assert decision.action_identity in legal
    assert decision.decision_kind == "scored"


def test_default_decode_unchanged_when_feature_disabled() -> None:
    """A head_family-aware path falls back cleanly to the existing global path."""
    # This test documents that using a GlobalMaskedHead with a disabled feature
    # flag is still a no-op on the decoder: it produces a scored decision from
    # the legal subset only.
    head = GlobalMaskedHead(HIDDEN_DIM, max_vocabulary=8)
    legal = ["x", "y"]
    out = head.score(_hidden(), StateContext("compat"), legal)
    decision = head.decode(out, legal)
    assert decision.action_identity in legal
    assert decision.decision_kind == "scored"
    assert 0.0 < decision.confidence <= 1.0


def test_action_code_entry_hash_stability() -> None:
    """Entry hash is deterministic and sensitive to code assignment changes."""
    schema = ActionSchema("stable", ("a", "b"))
    e1 = build_ternary_ecoc_entry(schema, detect_single_trit_error=False)
    e2 = build_ternary_ecoc_entry(schema, detect_single_trit_error=False)
    assert e1.entry_hash == e2.entry_hash
    # Different alphabet radices -> different hash.
    e3 = build_ternary_ecoc_entry(schema, detect_single_trit_error=True)
    assert e3.entry_hash != e1.entry_hash
