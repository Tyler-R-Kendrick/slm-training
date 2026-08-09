"""PCT-006: static/request-boundary differential + singleton zero-work guard.

Extends the E1 differential harness in ``test_static_control_domain.py`` —
not a parallel reimplementation. Every claim here drives the shipped
``static_control_domain`` / ``pack._openui_completion_domain`` entry points.

Scope (SLM-460, see docs/design/repository-ownership-map.md):

* E1-style live-oracle parity held across *distinct request-boundary facts*
  (``slot_contract``) at the same certified static artifact and prefix — the
  "overlay varied ... schema positions ... against the same static artifact"
  half of the issue.
* The certified static projection is identical underneath those
  request-boundary variations even though the combined candidate set
  genuinely differs (static keys exclude request-local facts).
* A budget-exhausted UNKNOWN witness beside exactly one proven candidate
  must not manufacture a complete singleton domain, driven through the real
  ``pack._openui_completion_domain`` — unlike the existing
  ``tests/test_models/test_decode_stats.py`` coverage, which pins the
  predicate shape but hand-reproduces the branch rather than calling it.

Confirmed by direct experimentation before writing this file: varying
``slot_contract`` does not merely *remove* candidates (a subset relation)
because it also changes what counts as a complete program (the minimum
slot-fill requirement), so no "dynamic only tightens" subset claim is made
here across distinct overlays — only the claim the artifact/pack modules
actually make: each overlay, independently, matches live construction
exactly (E1 parity), and the static side never moves underneath any of them.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.grammar.fastpath.static_control_domain import (
    HARD_PREFIX_SURFACE,
    compare_domain_parity,
    oracle_lark_domain_snapshot,
    production_domain_snapshot,
    require_certified_static_projection,
)
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

# Mirrors the E1 corpus in test_static_control_domain.py.
_CORPUS = (
    'root = TextContent(":slot_0")\n',
    "root = Card([])\n",
    'root = Card([b1, b2])\nb1 = TextContent(":slot_0")\nb2 = TextContent(":slot_1")\n',
    "root = Card([b1",
    HARD_PREFIX_SURFACE,
    "root =",
    "root",
    "",
)

# A request-boundary overlay distinct from the default ``(":slot_0",
# ":slot_1")`` contract already exercised by
# ``test_static_control_domain.py::test_e1_kernel_corpus_parity`` — that file
# owns default-contract parity; this one extends coverage to a second,
# narrower ``slot_contract`` fact against the same certified static artifact
# rather than re-running the already-proven default.
_NARROW_SLOT_CONTRACT = (":slot_0",)


@pytest.fixture(scope="module")
def tok() -> DSLNativeTokenizer:
    return DSLNativeTokenizer.build()


@pytest.mark.parametrize("program", _CORPUS, ids=lambda p: repr(p[-24:]))
def test_request_boundary_overlay_matches_live_oracle_every_prefix(
    tok: DSLNativeTokenizer, program: str
) -> None:
    """A second request-boundary overlay still proves exact live parity.

    PCT-006 scope: "[d]ifferential static-domain vs live grammar/tokenizer
    authority across prefixes" while "[o]verlay[ing] varied ... schema
    positions ... against the same static artifact." Walks a ``slot_contract``
    distinct from the default across every prefix of the corpus.
    """
    ids = [int(t) for t in tok.encode(program, add_special=False)]
    for end in range(0, len(ids) + 1):
        prefix = [tok.bos_id, *ids[:end]]
        production = production_domain_snapshot(
            prefix, tok, budget=32, slot_contract=_NARROW_SLOT_CONTRACT
        )
        oracle = oracle_lark_domain_snapshot(
            prefix, tok, budget=32, slot_contract=_NARROW_SLOT_CONTRACT
        )
        report = compare_domain_parity(production, oracle)
        assert report.equal, (program, end, report.reasons)
        assert report.no_unknown_collapse


def test_static_artifact_excludes_request_boundary_facts(
    tok: DSLNativeTokenizer,
) -> None:
    """The certified static projection is identical across request overlays.

    AC: "Static cache/artifact keys exclude request-local facts." Two
    requests carrying different ``slot_contract`` values must resolve to
    byte-identical static authority — fixed by grammar/tokenizer identity
    only, never by per-request facts — even though the two overlays produce
    genuinely different combined candidate sets (proving the overlay was
    real, not a no-op).
    """
    direct_map_a = require_certified_static_projection(tok)
    direct_map_b = require_certified_static_projection(tok)
    assert direct_map_a == direct_map_b

    prefix = [tok.bos_id, *tok.encode("root = TextContent(", add_special=False)]
    wide = production_domain_snapshot(
        prefix, tok, budget=32, slot_contract=(":slot_0", ":slot_1")
    )
    narrow = production_domain_snapshot(
        prefix, tok, budget=32, slot_contract=(":slot_0",)
    )
    # The overlay is genuine: distinct request-local facts change the
    # combined candidate set even though the static side never moved.
    assert wide.candidates != narrow.candidates
    assert require_certified_static_projection(tok) == direct_map_a


def test_budget_exhausted_unknown_witness_never_forces_singleton(
    tok: DSLNativeTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single proven candidate beside dropped UNKNOWN ones stays incomplete.

    Drives the real ``pack._openui_completion_domain`` (not the hand-copied
    branch in ``tests/test_models/test_decode_stats.py``): every
    ``terminal_witness`` query but the first is forced to UNKNOWN so exactly
    one candidate is proven while others were dropped for a budget reason,
    not a soundness proof. I2 requires this to fail closed to "incomplete",
    never surface as a manufactured complete singleton (AC: "Incomplete
    singleton-like domains do not force").
    """
    from slm_training.dsl.grammar.fastpath.completion_kernel import (
        CompletionSession,
        WitnessResult,
        WitnessStatus,
    )
    from slm_training.dsl.grammar_capabilities import CompletionDomainRequestV1
    from slm_training.dsl.pack import _openui_completion_domain
    from slm_training.models.decode_stats import collect_decode_stats

    prefix_ids = [tok.bos_id, *tok.encode("root = ", add_special=False)]
    baseline = production_domain_snapshot(prefix_ids, tok, budget=32)
    assert baseline.status == "complete"
    # A genuine branch: many real candidates exist before any patching.
    assert len(baseline.candidates) > 1

    original_terminal_witness = CompletionSession.terminal_witness
    calls = {"n": 0}

    def _first_real_rest_unknown(self, state_id, room):
        calls["n"] += 1
        if calls["n"] == 1:
            return original_terminal_witness(self, state_id, room)
        return WitnessResult(status=WitnessStatus.UNKNOWN)

    monkeypatch.setattr(CompletionSession, "terminal_witness", _first_real_rest_unknown)

    request = CompletionDomainRequestV1(
        prefix_ids=tuple(prefix_ids),
        tokenizer=tok,
        slot_contract=(":slot_0", ":slot_1"),
        remaining_tokens=32,
    )
    with collect_decode_stats() as stats:
        domain = _openui_completion_domain(request)

    assert calls["n"] > 1  # the hazard shape actually fired, not a no-op
    assert domain.status == "incomplete"
    assert domain.reason == "witness_false_singleton_risk"
    assert domain.candidates == ()
    assert stats.witness_false_singleton_risk == 1
    # A non-"complete" domain can never be treated as a proven singleton by
    # any downstream forced-commit consumer — I2's fail-closed contract lives
    # entirely in the "status == complete" gate. See
    # test_static_control_domain.py::test_e8_ambiguous_domain_refuses_singleton_bypass
    # for the sibling "ambiguous complete domain refuses" half of the guard.
