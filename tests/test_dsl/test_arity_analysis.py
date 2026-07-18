"""CAP0-02 exact arity analyzer tests (Torch-free, deterministic).

The pinned counts here are **repository certificates for the committed
``bounded-expr`` fixture only**. They are deliberately *not* the external
CAP0-01 source-reported estimates (130 bounded ASTs / 351 trie / 41 minimized /
162-190-345 / Hankel / residual). Those remain source-reported estimates per
``docs/design/calculated-arity-adaptive-precision.md`` and are NOT reproduced
here; the raw 86-state value is retired from new conclusions. Never edit these
assertions to chase 130/351/41 — assert whatever the analyzer actually computes.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from slm_training.dsl.lang_core import ParseError

# --- Certified counts for the committed fixture (byte-stable). --------------
# bounds: max_ast_nodes=6, max_live_bindings=2, dimensions=4, ops=(+,-,*,/).
FIXTURE_MAX_AST_NODES = 6
FIXTURE_MAX_LIVE_BINDINGS = 2
FIXTURE_DIMENSIONS = 4
EXPECT_CANONICAL_ASTS = 400
EXPECT_RAW_STATES = 11
EXPECT_TRIE_STATES = 844
EXPECT_MINIMIZED_STATES = 28
EXPECT_ACTION_ALPHABET = 8
EXPECT_SCOPE_SIGNATURES = 3
EXPECT_MAX_BRANCHING = 6
EXPECT_BRANCHING_HISTOGRAM = {0: 1, 1: 11, 2: 5, 3: 2, 4: 2, 5: 4, 6: 3}
EXPECT_COMPLETION_COUNTS = {0: 3, 1: 8, 2: 7, 3: 6, 4: 4}
EXPECT_MIN_K = 3
EXPECT_VALIDATE_REJECTED = 3
# The external estimates we must never claim to reproduce.
EXTERNAL_ESTIMATES = {"asts": 130, "trie": 351, "minimized": 41}


def _bounds():
    from slm_training.dsl.analysis.arity import AnalysisBounds

    return AnalysisBounds(
        max_ast_nodes=FIXTURE_MAX_AST_NODES,
        max_live_bindings=FIXTURE_MAX_LIVE_BINDINGS,
        template_classes=("N",),
        result_types=("number",),
    )


@pytest.fixture(scope="module")
def report():
    from slm_training.dsl.analysis.arity import analyze

    return analyze(
        fixture="bounded-expr", bounds=_bounds(), dimensions=FIXTURE_DIMENSIONS
    )


# --- Torch-free -------------------------------------------------------------

def test_analysis_package_is_torch_free() -> None:
    # Checked in a fresh interpreter so the result is independent of whatever a
    # sibling test suite imported earlier this session. This is the strong form:
    # torch may be installed, but importing the package must not pull it in.
    probe = (
        "import sys; import slm_training.dsl.analysis.arity as a; "
        "assert a.analyze and a.minimize and a.build_trie; "
        "leaked = sorted(m for m in sys.modules if m == 'torch' "
        "or m.startswith('torch.')); "
        "print('LEAKED', leaked); "
        "sys.exit(1 if leaked else 0)"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)


# --- Canonicalization: literals + names ------------------------------------

def test_literal_and_name_canonicalization() -> None:
    from slm_training.dsl.analysis.arity import program_from_source

    program = program_from_source("x = 3\ny = x * 4\nroot = y + 2")
    # literals collapse to the typed template symbol "N"; identifiers become
    # nearest-preceding de Bruijn refs (x is 1 back from y; y is 1 back from root).
    assert program.to_json() == [
        ["lit", "N"],
        ["op", "*", ["ref", 1], ["lit", "N"]],
        ["op", "+", ["ref", 1], ["lit", "N"]],
    ]


def test_alpha_equivalent_programs_share_fingerprint() -> None:
    from slm_training.dsl.analysis.arity import program_from_source

    a = program_from_source("x = 3\ny = x * 4\nroot = y + 2")
    b = program_from_source("p = 9\nq = p * 8\nroot = q + 7")
    assert a.fingerprint() == b.fingerprint()
    # A structurally different program must differ.
    c = program_from_source("x = 3\ny = x + 4\nroot = y + 2")
    assert a.fingerprint() != c.fingerprint()


def test_shadowing_resolves_to_nearest_preceding_binder() -> None:
    from slm_training.dsl.analysis.arity import program_from_source

    # The later ``x`` shadows the earlier one; root's ref binds to the nearest.
    shadowed = program_from_source("x = 1\nx = 2\nroot = x + 3")
    assert shadowed.to_json()[2] == ["op", "+", ["ref", 1], ["lit", "N"]]


def test_materialize_roundtrips_to_canonical() -> None:
    from slm_training.dsl.analysis.arity import (
        CanonicalProgram,
        materialize,
        program_from_source,
    )
    from slm_training.dsl.analysis.arity.canonical import lit, op, ref

    program = CanonicalProgram(
        (op("*", lit(), lit()), op("+", ref(1), lit()))
    )
    round_tripped = program_from_source(materialize(program))
    assert round_tripped.fingerprint() == program.fingerprint()


# --- Type rejection via the backend oracle ---------------------------------

def test_type_rejection_raises_on_bare_atom_root() -> None:
    from slm_training.dsl.analysis.arity import is_type_valid
    from slm_training.dsl.analysis.arity.canonical import assert_type_valid

    # A bare-atom ``root`` is rejected by the arith-sketch backend's validate.
    assert is_type_valid("root = 5") is False
    with pytest.raises(ParseError):
        assert_type_valid("root = 5")
    # A compound ``root`` expression is accepted.
    assert is_type_valid("root = 5 + 1") is True


def test_validate_gate_is_non_vacuous(report) -> None:
    # The pipeline actually rejects type-invalid candidates before counting.
    assert report.work_counters["validate_rejected"] == EXPECT_VALIDATE_REJECTED
    assert report.work_counters["validate_rejected"] >= 1


# --- Deterministic enumeration ---------------------------------------------

def test_enumeration_is_deterministic() -> None:
    from slm_training.dsl.analysis.arity import enumerate_programs
    from slm_training.dsl.analysis.arity.state_graph import EnumerationBounds

    bounds = EnumerationBounds(
        max_ast_nodes=FIXTURE_MAX_AST_NODES,
        max_ast_depth=None,
        max_live_bindings=FIXTURE_MAX_LIVE_BINDINGS,
        operators=("+", "-", "*", "/"),
        template_classes=("N",),
    )
    first = enumerate_programs(bounds)
    second = enumerate_programs(bounds)
    assert first.complete and second.complete
    assert [p.fingerprint() for p in first.programs] == [
        p.fingerprint() for p in second.programs
    ]
    # No fingerprint appears twice (canonical dedup holds).
    fingerprints = [p.fingerprint() for p in first.programs]
    assert len(fingerprints) == len(set(fingerprints))


# --- Exact bottom-up minimization (tiny hand-checkable case) ----------------

def test_minimize_tiny_handchecked_case() -> None:
    from slm_training.dsl.analysis.arity import build_trie, minimize
    from slm_training.dsl.analysis.arity.canonical import CanonicalProgram, lit, op
    from slm_training.dsl.analysis.arity.minimize import minimal_completion_lengths

    # Two single-statement programs: root = N+N and root = N*N.
    #   trie = () [=] [=o:+] [=o:+#N] [=o:+#N#N*] and the o:* twin  -> 8 nodes.
    #   the two accepting leaves merge, then their #N parents, then the operator
    #   subtrees, leaving 5 classes: sink, atom-tail, one-atom-left, op-choice,
    #   and the start "=" state.
    p_add = CanonicalProgram((op("+", lit(), lit()),))
    p_mul = CanonicalProgram((op("*", lit(), lit()),))
    trie = build_trie((p_add, p_mul), max_live=0)
    dfa = minimize(trie)
    assert len(trie.nodes) == 8
    assert dfa.class_count == 5
    assert minimal_completion_lengths(dfa) == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}

    # A single program is a pure chain: nothing merges (5 nodes -> 5 classes).
    chain = minimize(build_trie((p_add,), max_live=0))
    assert chain.class_count == 5


# --- StateSignature JSON round-trip + stale-version rejection ----------------

def test_state_signature_json_roundtrip_is_stable() -> None:
    from slm_training.dsl.analysis.arity import SIGNATURE_VERSION, StateSignature

    sig = StateSignature(
        version=SIGNATURE_VERSION,
        generation_order=3,
        grammar_state=("=", "o:+"),
        frontier=("expr", "expr"),
        scope_signature=1,
        expected_type="expr",
        template_state=("N",),
    )
    restored = StateSignature.from_dict(sig.to_dict())
    assert restored == sig
    assert restored.fingerprint() == sig.fingerprint()
    # Fingerprint is deterministic across repeated serialisation.
    assert sig.fingerprint() == StateSignature.from_dict(sig.to_dict()).fingerprint()


def test_state_signature_stale_version_rejected() -> None:
    from slm_training.dsl.analysis.arity import SIGNATURE_VERSION, StateSignature
    from slm_training.dsl.analysis.arity.report import SchemaError

    sig = StateSignature(
        version=SIGNATURE_VERSION,
        generation_order=0,
        grammar_state=(),
        frontier=(),
        scope_signature=0,
        expected_type="stmt_or_end",
        template_state=("N",),
    )
    stale = sig.to_dict()
    stale["version"] = SIGNATURE_VERSION + 99
    with pytest.raises(SchemaError):
        StateSignature.from_dict(stale)


# --- Report schema/version guards ------------------------------------------

def test_report_roundtrips_and_rejects_stale_or_missing_metadata(report) -> None:
    from slm_training.dsl.analysis.arity import ExactArityReport
    from slm_training.dsl.analysis.arity.report import SchemaError

    payload = report.to_dict()
    assert ExactArityReport.from_dict(payload).to_json() == report.to_json()

    stale_schema = report.to_dict()
    stale_schema["schema_version"] = 999
    with pytest.raises(SchemaError):
        ExactArityReport.from_dict(stale_schema)

    stale_signature = report.to_dict()
    stale_signature["signature_version"] = 999
    with pytest.raises(SchemaError):
        ExactArityReport.from_dict(stale_signature)

    missing = report.to_dict()
    del missing["parser_version"]
    with pytest.raises(SchemaError):
        ExactArityReport.from_dict(missing)


# --- K^d capacity: exact integer arithmetic --------------------------------

def test_capacity_uses_exact_integer_arithmetic() -> None:
    from slm_training.dsl.analysis.arity import min_alphabet_for_capacity

    # Boundaries computed with integers only (no float log/pow rounding).
    assert min_alphabet_for_capacity(EXPECT_MINIMIZED_STATES, 4) == EXPECT_MIN_K
    assert min_alphabet_for_capacity(16, 4) == 2  # 2**4 == 16 exactly
    assert min_alphabet_for_capacity(17, 4) == 3  # just over 2**4
    assert min_alphabet_for_capacity(81, 4) == 3  # 3**4 == 81 exactly
    assert min_alphabet_for_capacity(82, 4) == 4  # just over 3**4
    assert min_alphabet_for_capacity(1, 4) == 1
    # The design-doc comparator quotient M=41: (K=2,d=6) and (K=3,d=4) both fit.
    assert min_alphabet_for_capacity(41, 6) == 2  # 2**6 == 64 >= 41
    assert min_alphabet_for_capacity(41, 4) == 3  # 3**4 == 81 >= 41
    with pytest.raises(ValueError):
        min_alphabet_for_capacity(10, 0)


# --- The committed fixture certificate --------------------------------------

def test_committed_fixture_exact_counts(report) -> None:
    assert report.complete is True
    assert report.canonical_ast_count == EXPECT_CANONICAL_ASTS
    assert report.raw_state_count == EXPECT_RAW_STATES
    assert report.trie_state_count == EXPECT_TRIE_STATES
    assert report.minimized_state_count == EXPECT_MINIMIZED_STATES
    assert report.action_alphabet_size == EXPECT_ACTION_ALPHABET
    assert report.scope_signature_count == EXPECT_SCOPE_SIGNATURES
    assert report.max_local_branching == EXPECT_MAX_BRANCHING
    assert report.branching_histogram == EXPECT_BRANCHING_HISTOGRAM
    assert report.completion_counts == EXPECT_COMPLETION_COUNTS
    assert report.forced_visit_fraction["numerator"] == 9
    assert report.forced_visit_fraction["denominator"] == 27
    assert report.capacity == {
        "state_count": EXPECT_MINIMIZED_STATES,
        "d": FIXTURE_DIMENSIONS,
        "min_k": EXPECT_MIN_K,
    }


def test_branching_and_completion_histograms_sum_to_minimized(report) -> None:
    assert sum(report.branching_histogram.values()) == EXPECT_MINIMIZED_STATES
    assert sum(report.completion_counts.values()) == EXPECT_MINIMIZED_STATES


def test_fixture_report_is_byte_stable(report) -> None:
    import hashlib

    from slm_training.dsl.analysis.arity import analyze

    again = analyze(
        fixture="bounded-expr", bounds=_bounds(), dimensions=FIXTURE_DIMENSIONS
    )
    first = report.to_json()
    second = again.to_json()
    assert first == second
    assert hashlib.sha256(first.encode("utf-8")).hexdigest() == hashlib.sha256(
        second.encode("utf-8")
    ).hexdigest()


def test_external_estimates_are_not_reproduced(report) -> None:
    # Honesty guard: the certificate must never coincide with the external
    # source-reported estimates, and must flag that it does not reproduce them.
    assert report.provenance["external_estimates_reproduced"] is False
    assert report.canonical_ast_count != EXTERNAL_ESTIMATES["asts"]
    assert report.trie_state_count != EXTERNAL_ESTIMATES["trie"]
    assert report.minimized_state_count != EXTERNAL_ESTIMATES["minimized"]


# --- Bounds validation + fail-closed CLI ------------------------------------

def test_bounds_reject_negatives() -> None:
    from slm_training.dsl.analysis.arity import AnalysisBounds

    with pytest.raises(ValueError):
        AnalysisBounds(max_ast_nodes=-1)
    with pytest.raises(ValueError):
        AnalysisBounds(max_ast_nodes=6, max_live_bindings=-1)
    with pytest.raises(ValueError):
        AnalysisBounds(max_ast_nodes=6, max_ast_depth=-2)


def test_cli_fails_closed_on_incomplete_enumeration(tmp_path) -> None:
    from scripts.analyze_grammar_arity import main

    out = tmp_path / "scratch.json"
    durable = tmp_path / "durable.json"
    code = main(
        [
            "--fixture", "bounded-expr",
            "--max-ast-nodes", "6",
            "--max-live-bindings", "2",
            "--dimensions", "4",
            "--max-programs", "5",  # force the safety cap -> incomplete
            "--out", str(out),
            "--durable-out", str(durable),
        ]
    )
    assert code == 1
    assert not out.exists()  # nothing certified when incomplete


def test_cli_writes_both_outputs(tmp_path) -> None:
    import json

    from scripts.analyze_grammar_arity import main

    out = tmp_path / "scratch.json"
    durable = tmp_path / "durable.json"
    code = main(
        [
            "--fixture", "bounded-expr",
            "--max-ast-nodes", "6",
            "--max-live-bindings", "2",
            "--dimensions", "4",
            "--out", str(out),
            "--durable-out", str(durable),
        ]
    )
    assert code == 0
    assert out.exists() and durable.exists()
    assert out.read_text(encoding="utf-8") == durable.read_text(encoding="utf-8")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["minimized_state_count"] == EXPECT_MINIMIZED_STATES
    assert payload["schema_version"] == 1


def test_cli_rejects_unknown_fixture_and_zero_nodes(tmp_path) -> None:
    from scripts.analyze_grammar_arity import main

    assert main(["--fixture", "nope", "--max-ast-nodes", "6"]) == 2
    assert main(["--fixture", "bounded-expr", "--max-ast-nodes", "0"]) == 2
