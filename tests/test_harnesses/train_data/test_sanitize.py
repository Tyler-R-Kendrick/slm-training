"""Record-level sanitization pass tests (harnesses.train_data.sanitize)."""

from __future__ import annotations

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.train_data.pipeline import _normalize_record
from slm_training.harnesses.train_data.integrity import evaluate_integrity
from slm_training.harnesses.train_data.sanitize import (
    SanitizeOptions,
    aggregate_sanitization,
    sanitize_openui,
    should_sanitize,
)

ENFORCE = SanitizeOptions(mode="enforce")

LITERAL_LAYOUT = (
    'root = Stack([hero, note], "column")\n'
    "hero = Card([hdr])\n"
    'hdr = CardHeader("Welcome back")\n'
    'note = TextContent("Sign in to continue")\n'
    'unused = TextContent(":dead.slot")'
)


def _record(openui: str, **kwargs: object) -> ExampleRecord:
    return ExampleRecord(
        id="test-1",
        prompt="Build a welcome layout.",
        openui=openui,
        split="train",
        source="fixture",
        **kwargs,
    )


def test_should_sanitize_skip_matrix() -> None:
    plain = _record("root = Stack([])")
    assert should_sanitize(plain) == (True, "")

    verbatim = _record("root = Stack([])", meta={"preserve_verbatim": True})
    assert should_sanitize(verbatim) == (False, "preserve_verbatim")

    scoped = _record("root = Stack([])", meta={"scope_slice": {"scope": "document"}})
    assert should_sanitize(scoped) == (False, "scope_slice")

    for task in ("identity", "repair", "edit"):
        record = _record("root = Stack([])", meta={"task": task})
        assert should_sanitize(record) == (False, f"task_{task}")

    # Edit/repair-*derived* generation rows are eligible.
    derived = _record(
        "root = Stack([])", meta={"task": "generation", "edit": {"op": "add"}}
    )
    assert should_sanitize(derived) == (True, "")

    fragment = _record("Button(:a.b)", target_kind="expression")
    assert should_sanitize(fragment) == (False, "non_document")


def test_sanitize_rescues_literals_and_dead_bindings() -> None:
    outcome = sanitize_openui(
        LITERAL_LAYOUT, prompt="Build a welcome layout.", options=ENFORCE
    )
    assert outcome.applied and not outcome.fallback
    assert outcome.changed
    assert ":dead.slot" not in outcome.openui
    assert '"Welcome back"' not in outcome.openui
    assert outcome.rewrites["dead_bindings_removed"] == 1
    assert outcome.literals_templatized == 2
    assert set(outcome.template_fills.values()) == {
        "Welcome back",
        "Sign in to continue",
    }


def test_sanitize_is_idempotent() -> None:
    once = sanitize_openui(LITERAL_LAYOUT, options=ENFORCE)
    twice = sanitize_openui(once.openui, options=ENFORCE)
    assert twice.openui == once.openui
    assert not twice.changed


def test_sanitize_falls_back_on_unparseable_input() -> None:
    broken = 'root = Card(["unclosed'
    outcome = sanitize_openui(broken, options=ENFORCE)
    assert outcome.fallback and not outcome.applied
    assert outcome.openui == broken
    assert outcome.reasons


def test_sanitized_record_passes_integrity_checks() -> None:
    outcome = sanitize_openui(LITERAL_LAYOUT, options=ENFORCE)
    record = _record(outcome.openui, placeholders=list(outcome.placeholders))
    report = evaluate_integrity(record)
    assert report.passed, [c.to_dict() for c in report.checks if c.status.value == "fail"]


def test_normalize_record_seam_rescues_dead_binding_quarantine() -> None:
    seed = _record(LITERAL_LAYOUT, placeholders=[":dead.slot"])

    try:
        baseline = _normalize_record(seed, sanitize=None)
        baseline_state = str(baseline.meta.get("verification_tier"))
    except Exception:  # noqa: BLE001 - strict parsers may reject outright
        baseline_state = "rejected"
    # Without sanitization the candidate is unusable: either the official
    # parser rejects the content-policy literals or G3 quarantines the dead
    # binder downstream.
    assert baseline_state in {"rejected", "Quarantine"}

    admitted = _normalize_record(seed, sanitize=ENFORCE)
    assert str(admitted.meta.get("verification_tier")) not in {"Quarantine", "None"}
    assert ":dead.slot" not in admitted.openui
    assert '"Welcome back"' not in admitted.openui
    block = admitted.meta.get("sanitize")
    assert isinstance(block, dict) and block["applied"] and block["mode"] == "enforce"
    assert block["rewrites"]["dead_bindings_removed"] == 1
    # New placeholders flowed into the declared inventory (honest slot contract).
    assert set(block["template_fills"]).issubset(set(admitted.placeholders))


def test_normalize_record_audit_mode_keeps_bytes() -> None:
    clean = (
        'root = Stack([hero], "column")\n'
        "hero = Card([hdr])\n"
        'hdr = CardHeader(":w.title")'
    )
    seed = _record(clean, placeholders=[":w.title"])
    audit = _normalize_record(seed, sanitize=SanitizeOptions(mode="audit"))
    plain = _normalize_record(seed, sanitize=None)
    assert audit.openui == plain.openui
    block = audit.meta.get("sanitize")
    assert isinstance(block, dict) and block["mode"] == "audit"
    # The outcome still reports what enforce WOULD have done.
    assert block["rewrites"]["defaults_elided"] == 1


def test_normalize_record_skips_scope_slices_byte_identically() -> None:
    scoped = _record(
        'root = Stack([t])\nt = TextContent(":a.b")',
        placeholders=[":a.b"],
        meta={"scope_slice": {"scope": "document"}, "task": "generation"},
    )
    sanitized = _normalize_record(scoped, sanitize=ENFORCE)
    plain = _normalize_record(scoped, sanitize=None)
    assert sanitized.openui == plain.openui
    assert sanitized.meta["sanitize"] == {"mode": "enforce", "skip_reason": "scope_slice"}


def test_aggregate_sanitization_shape() -> None:
    outcome = sanitize_openui(LITERAL_LAYOUT, options=ENFORCE)
    touched = _record(
        outcome.openui, meta={"sanitize": outcome.to_meta("enforce")}
    )
    skipped = _record(
        "root = Stack([])",
        meta={"sanitize": {"mode": "enforce", "skip_reason": "scope_slice"}},
    )
    verbatim = _record("root = Stack([])", meta={"preserve_verbatim": True})
    section = aggregate_sanitization([touched, skipped, verbatim], mode="enforce")
    assert section["mode"] == "enforce"
    assert section["sanitized"] == 1
    assert section["changed"] == 1
    assert section["skipped"] == {"preserve_verbatim": 1, "scope_slice": 1}
    assert section["rewrites"]["dead_bindings_removed"] == 1
    assert section["literals_templatized"] == 2
    assert section["records_templatized"] == 1
    assert section["fallbacks"] == 0
    assert set(section["cache"]) == {"hits", "misses"}
