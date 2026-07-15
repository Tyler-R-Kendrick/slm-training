from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from slm_training.data.verify import (
    Gate,
    GateStatus,
    RuntimeEvidence,
    Tier,
    VerificationContext,
    evaluate_gate,
    run_preview_verifier,
    stamp_record,
    verify_record,
)
from slm_training.dsl.lang_core import Program, bridge_available
from slm_training.dsl.schema import ExampleRecord


VALID = 'root = Stack([cta], "column")\ncta = Button(":cta.label")'


def record(source: str = VALID, *, kind: str = "program-first") -> ExampleRecord:
    return ExampleRecord(
        id="verify-1",
        prompt="Build a call to action",
        openui=source,
        placeholders=[":cta.label"],
        source=kind,
    )


@pytest.mark.parametrize(
    ("gate", "valid_context", "invalid_record", "invalid_context"),
    [
        (
            Gate.LEXICAL,
            VerificationContext(),
            record('root = TextContent(":ok)'),
            VerificationContext(),
        ),
        (Gate.GRAMMAR, VerificationContext(), record("root = Stack([cta]"), VerificationContext()),
        (Gate.REFERENCES, VerificationContext(), record("root = Stack([missing])"), VerificationContext()),
        (Gate.DATAFLOW, VerificationContext(), record('$state = "x"\nroot = TextContent(":ok")'), VerificationContext()),
        (
            Gate.RUNTIME,
            VerificationContext(runtime=RuntimeEvidence(rendered=True)),
            record(),
            VerificationContext(runtime=RuntimeEvidence(rendered=False)),
        ),
        (
            Gate.BEHAVIOR,
            VerificationContext(
                runtime=RuntimeEvidence(rendered=True, interaction_trace=("click:button",)),
                require_behavior=True,
            ),
            record(),
            VerificationContext(
                runtime=RuntimeEvidence(
                    rendered=True,
                    behavior_errors=("seeded behavior error",),
                )
            ),
        ),
        (
            Gate.GROUNDING,
            VerificationContext(
                required_facts=("component:Button",),
                forbidden_facts=("component:ImageBlock",),
            ),
            record(),
            VerificationContext(required_facts=("component:ImageBlock",)),
        ),
        (
            Gate.PATCH,
            VerificationContext(
                patch_before="old",
                patch="new",
                patch_after="new",
                patch_applier=lambda _before, patch: patch,
            ),
            record(),
            VerificationContext(
                patch_before="old",
                patch="wrong",
                patch_after="new",
                patch_applier=lambda _before, patch: patch,
            ),
        ),
        (Gate.PROVENANCE, VerificationContext(), record(), VerificationContext(provenance_complete=False)),
        (
            Gate.INDEPENDENT_JUDGE,
            VerificationContext(independent_judge_passed=True),
            record(),
            VerificationContext(independent_judge_passed=False),
        ),
        (
            Gate.HUMAN_AUDIT,
            VerificationContext(human_audit_passed=True),
            record(),
            VerificationContext(human_audit_passed=False),
        ),
    ],
)
def test_gate_passes_valid_and_rejects_targeted_invalid(
    gate: Gate,
    valid_context: VerificationContext,
    invalid_record: ExampleRecord,
    invalid_context: VerificationContext,
) -> None:
    assert evaluate_gate(gate, record(), valid_context).status is GateStatus.PASS
    assert evaluate_gate(gate, invalid_record, invalid_context).status is GateStatus.FAIL


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge dependencies unavailable")
def test_schema_gate_passes_known_and_rejects_unknown_component() -> None:
    assert evaluate_gate(Gate.SCHEMA, record()).status is GateStatus.PASS
    assert evaluate_gate(Gate.SCHEMA, record("root = Broken()")).status is GateStatus.FAIL


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge dependencies unavailable")
def test_canonical_gate_passes_and_detects_non_idempotence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert evaluate_gate(Gate.CANONICAL, record()).status is GateStatus.PASS
    serializations = iter(("first", "second"))
    monkeypatch.setattr(
        "slm_training.data.verify.stack.validate",
        lambda _source: Program(source="", serialized=next(serializations)),
    )
    assert evaluate_gate(Gate.CANONICAL, record()).status is GateStatus.FAIL


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge dependencies unavailable")
def test_tiers_and_record_stamp_are_deterministic() -> None:
    silver = verify_record(record())
    assert silver.tier is Tier.SILVER
    assert silver.failing_gate is None
    assert verify_record(record()).to_dict() == silver.to_dict()

    assert verify_record(record(kind="teacher")).tier is Tier.BRONZE
    assert verify_record(
        record(), VerificationContext(human_audit_passed=True)
    ).tier is Tier.GOLD
    quarantined = verify_record(
        record(), VerificationContext(provenance_complete=False)
    )
    assert quarantined.tier is Tier.QUARANTINE
    assert quarantined.failing_gate is Gate.PROVENANCE

    stamped = stamp_record(record())
    assert stamped.meta["verification_tier"] == "Silver"
    assert stamped.meta["failing_gate"] is None
    assert len(stamped.meta["verification"]["gates"]) == 13


def _browser_ready() -> bool:
    root = Path(__file__).resolve().parents[2]
    node = shutil.which("node")
    if not node or not (root / "node_modules" / "@playwright" / "test").exists():
        return False
    try:
        probe = subprocess.run(
            [
                node,
                "--input-type=module",
                "-e",
                "import { chromium } from '@playwright/test'; "
                "try { const browser = await chromium.launch({ headless: true }); "
                "await browser.close(); } catch { process.exit(1); }",
            ],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return False
    return probe.returncode == 0


@pytest.mark.skipif(not _browser_ready(), reason="Playwright/preview dependencies unavailable")
def test_preview_runtime_and_behavior_seeded_failures() -> None:
    clean = run_preview_verifier(VALID)
    assert clean.rendered
    assert clean.console_errors == ()
    assert clean.behavior_errors == ()
    assert clean.interaction_trace == ("click:button",)

    console = run_preview_verifier(VALID, seed_console_error=True)
    assert evaluate_gate(
        Gate.RUNTIME,
        record(),
        VerificationContext(runtime=console),
    ).status is GateStatus.FAIL

    behavior = run_preview_verifier(VALID, seed_behavior_error=True)
    assert evaluate_gate(
        Gate.BEHAVIOR,
        record(),
        VerificationContext(runtime=behavior),
    ).status is GateStatus.FAIL
