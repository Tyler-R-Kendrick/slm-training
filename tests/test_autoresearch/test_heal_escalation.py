"""Escalation ledger: cross-campaign identity, governor math, advisory-only."""

from __future__ import annotations

from pathlib import Path

from slm_training.autoresearch.heal import escalation as esc
from slm_training.autoresearch.heal.escalation import (
    EscalationLedger,
    blocker_fingerprint,
    next_backoff_seconds,
)


class TestFingerprint:
    def test_campaign_is_payload_never_key(self) -> None:
        # The same durable blocker seen from two campaigns keeps one identity
        # (the state.json fingerprint was campaign-bound and never dedduped).
        a = blocker_fingerprint("repair_harness", "AgentV SDK is unavailable")
        b = blocker_fingerprint("repair_harness", "AgentV SDK is unavailable")
        assert a == b

    def test_volatile_tokens_are_normalized(self) -> None:
        a = blocker_fingerprint(
            "repair_harness", "crash in campaign c1832 sha " + "a" * 64
        )
        b = blocker_fingerprint(
            "repair_harness", "crash in campaign c1901 sha " + "b" * 64
        )
        assert a == b

    def test_distinct_kinds_are_distinct(self) -> None:
        assert blocker_fingerprint(
            "repair_harness", "x"
        ) != blocker_fingerprint("repair_formal", "x")


class TestGovernor:
    def test_backoff_doubles_to_cap(self) -> None:
        assert next_backoff_seconds(1) == 30
        assert next_backoff_seconds(2) == 60
        assert next_backoff_seconds(3) == 120
        assert next_backoff_seconds(7) == 1920
        assert next_backoff_seconds(8) == 3600  # 3840 capped at 1h
        assert next_backoff_seconds(20) == 3600

    def test_backoff_monotone(self) -> None:
        values = [next_backoff_seconds(n) for n in range(1, 24)]
        assert values == sorted(values)
        assert max(values) == 3600


class TestLedger:
    def test_save_rewrites_same_fingerprint(self, tmp_path: Path) -> None:
        ledger = EscalationLedger.load(tmp_path, "loop-1")
        first = ledger.observe(
            kind="foreign_dirty_tree",
            reason="non-closeout dirty paths: ['.serena/project.yml']",
            blocker_class="dirty_tree",
            campaign_id="c1",
        )
        ledger.escalate(first.fingerprint, note="first")
        ledger.save()
        ledger.observe(
            kind="foreign_dirty_tree",
            reason="non-closeout dirty paths: ['.serena/project.yml']",
            blocker_class="dirty_tree",
            campaign_id="c2",
        )
        ledger.save()
        path = EscalationLedger.path_for(tmp_path, "loop-1")
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 1
        folded = EscalationLedger.load(tmp_path, "loop-1").records
        assert len(folded) == 1
        record = next(iter(folded.values()))
        assert record.seen_count == 2
        assert record.campaign_ids == ("c1", "c2")

    def test_observe_dedups_and_accumulates(self, tmp_path: Path) -> None:
        ledger = EscalationLedger.load(tmp_path, "loop-1")
        first = ledger.observe(
            kind="repair_harness",
            reason="AgentV SDK is unavailable",
            blocker_class="environment",
            campaign_id="c1",
        )
        second = ledger.observe(
            kind="repair_harness",
            reason="AgentV SDK is unavailable",
            blocker_class="environment",
            campaign_id="c2",
        )
        assert first.fingerprint == second.fingerprint
        assert second.seen_count == 2
        assert second.campaign_ids == ("c1", "c2")
        assert second.next_backoff_seconds == 60

    def test_roundtrip_persists_fold(self, tmp_path: Path) -> None:
        ledger = EscalationLedger.load(tmp_path, "loop-1")
        record = ledger.observe(
            kind="repair_formal",
            reason="theorem_backed_band_miss",
            blocker_class="formal_contradiction",
            campaign_id="c1",
        )
        ledger.record_attempt(record.fingerprint, "some_playbook/v1")
        ledger.escalate(record.fingerprint, note="needs improve-lean-optimums")
        ledger.save()

        reloaded = EscalationLedger.load(tmp_path, "loop-1")
        folded = reloaded.records[record.fingerprint]
        assert folded.status == "escalated"
        assert folded.attempts == 1
        assert folded.playbooks_tried == ("some_playbook/v1",)
        assert folded.owner_skill == "improve-lean-optimums"

    def test_resolved_reopens_on_new_sighting(self, tmp_path: Path) -> None:
        ledger = EscalationLedger.load(tmp_path, "loop-1")
        record = ledger.observe(
            kind="repair_harness",
            reason="npm ci needed",
            blocker_class="environment",
            campaign_id="c1",
        )
        ledger.resolve(record.fingerprint, note="healed")
        again = ledger.observe(
            kind="repair_harness",
            reason="npm ci needed",
            blocker_class="environment",
            campaign_id="c3",
        )
        assert again.status == "open"

    def test_sleep_seconds_uses_max_open_backoff(self, tmp_path: Path) -> None:
        ledger = EscalationLedger.load(tmp_path, "loop-1")
        assert ledger.sleep_seconds(default=7.0) == 7.0
        record = ledger.observe(
            kind="repair_harness",
            reason="x",
            blocker_class="code",
            campaign_id="c1",
        )
        ledger.observe(
            kind="repair_harness",
            reason="x",
            blocker_class="code",
            campaign_id="c1",
        )
        assert ledger.sleep_seconds() == 60.0
        stale = ledger.observe(
            kind="repair_harness",
            reason="stale",
            blocker_class="code",
            campaign_id="old",
        )
        for _ in range(7):
            stale = ledger.observe(
                kind="repair_harness",
                reason="stale",
                blocker_class="code",
                campaign_id="old",
            )
        assert ledger.sleep_seconds() == 3600.0
        assert ledger.sleep_seconds(fingerprints={record.fingerprint}) == 60.0
        ledger.resolve(record.fingerprint)
        assert ledger.sleep_seconds(
            default=9.0, fingerprints={record.fingerprint}
        ) == 9.0

    def test_malformed_rows_never_block_the_fold(self, tmp_path: Path) -> None:
        path = EscalationLedger.path_for(tmp_path, "loop-1")
        path.parent.mkdir(parents=True)
        path.write_text('{"not": "a record"}\ngarbage\n', encoding="utf-8")
        ledger = EscalationLedger.load(tmp_path, "loop-1")
        assert ledger.records == {}


class TestAdvisoryOnly:
    def test_ledger_exposes_no_downgrade_authority(self) -> None:
        # RC4 guard: the ledger schedules and routes; it must never be able to
        # acknowledge, soften, or expire a hard action. No public API may
        # touch handoffs, receipts, or action state.
        forbidden_tokens = ("acknowledge", "handoff", "receipt", "action", "waiv")
        public = [
            name
            for name in dir(EscalationLedger)
            if not name.startswith("_")
        ]
        for name in public:
            lowered = name.lower()
            assert not any(tok in lowered for tok in forbidden_tokens), name
        # Restrict the module scan to names esc itself defines — imported
        # helpers must not fail this test without a real authority change.
        module_public = [
            name
            for name in dir(esc)
            if not name.startswith("_")
            and getattr(getattr(esc, name), "__module__", esc.__name__)
            == esc.__name__
        ]
        for name in module_public:
            lowered = name.lower()
            assert not any(tok in lowered for tok in forbidden_tokens), name
