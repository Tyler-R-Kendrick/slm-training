"""Discovered heal playbooks — every module exposing ``PLAYBOOK`` attaches.

Contract mirror of ``autoresearch/preflight``: presence is discovery, absence
is never an error, and one broken module never blocks its siblings. Playbook
law lives in ``heal/__init__.py``; each module documents which blocker class
it handles and why its repair is deterministic.

Registered playbooks (module → blocker class):

- ``npm_bridges`` → ``environment`` (documented ``npm ci`` bridge repair)
- ``quarantine_dirt`` → ``dirty_tree`` (guarded, reversible stash)
- ``harness_crash`` → ``code`` (crash triage: traceback capture + typed
  ``repair_harness`` action + one module test run; receipt ``attempted``,
  never ``healed``)
- ``data_rebuild`` → ``data`` (existing rebuild seam under a measured
  ``records_after > records_before`` postcondition)

Playbooks may additionally expose ``execute(blocker, *, cwd, root, loop_id,
campaign_id) -> HealAttemptReceiptV1`` for in-process attempts whose receipt
outcome is not decided by a subprocess verify probe; ``plan`` stays the
runner-compatible fallback.
"""

REGISTERED_PLAYBOOK_MODULES: tuple[str, ...] = (
    "data_rebuild",
    "harness_crash",
    "npm_bridges",
    "quarantine_dirt",
)

__all__ = ["REGISTERED_PLAYBOOK_MODULES"]
