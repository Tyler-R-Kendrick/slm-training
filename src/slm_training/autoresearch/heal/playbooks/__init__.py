"""Discovered heal playbooks — every module exposing ``PLAYBOOK`` attaches.

Contract mirror of ``autoresearch/preflight``: presence is discovery, absence
is never an error, and one broken module never blocks its siblings. Playbook
law lives in ``heal/__init__.py``; each module documents which blocker class
it handles and why its repair is deterministic.
"""
