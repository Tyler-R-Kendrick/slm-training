"""Certify that every agent surface carries every repository law.

The repo configures several coding harnesses (Claude Code, Codex, Cursor,
Gemini, Copilot/GHCP, Grok). Each reads a different instruction file, so a law
stated only in ``AGENTS.md`` reaches only the agents that happen to read it.

``verify_decode_invariants`` already proved the pattern works for one law — the
decode invariants are present on every surface. This module generalises it to
the declarative matrix below so the *other* laws cannot drift the way they had
(run cap, iron law, data-quality law, model card, version stamps, dashboard
parity, campaign law were each missing from one or more surface; see
``docs/design/agent-harness-parity-audit.md``).

Markers are deliberately the **skill and script names**, not prose. Those are
what an agent has to know in order to act; requiring the prose instead would
push every surface toward being a third copy of ``AGENTS.md``.

Usage::

    python -m scripts.verify_agent_surfaces
    python -m scripts.verify_agent_surfaces --obligation decode.invariants
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DOC = "docs/design/decode-invariants.md"

# Instruction surfaces that must carry the shared laws. Skill files and the
# Grok workflow carry a narrower set and are named per-obligation below.
PRIMARY_SURFACES: tuple[str, ...] = (
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    ".github/copilot-instructions.md",
    ".cursor/rules/repo-laws.mdc",
)


class AgentSurfaceError(AssertionError):
    """An agent surface stopped carrying a repository law."""


# Harnesses with a committed hook configuration. Cursor has no hook mechanism
# in this repo; that is stated in .cursor/rules/repo-laws.mdc rather than being
# left for an agent to discover.
HOOK_CONFIGS: tuple[str, ...] = (
    ".claude/settings.json",
    ".codex/hooks.json",
)


@dataclass(frozen=True)
class Obligation:
    """One repository law and the markers each surface must carry for it."""

    id: str
    why: str
    requires: tuple[tuple[str, tuple[str, ...]], ...]
    scope: str = "instructions"

    def surfaces(self) -> tuple[str, ...]:
        return tuple(relative for relative, _ in self.requires)


def _on(
    surfaces: tuple[str, ...], *markers: str
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple((relative, markers) for relative in surfaces)


OBLIGATIONS: tuple[Obligation, ...] = (
    Obligation(
        id="autotrain.continuous-harness-loop",
        why="bare autotrain must keep improving models and evidence-implicated canonical harnesses",
        requires=(
            (
                ".agents/skills/autotrain/SKILL.md",
                (
                    "continuous",
                    "improve-openui-harnesses",
                    "improve-lean-optimums",
                    "--matrix",
                ),
            ),
            (
                ".grok/workflows/autotrain.rhai",
                (
                    "continuous",
                    "improve-openui-harnesses",
                    "improve-lean-optimums",
                    "origin/main",
                    "--matrix",
                ),
            ),
        ),
        scope="skills",
    ),
    Obligation(
        id="decode.invariants",
        why="constrained decoding is the product; a silent weakening is a regression",
        requires=(
            ("AGENTS.md", ("Non-negotiable architecture invariants", CANONICAL_DOC)),
            ("CLAUDE.md", ("Non-negotiable architecture invariants", CANONICAL_DOC)),
            ("GEMINI.md", ("Non-negotiable architecture invariants", CANONICAL_DOC)),
            (
                ".github/copilot-instructions.md",
                ("Non-negotiable architecture invariants", "AGENTS.md"),
            ),
            (
                ".cursor/rules/decode-invariants.mdc",
                ("alwaysApply: true", CANONICAL_DOC),
            ),
            (".agents/skills/autotrain/SKILL.md", ("decode-invariants.md",)),
            (".agents/skills/honest-ship-eval/SKILL.md", ("decode-invariants.md",)),
            (
                ".agents/skills/improve-openui-harnesses/SKILL.md",
                ("decode-invariants.md",),
            ),
            (
                ".agents/skills/running-experiment-matrices/SKILL.md",
                ("decode-invariants.md",),
            ),
            (".grok/workflows/autotrain.rhai", ("decode-invariants.md",)),
        ),
    ),
    Obligation(
        id="canonical.agents-md",
        why="tool-specific defaults must lose to AGENTS.md on process conflicts",
        requires=_on(
            tuple(s for s in PRIMARY_SURFACES if s != "AGENTS.md")
            + (".grok/workflows/autotrain.rhai",),
            "AGENTS.md",
        ),
    ),
    Obligation(
        id="run.cap",
        why="a timed out, interrupted, or killed run is never evidence",
        requires=_on(
            PRIMARY_SURFACES + (".grok/workflows/autotrain.rhai",), "MAX_RUN_MINUTES"
        ),
    ),
    Obligation(
        id="docs.iron-law",
        why="numbers only in outputs/, chat, or a PR comment are incomplete work",
        requires=_on(
            PRIMARY_SURFACES + (".grok/workflows/autotrain.rhai",),
            "documenting-experiment-results",
        ),
    ),
    Obligation(
        id="gates.honest-ship",
        why="fixture demo is not a production ship, and gates are never weakened to green CI",
        requires=_on(PRIMARY_SURFACES, "honest-ship-eval"),
    ),
    Obligation(
        id="data.quality-law",
        why="every data build reads its own quality report and fixes the synthesis harness",
        requires=_on(PRIMARY_SURFACES, "synthesis-feedback"),
    ),
    Obligation(
        id="checkpoint.model-card",
        why="a checkpoint without a model-card and README summary update is incomplete work",
        requires=_on(
            PRIMARY_SURFACES + (".grok/workflows/autotrain.rhai",), "MODEL_CARD.md"
        ),
    ),
    Obligation(
        id="versioning.component-bump",
        why="results must say which revision of the constraints produced them",
        requires=_on(PRIMARY_SURFACES, "verify_version_stamps"),
    ),
    Obligation(
        id="test-cases.agent-refresh",
        why="agents refresh committed cases while ordinary tests and CI stay read-only",
        requires=_on(PRIMARY_SURFACES, "refresh_test_cases"),
    ),
    Obligation(
        id="dashboard.openui-parity",
        why="a page change that leaves interpreted mode wrong is incomplete work",
        requires=_on(PRIMARY_SURFACES, "dashboard-openui-parity"),
    ),
    Obligation(
        id="campaign.preregistration",
        why="the locked confirmatory endpoint may never be replaced after outcomes are visible",
        requires=_on(PRIMARY_SURFACES, "ExperimentCampaignV1"),
    ),
    Obligation(
        id="repo.organization",
        why="tracked relocations use git mv and follow the placement guide",
        requires=_on(PRIMARY_SURFACES, "organize-repository", "git mv"),
    ),
    Obligation(
        id="skills.canonical-root",
        why="skills are edited only under .agents/skills; the rest are discovery symlinks",
        requires=_on(PRIMARY_SURFACES, ".agents/skills"),
    ),
    # Automated feedback, not just prose. These had been Claude-Code-only, so a
    # Codex or Copilot agent editing a dashboard page or a watched harness file
    # got no signal at all.
    Obligation(
        id="hooks.raw-mv-guard",
        why="tracked paths must move with git mv; every harness blocks raw mv",
        requires=_on(
            HOOK_CONFIGS + (".github/hooks/repo-policy.json",),
            "scripts.repo_policy --pre-tool-use",
        ),
        scope="hooks",
    ),
    Obligation(
        id="hooks.post-edit-checks",
        why="dashboard parity and component bumps need automated feedback on every harness",
        requires=_on(
            HOOK_CONFIGS + (".github/hooks/changed-tests.json",),
            "validate_page_dsl.py --changed",
            "scripts.verify_version_stamps --post-tool-use",
            "scripts.refresh_test_cases --check --changed",
        ),
        scope="hooks",
    ),
)


def _read(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise AgentSurfaceError(
            f"missing agent surface: {relative}; every configured harness needs "
            "an instruction surface carrying the repository laws"
        )
    return path.read_text(encoding="utf-8")


def check(*, obligation_id: str | None = None) -> dict[str, list[str]]:
    """Return {obligation id: surfaces checked}, raising if anything is missing.

    Reports *every* gap rather than the first, so backfilling a surface is one
    pass instead of one run per missing marker.
    """
    selected = [o for o in OBLIGATIONS if obligation_id in (None, o.id)]
    if obligation_id is not None and not selected:
        raise AgentSurfaceError(f"unknown obligation id: {obligation_id}")
    checked: dict[str, list[str]] = {}
    gaps: list[str] = []
    for obligation in selected:
        for relative, markers in obligation.requires:
            try:
                text = _read(relative)
            except AgentSurfaceError as exc:
                gaps.append(str(exc))
                continue
            missing = [marker for marker in markers if marker not in text]
            if missing:
                gaps.append(
                    f"{relative} does not carry "
                    f"{', '.join(repr(m) for m in missing)} for obligation "
                    f"{obligation.id!r} ({obligation.why})"
                )
        checked[obligation.id] = list(obligation.surfaces())
    if gaps:
        raise AgentSurfaceError(
            "every configured harness must enforce the same laws (see AGENTS.md "
            "and docs/design/agent-harness-parity-audit.md):\n- " + "\n- ".join(gaps)
        )
    return checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obligation",
        help="certify a single obligation id instead of the whole matrix",
    )
    args = parser.parse_args(argv)
    try:
        checked = check(obligation_id=args.obligation)
    except AgentSurfaceError as exc:
        print(f"agent-surfaces: failed\n- {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "obligations": len(checked),
                "surfaces": sorted({s for v in checked.values() for s in v}),
                "checked": {k: sorted(v) for k, v in checked.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
