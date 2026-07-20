"""Structural guards for the autoresearch orchestration skill and its brains.

The skill is instruction-only (no CLI of its own), so these tests keep it honest:
frontmatter sanity, discovery symlinks, routing-table references on disk, and
resolvable relative links across the skill and the docs/brains scaffold.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / ".agents" / "skills" / "autoresearch"
REFERENCES = SKILL_DIR / "references"
BRAINS = REPO / "docs" / "brains"

EXPECTED_REFERENCES = {
    "brains",
    "openwiki",
    "discovery",
    "hypothesis",
    "linear",
    "loop",
    "contracts",
}


def _frontmatter(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "---", path
    closing = lines.index("---", 1)
    return dict(
        line.split(":", 1) for line in lines[1:closing] if ":" in line
    )


def test_frontmatter_sane() -> None:
    fields = _frontmatter(SKILL_DIR / "SKILL.md")
    assert fields["name"].strip() == "autoresearch"
    description = fields["description"].strip()
    assert description
    assert len(description) <= 1024


def test_discovery_symlinks() -> None:
    for root in (".claude", ".cursor"):
        link = REPO / root / "skills" / "autoresearch"
        assert link.is_symlink(), link
        assert os.readlink(link) == "../../.agents/skills/autoresearch"


def test_reference_set_matches_disk() -> None:
    on_disk = {path.stem for path in REFERENCES.glob("*.md")}
    assert on_disk == EXPECTED_REFERENCES


def test_skill_routes_to_every_reference() -> None:
    skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    for slug in EXPECTED_REFERENCES:
        assert f"references/{slug}.md" in skill, slug


def test_relative_links_resolve() -> None:
    files = list(SKILL_DIR.rglob("*.md")) + list(BRAINS.rglob("*.md"))
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    broken: list[str] = []
    for f in files:
        for match in link_re.finditer(f.read_text(encoding="utf-8")):
            target = match.group(1).split("#")[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (f.parent / target).resolve().exists():
                broken.append(f"{f.relative_to(REPO)} -> {target}")
    assert not broken, "broken relative links: " + "; ".join(broken)


def test_brains_scaffold_present() -> None:
    for rel in (
        "README.md",
        "repo/MOC.md",
        "personal/README.md",
        "personal/example/home.md",
        "templates/concept-note.md",
        "templates/source-note.md",
        "templates/experiment-idea.md",
    ):
        assert (BRAINS / rel).is_file(), rel
