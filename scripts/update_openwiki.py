"""Run OpenWiki while keeping its generated pages under docs/openwiki."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WIKI = ROOT / "docs" / "openwiki"
OPENWIKI_LINK = ROOT / "openwiki"
SCAFFOLD_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path(".github/workflows/openwiki-update.yml"),
)


def run_openwiki(args: list[str], *, root: Path = ROOT) -> int:
    wiki = root / "docs" / "openwiki"
    link = root / "openwiki"
    if link.exists() or link.is_symlink():
        raise RuntimeError(f"temporary OpenWiki path already exists: {link}")
    snapshots = {
        relative: (root / relative).read_bytes()
        for relative in SCAFFOLD_PATHS
        if (root / relative).is_file()
    }
    link.symlink_to(Path("docs/openwiki"), target_is_directory=True)
    try:
        return subprocess.run(["openwiki", "code", *args], cwd=root, check=False).returncode
    finally:
        for relative, content in snapshots.items():
            (root / relative).write_bytes(content)
        link.unlink(missing_ok=True)
        if not wiki.is_dir():
            raise RuntimeError(f"OpenWiki did not preserve its docs directory: {wiki}")


def main(argv: list[str] | None = None) -> int:
    return run_openwiki(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
