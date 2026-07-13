#!/usr/bin/env python3
"""Run the GPU multi-farm MCP server (stdio)."""

from __future__ import annotations


def main() -> None:
    from gpu_multi_farm.server import mcp

    mcp.run()


if __name__ == "__main__":
    main()
