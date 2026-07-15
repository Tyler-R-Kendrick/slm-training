#!/usr/bin/env python3
"""Run one upstream researcher inside its isolated Python environment."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "content"):
        return {"type": type(value).__name__, "content": _jsonable(value.content)}
    return str(value)


async def _open_deep_research(checkout: Path, request: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(checkout / "src"))
    from open_deep_research.deep_researcher import deep_researcher

    config = dict(request["config"])
    config["allow_clarification"] = False
    result = await deep_researcher.ainvoke(
        {"messages": [{"role": "user", "content": request["prompt"]}]},
        {"configurable": config},
    )
    memo = str(result.get("final_report") or "")
    if memo.startswith("Error generating final report:"):
        raise RuntimeError(memo)
    return {
        "memo": memo,
        "trace": _jsonable(result),
        "telemetry": {"graph": "deep_researcher", "config": config},
    }


async def _open_researcher(checkout: Path, request: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(checkout))
    from deploy_agent import BrowserPool, run_one
    from utils.openai_generator import OpenAIAsyncGenerator

    config = request["config"]
    generator = OpenAIAsyncGenerator(
        base_url=config["base_url"],
        model_name=config["model"],
        use_native_tools=True,
    )
    browser_pool = BrowserPool(
        search_url=config.get("search_url"),
        browser_backend=config["browser_backend"],
    )
    try:
        messages = await run_one(
            question=request["prompt"],
            qid=request["campaign_id"],
            generator=generator,
            browser_pool=browser_pool,
            max_rounds=config["max_rounds"],
        )
    finally:
        client = getattr(generator, "client", None)
        if client is not None and hasattr(client, "aclose"):
            await client.aclose()
    memo = ""
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            memo = str(message["content"])
            break
    return {
        "memo": memo,
        "trace": {"messages": _jsonable(messages)},
        "telemetry": {
            "model": config["model"],
            "browser_backend": config["browser_backend"],
            "max_rounds": config["max_rounds"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("open-deep-research", "open-researcher"), required=True
    )
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if args.backend == "open-deep-research":
        result = asyncio.run(_open_deep_research(args.checkout, request))
    else:
        result = asyncio.run(_open_researcher(args.checkout, request))
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
