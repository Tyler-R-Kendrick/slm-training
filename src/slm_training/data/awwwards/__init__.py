"""Awwwards / award-site scraping → ExampleRecord (+ DESIGN.md)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from slm_training.data.design_md.extract import extract_and_filter
from slm_training.dsl.schema import ExampleRecord

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class AwwwardsConfig:
    cache_dir: Path = Path("fixtures/awwwards/cache")
    fixture_path: Path = Path("fixtures/awwwards/sites.jsonl")
    rate_limit_s: float = 1.0
    user_agent: str = "slm-training-awwwards/0.1 (+research; respectful)"
    live: bool = False
    max_sites: int = 20


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")[:48] or "site"


def load_fixture_sites(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _heuristic_openui(site: dict[str, Any]) -> tuple[str, list[str]]:
    """Map site metadata to a placeholder openuiLibrary skeleton."""
    name = _slug(str(site.get("title") or site.get("id") or "site")).replace("-", "_")
    if name[0].isdigit():
        name = f"site_{name}"
    tags = [str(t).lower() for t in (site.get("tags") or [])]
    wants_form = any(t in tags for t in ("form", "signup", "login", "contact"))
    wants_image = any(t in tags for t in ("image", "photo", "gallery", "visual"))
    children: list[str] = ["hero"]
    lines = [
        'hero_title = TextContent(":hero.title", "large-heavy")',
        'hero_body = TextContent(":hero.body")',
        "hero = Card([hero_title, hero_body])",
    ]
    placeholders = [":hero.title", ":hero.body"]
    if wants_image:
        children.append("visual")
        lines.append('visual = ImageBlock(":assets.hero", ":hero.alt")')
        placeholders.extend([":assets.hero", ":hero.alt"])
    if wants_form:
        children.extend(["email", "submit"])
        lines.append(f'email = Input("email", ":{name}.email.placeholder")')
        lines.append(f'submit = Button(":{name}.submit")')
        placeholders.extend([f":{name}.email.placeholder", f":{name}.submit"])
    else:
        children.append("cta")
        lines.append(f'cta = Button(":{name}.cta")')
        placeholders.append(f":{name}.cta")
    root = f'root = Stack([{", ".join(children)}], "column", "m")'
    return "\n".join([root, *lines]), placeholders


def site_to_record(
    site: dict[str, Any],
    *,
    split: str = "train",
    attach_design_md: bool = True,
) -> ExampleRecord:
    openui, placeholders = _heuristic_openui(site)
    title = str(site.get("title") or "Award site")
    prompt = (
        f"Build a polished {title} landing layout in OpenUI with placeholders "
        f"(inspired by award-site metadata)."
    )
    design_md = None
    design_meta: dict[str, Any] = {}
    if attach_design_md:
        design_md, design_meta = extract_and_filter(
            title=title,
            description=str(site.get("description") or ""),
            tags=list(site.get("tags") or []),
        )
    record_id = str(site.get("id") or f"awwwards_{_slug(title)}")
    return ExampleRecord(
        id=record_id,
        prompt=prompt,
        openui=openui,
        placeholders=placeholders,
        split=split,
        source="awwwards",
        meta={
            "title": title,
            "url": site.get("url"),
            "tags": site.get("tags") or [],
            "design_lint": {
                "score": design_meta.get("score"),
                "summary": design_meta.get("summary"),
            },
        },
        design_md=design_md,
    )


def fetch_url(url: str, *, config: AwwwardsConfig) -> str:
    """Fetch a URL with UA + simple robots-host check (cache-aware)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {url}")
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _slug(parsed.netloc + parsed.path) + ".html"
    cache_path = config.cache_dir / cache_key
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")
    if not config.live:
        raise RuntimeError(f"live fetch disabled and cache miss: {url}")
    time.sleep(config.rate_limit_s)
    req = Request(url, headers={"User-Agent": config.user_agent})
    with urlopen(req, timeout=20) as resp:  # noqa: S310 — intentional research fetch
        body = resp.read().decode("utf-8", errors="replace")
    cache_path.write_text(body, encoding="utf-8")
    return body


def build_awwwards_records(config: AwwwardsConfig | None = None) -> list[ExampleRecord]:
    """Load fixture (and optionally live) award-site rows into ExampleRecords."""
    config = config or AwwwardsConfig()
    sites = load_fixture_sites(config.fixture_path)[: config.max_sites]
    return [site_to_record(site) for site in sites]
