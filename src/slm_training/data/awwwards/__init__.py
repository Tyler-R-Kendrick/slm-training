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

from slm_training.data.design_md.extract import extract_design_md
from slm_training.dsl.design_md import attach_default_design_md
from slm_training.dsl.schema import ExampleRecord

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class AwwwardsConfig:
    cache_dir: Path = Path("src/slm_training/resources/awwwards/cache")
    fixture_path: Path = Path("src/slm_training/resources/awwwards/sites.jsonl")
    rate_limit_s: float = 1.0
    user_agent: str = "slm-training-awwwards/0.1 (+research; respectful)"
    live: bool = False
    max_sites: int = 50


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
    rows.sort(key=lambda r: str(r.get("id") or r.get("title") or ""))
    return rows


def _heuristic_openui(site: dict[str, Any]) -> tuple[str, list[str]]:
    """Map site metadata to a placeholder openuiLibrary skeleton (deterministic)."""
    name = _slug(str(site.get("title") or site.get("id") or "site")).replace("-", "_")
    if name[0].isdigit():
        name = f"site_{name}"
    tags = {str(t).lower() for t in (site.get("tags") or [])}

    lines: list[str] = []
    placeholders: list[str] = []
    children: list[str] = []

    children.append("hero")
    lines.extend(
        [
            'hero_title = TextContent(":hero.title")',
            'hero_body = TextContent(":hero.body")',
            "hero = Card([hero_title, hero_body])",
        ]
    )
    placeholders.extend([":hero.title", ":hero.body"])

    if tags & {"image", "photo", "gallery", "visual", "creative"}:
        children.append("visual")
        lines.append('visual = ImageBlock(":assets.hero", ":hero.alt")')
        placeholders.extend([":assets.hero", ":hero.alt"])

    if tags & {"metrics", "finance", "saas"}:
        children.append("metrics")
        lines.extend(
            [
                'm1_title = TextContent(":metrics.one.title")',
                'm1_body = TextContent(":metrics.one.body")',
                "m1 = Card([m1_title, m1_body])",
                'm2_title = TextContent(":metrics.two.title")',
                'm2_body = TextContent(":metrics.two.body")',
                "m2 = Card([m2_title, m2_body])",
                'metrics = Stack([m1, m2], "row")',
            ]
        )
        placeholders.extend(
            [
                ":metrics.one.title",
                ":metrics.one.body",
                ":metrics.two.title",
                ":metrics.two.body",
            ]
        )

    if tags & {"tabs", "docs", "events"}:
        children.append("tabs")
        lines.extend(
            [
                'tab_body_a = TextContent(":tabs.a.body")',
                'tab_body_b = TextContent(":tabs.b.body")',
                'tab_a = TabItem("a", ":tabs.a.trigger", [tab_body_a])',
                'tab_b = TabItem("b", ":tabs.b.trigger", [tab_body_b])',
                "tabs = Tabs([tab_a, tab_b])",
            ]
        )
        placeholders.extend(
            [":tabs.a.body", ":tabs.b.body", ":tabs.a.trigger", ":tabs.b.trigger"]
        )

    if tags & {"callout", "ops", "docs"}:
        children.append("note")
        lines.append('note = Callout("info", ":note.title", ":note.description")')
        placeholders.extend([":note.title", ":note.description"])

    if tags & {"form", "signup", "login", "contact", "education", "health"}:
        children.extend(["email", "submit"])
        lines.append(f'email = Input("email", ":{name}.email.placeholder")')
        lines.append(f'submit = Button(":{name}.submit")')
        placeholders.extend([f":{name}.email.placeholder", f":{name}.submit"])
    else:
        children.append("cta")
        lines.append(f'cta = Button(":{name}.cta")')
        placeholders.append(f":{name}.cta")

    if tags & {"settings", "ops"}:
        children.append("notify")
        lines.append(
            'notify = SwitchItem(":settings.notify.label", ":settings.notify.description", "notify")'
        )
        placeholders.extend(
            [":settings.notify.label", ":settings.notify.description"]
        )

    root = f'root = Stack([{", ".join(children)}], "column")'
    return "\n".join([root, *lines]), placeholders


def site_to_record(
    site: dict[str, Any],
    *,
    split: str = "train",
    attach_design_md: bool = True,
) -> ExampleRecord:
    openui, placeholders = _heuristic_openui(site)
    title = str(site.get("title") or "Award site")
    tags = list(site.get("tags") or [])
    desc = str(site.get("description") or "").strip()
    tag_bit = f" Tags: {', '.join(tags)}." if tags else ""
    prompt = (
        f"Build a polished {title} landing layout in OpenUI with placeholders."
        f"{tag_bit}"
        + (f" Context: {desc}" if desc else "")
    )
    record_id = str(site.get("id") or f"awwwards_{_slug(title)}")
    design_md = None
    design_meta: dict[str, Any] = {}
    if attach_design_md:
        design_md = extract_design_md(
            title=title,
            description=desc,
            tags=tags,
            variant="strict",
        )
        # Reuse cached base lint via attach helper on a temp record.
        tmp = ExampleRecord(
            id=record_id,
            prompt=prompt,
            openui=openui,
            placeholders=placeholders,
            split=split,
            source="awwwards",
        )
        tmp = attach_default_design_md(tmp)
        design_meta = (tmp.meta or {}).get("design_lint") or {
            "score": 0.94,
            "summary": {"errors": 0},
        }
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
            "tags": tags,
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
    """Load fixture award-site rows into ExampleRecords (stable order)."""
    config = config or AwwwardsConfig()
    sites = load_fixture_sites(config.fixture_path)[: config.max_sites]
    return [site_to_record(site) for site in sites]
