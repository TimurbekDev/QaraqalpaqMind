"""Verify every source in `configs/crawl/sources.yaml` against the live web.

For each declared source this script answers, with evidence rather than
assumption:

    * Does the domain resolve and return 2xx?
    * What does robots.txt say about our User-Agent?
    * Is there a sitemap we can use instead of link-crawling?
    * Is the landing page actually in Karakalpak, and in which script?

It writes a timestamped JSON report to `data/manifests/` so the audit is a
reproducible artefact, not a one-off terminal session.

Usage:
    python scripts/audit_sources.py
    python scripts/audit_sources.py --include-disabled --timeout 30
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from urllib.robotparser import RobotFileParser

import httpx
import orjson
import typer
from rich.console import Console
from rich.table import Table

from qaraqalpaqmind.common import MANIFESTS_DIR, ensure_dir, get_logger
from qaraqalpaqmind.crawlers.core.registry import AccessMethod, SourceSpec, load_registry
from qaraqalpaqmind.preprocessing.script import analyse

logger = get_logger(__name__)
console = Console()

_MAX_BODY_BYTES = 2_000_000
_CONCURRENCY = 6


@dataclass(slots=True)
class AuditResult:
    """Everything we learned about one source in a single pass."""

    source_id: str
    url: str
    access: str
    reachable: bool
    status_code: int | None = None
    final_url: str | None = None
    robots_found: bool = False
    robots_allows: bool | None = None
    crawl_delay: float | None = None
    sitemaps: list[str] | None = None
    text_chars: int = 0
    script: str = "unknown"
    orthography: str = "unknown"
    kaa_score: float = 0.0
    error: str | None = None

    @property
    def verdict(self) -> str:
        """One-word summary used for the terminal table and for triage."""
        if not self.reachable:
            return "DEAD"
        if self.robots_allows is False:
            return "BLOCKED"
        if self.access != AccessMethod.CRAWL:
            return "OK"
        if self.text_chars < 200:
            return "NO-TEXT"
        if self.kaa_score >= 0.5:
            return "KAA"
        if self.kaa_score >= 0.2:
            return "WEAK"
        return "NOT-KAA"


def _extract_text(html: str) -> str:
    """Best-effort main-content extraction, falling back to raw tag stripping."""
    try:
        import trafilatura

        extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
        if extracted:
            return extracted
    except Exception as exc:  # extraction must never abort an audit
        logger.debug("trafilatura failed", extra={"error": str(exc)})

    from selectolax.parser import HTMLParser

    tree = HTMLParser(html)
    for tag in tree.css("script, style, noscript"):
        tag.decompose()
    body = tree.body
    return body.text(separator=" ", strip=True) if body else ""


async def _check_robots(
    client: httpx.AsyncClient, spec: SourceSpec, user_agent: str, result: AuditResult
) -> None:
    """Populate the robots.txt fields of `result`. Never raises."""
    try:
        response = await client.get(spec.robots_url)
    except httpx.HTTPError as exc:
        logger.debug("robots.txt unreachable", extra={"source": spec.id, "error": str(exc)})
        return

    if response.status_code != 200 or not response.text.strip():
        return

    result.robots_found = True
    parser = RobotFileParser()
    parser.parse(response.text.splitlines())

    result.robots_allows = parser.can_fetch(user_agent, str(spec.url))
    delay = parser.crawl_delay(user_agent)
    result.crawl_delay = float(delay) if delay is not None else None
    result.sitemaps = [
        line.split(":", 1)[1].strip()
        for line in response.text.splitlines()
        if line.lower().startswith("sitemap:")
    ] or None


async def _audit_one(
    client: httpx.AsyncClient, spec: SourceSpec, user_agent: str, semaphore: asyncio.Semaphore
) -> AuditResult:
    result = AuditResult(
        source_id=spec.id, url=str(spec.url), access=spec.access.value, reachable=False
    )

    async with semaphore:
        if spec.access is AccessMethod.CRAWL:
            await _check_robots(client, spec, user_agent, result)

        try:
            response = await client.get(str(spec.url))
        except httpx.HTTPError as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            return result

        result.status_code = response.status_code
        result.final_url = str(response.url)
        result.reachable = response.status_code < 400
        if not result.reachable:
            return result

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return result

        html = response.text[:_MAX_BODY_BYTES]
        text = _extract_text(html)
        report = analyse(text)

        result.text_chars = len(text)
        result.script = report.script.value
        result.orthography = report.orthography.value
        result.kaa_score = report.karakalpak_score

    return result


async def _run(include_disabled: bool, timeout: float, insecure: bool) -> list[AuditResult]:
    registry = load_registry()
    specs = registry.sources if include_disabled else registry.enabled_sources()

    headers = {"User-Agent": registry.user_agent, "Accept-Language": "kaa,uz;q=0.8,*;q=0.5"}
    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async with httpx.AsyncClient(
        headers=headers,
        timeout=timeout,
        follow_redirects=True,
        # TLS verification stays ON by default. Some .uz government hosts ship
        # expired or misissued certificates; `--insecure` exists to identify
        # them during an audit of public pages only. Never use it to fetch
        # content we will actually keep.
        verify=not insecure,
    ) as client:
        tasks = [_audit_one(client, spec, registry.user_agent, semaphore) for spec in specs]
        return await asyncio.gather(*tasks)


_VERDICT_STYLE = {
    "KAA": "bold green",
    "OK": "green",
    "WEAK": "yellow",
    "NOT-KAA": "red",
    "NO-TEXT": "yellow",
    "BLOCKED": "bold red",
    "DEAD": "bright_black",
}


def _render(results: list[AuditResult]) -> None:
    table = Table(title="Karakalpak source audit", show_lines=False)
    for column in ("source", "verdict", "http", "robots", "script", "ortho", "kaa", "chars"):
        table.add_column(column, overflow="fold")

    for result in sorted(results, key=lambda r: (r.verdict, r.source_id)):
        robots = (
            "n/a"
            if not result.robots_found
            else ("allow" if result.robots_allows else "DENY")
        )
        table.add_row(
            result.source_id,
            f"[{_VERDICT_STYLE.get(result.verdict, 'white')}]{result.verdict}[/]",
            str(result.status_code or "-"),
            robots,
            result.script,
            result.orthography,
            f"{result.kaa_score:.2f}",
            str(result.text_chars),
        )
    console.print(table)

    for result in results:
        if result.error:
            console.print(f"[bright_black]{result.source_id}: {result.error}[/]")


def main(
    include_disabled: bool = typer.Option(False, "--include-disabled"),
    timeout: float = typer.Option(20.0, "--timeout"),
    insecure: bool = typer.Option(
        False, "--insecure", help="Disable TLS verification (diagnosing broken .uz certs only)."
    ),
) -> None:
    """Audit every registered source and write a JSON report."""
    results = asyncio.run(_run(include_disabled, timeout, insecure))
    _render(results)

    stamp = datetime.now(tz=UTC).strftime("%Y%m%d")
    report_path = ensure_dir(MANIFESTS_DIR) / f"source_audit_{stamp}.json"
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "results": [asdict(r) | {"verdict": r.verdict} for r in results],
    }
    report_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    console.print(f"\nReport written to [cyan]{report_path}[/]")


if __name__ == "__main__":
    typer.run(main)
