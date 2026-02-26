"""Click-based CLI for the Polymarket political market surveillance system.

Entry point registered as ``prediction-market`` in pyproject.toml::

    prediction-market monitor                      # continuous monitoring
    prediction-market monitor --agent info-leak    # only Agent 1
    prediction-market monitor --agent manipulation # only Agent 2
    prediction-market scan                         # one-shot scan
    prediction-market backfill --days 30           # historical data
    prediction-market markets                      # list tracked markets
    prediction-market reports --severity high      # browse anomaly reports
    prediction-market report <id>                  # view a single report
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

from prediction_market.config import AppConfig, load_config

logger = logging.getLogger("prediction_market")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(level: str) -> None:
    """Configure root and package loggers."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    # Quieten noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def _load(config_path: str | None) -> AppConfig:
    """Load configuration, honouring the optional ``--config`` flag."""
    path = Path(config_path) if config_path else None
    return load_config(path)


def _run_async(coro: Any) -> Any:
    """Execute an async coroutine in the default event loop.

    Handles ``KeyboardInterrupt`` so that the CLI exits cleanly on
    Ctrl-C without a traceback.
    """
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        click.echo("\nInterrupted.", err=True)
        raise SystemExit(130)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=False),
    default=None,
    help="Path to a custom TOML configuration file.",
)
@click.pass_context
def main(ctx: click.Context, config_path: str | None) -> None:
    """Polymarket political market surveillance system."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--agent",
    "agent_filter",
    type=click.Choice(["info-leak", "manipulation"], case_sensitive=False),
    default=None,
    help="Run only the specified agent instead of both.",
)
@click.pass_context
def monitor(ctx: click.Context, agent_filter: str | None) -> None:
    """Start continuous surveillance of political prediction markets.

    Runs both the information-leak detector and the manipulation guard
    by default.  Use ``--agent`` to restrict to a single agent.
    """
    config = _load(ctx.obj["config_path"])
    _setup_logging(config.log_level)

    from prediction_market.orchestrator import Orchestrator

    orchestrator = Orchestrator(config, agent_filter=agent_filter)

    agent_label = agent_filter or "all agents"
    click.echo(f"Starting continuous monitoring ({agent_label})...", err=True)

    _run_async(orchestrator.start())


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


@main.command()
@click.pass_context
def scan(ctx: click.Context) -> None:
    """One-shot scan of all active political markets.

    Discovers markets, classifies them as political, and prints a
    summary table to stdout.
    """
    config = _load(ctx.obj["config_path"])
    _setup_logging(config.log_level)

    from prediction_market.orchestrator import Orchestrator

    orchestrator = Orchestrator(config)

    click.echo("Scanning political markets...", err=True)
    results = _run_async(orchestrator.scan_once())

    if not results:
        click.echo("No political markets found.")
        return

    # Sort by volume descending
    results.sort(key=lambda r: r.get("volume", 0), reverse=True)

    # Header
    click.echo(
        f"\n{'ID':<40} {'Volume':>12} {'Liq':>10} {'Conf':>6}  Question"
    )
    click.echo("-" * 120)

    for r in results:
        market_id = r["id"][:38]
        vol = f"${r['volume']:,.0f}"
        liq = f"${r['liquidity']:,.0f}"
        conf = f"{r['political_confidence']:.0%}"
        q = r["question"][:50]
        click.echo(f"{market_id:<40} {vol:>12} {liq:>10} {conf:>6}  {q}")

    click.echo(f"\nTotal: {len(results)} political markets")


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--days",
    type=int,
    default=30,
    show_default=True,
    help="Number of days of historical data to fetch.",
)
@click.pass_context
def backfill(ctx: click.Context, days: int) -> None:
    """Populate the database with historical price and trade data.

    Fetches price history and recent trades for all tracked political
    markets and stores them in the local SQLite database.
    """
    config = _load(ctx.obj["config_path"])
    _setup_logging(config.log_level)

    from prediction_market.orchestrator import Orchestrator

    orchestrator = Orchestrator(config)

    click.echo(f"Backfilling {days} days of historical data...", err=True)
    total = _run_async(orchestrator.backfill(days=days))

    click.echo(f"Backfill complete: {total} price data points ingested.")


# ---------------------------------------------------------------------------
# markets
# ---------------------------------------------------------------------------


@main.command()
@click.pass_context
def markets(ctx: click.Context) -> None:
    """List all tracked political markets from the database.

    Reads previously discovered markets from the local database rather
    than making live API calls.
    """
    config = _load(ctx.obj["config_path"])
    _setup_logging(config.log_level)

    async def _list_markets() -> list[dict[str, Any]]:
        from prediction_market.store.database import get_database

        db = await get_database(config)
        try:
            cursor = await db.execute(
                """
                SELECT id, question, volume, liquidity, political_confidence,
                       active, category, last_updated
                FROM markets
                WHERE political_confidence > 0
                ORDER BY volume DESC
                """
            )
            rows = await cursor.fetchall()
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        finally:
            await db.close()

    rows = _run_async(_list_markets())

    if not rows:
        click.echo("No political markets in database. Run 'scan' or 'backfill' first.")
        return

    click.echo(
        f"\n{'ID':<40} {'Volume':>12} {'Liq':>10} {'Conf':>6} {'Active':>7}  Question"
    )
    click.echo("-" * 130)

    for r in rows:
        market_id = str(r["id"])[:38]
        vol = f"${r['volume']:,.0f}" if r["volume"] else "$0"
        liq = f"${r['liquidity']:,.0f}" if r["liquidity"] else "$0"
        conf = f"{r['political_confidence']:.0%}" if r["political_confidence"] else "0%"
        active = "yes" if r["active"] else "no"
        q = str(r["question"])[:48]
        click.echo(f"{market_id:<40} {vol:>12} {liq:>10} {conf:>6} {active:>7}  {q}")

    click.echo(f"\nTotal: {len(rows)} political markets in database")


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--severity",
    type=click.Choice(["low", "medium", "high", "critical"], case_sensitive=False),
    default=None,
    help="Filter reports by minimum severity level.",
)
@click.option(
    "--agent",
    "agent_name",
    type=click.Choice(["info_leak", "manipulation"], case_sensitive=False),
    default=None,
    help="Filter reports by agent name.",
)
@click.option(
    "--limit",
    type=int,
    default=50,
    show_default=True,
    help="Maximum number of reports to display.",
)
@click.pass_context
def reports(
    ctx: click.Context,
    severity: str | None,
    agent_name: str | None,
    limit: int,
) -> None:
    """Browse anomaly reports stored in the database.

    Results are ordered by creation time (newest first).  Use
    ``--severity`` to filter to a minimum severity level and ``--agent``
    to restrict to a specific detector.
    """
    config = _load(ctx.obj["config_path"])
    _setup_logging(config.log_level)

    from prediction_market.store.models import SEVERITY_ORDER

    severity_levels = ["low", "medium", "high", "critical"]

    async def _query_reports() -> list[dict[str, Any]]:
        from prediction_market.store.database import get_database

        db = await get_database(config)
        try:
            query = "SELECT * FROM anomaly_reports WHERE 1=1"
            params: list[Any] = []

            if severity:
                min_idx = SEVERITY_ORDER.get(severity.lower(), 0)
                allowed = [s for s in severity_levels if SEVERITY_ORDER.get(s, 0) >= min_idx]
                placeholders = ",".join("?" for _ in allowed)
                query += f" AND severity IN ({placeholders})"
                params.extend(allowed)

            if agent_name:
                query += " AND agent = ?"
                params.append(agent_name.lower())

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        finally:
            await db.close()

    rows = _run_async(_query_reports())

    if not rows:
        click.echo("No anomaly reports found matching the given criteria.")
        return

    click.echo(
        f"\n{'ID':>6} {'Severity':<10} {'Agent':<15} {'Score':>7} {'Conf':>6}  "
        f"{'Created':<20} Summary"
    )
    click.echo("-" * 120)

    for r in rows:
        report_id = str(r["id"])
        sev = r["severity"].upper()
        agent = r["agent"]
        score = f"{r['anomaly_score']:.3f}" if r["anomaly_score"] else "N/A"
        conf = f"{r['confidence']:.0%}" if r["confidence"] else "N/A"
        created = str(r["created_at"])[:19]
        summary = str(r["summary"])[:50]
        click.echo(
            f"{report_id:>6} {sev:<10} {agent:<15} {score:>7} {conf:>6}  "
            f"{created:<20} {summary}"
        )

    click.echo(f"\nShowing {len(rows)} report(s)")


# ---------------------------------------------------------------------------
# report (single)
# ---------------------------------------------------------------------------


@main.command("report")
@click.argument("report_id", type=int)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Output the report as raw JSON instead of Markdown.",
)
@click.pass_context
def report_detail(ctx: click.Context, report_id: int, as_json: bool) -> None:
    """View a specific anomaly report by its numeric ID.

    By default the report is rendered as human-readable Markdown.
    Pass ``--json`` for the raw JSON payload.
    """
    config = _load(ctx.obj["config_path"])
    _setup_logging(config.log_level)

    async def _fetch_report() -> dict[str, Any] | None:
        from prediction_market.store.database import get_database

        db = await get_database(config)
        try:
            cursor = await db.execute(
                "SELECT * FROM anomaly_reports WHERE id = ?", (report_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            columns = [d[0] for d in cursor.description]
            return dict(zip(columns, row))
        finally:
            await db.close()

    row = _run_async(_fetch_report())

    if row is None:
        click.echo(f"Report #{report_id} not found.", err=True)
        raise SystemExit(1)

    if as_json:
        # Parse JSON-encoded columns for a cleaner output
        for col in ("details", "price_evidence", "volume_evidence", "news_check"):
            if isinstance(row.get(col), str):
                try:
                    row[col] = json.loads(row[col])
                except (json.JSONDecodeError, TypeError):
                    pass
        if isinstance(row.get("calendar_matches"), str):
            try:
                row["calendar_matches"] = json.loads(row["calendar_matches"])
            except (json.JSONDecodeError, TypeError):
                pass
        click.echo(json.dumps(row, indent=2, default=str))
    else:
        # Reconstruct an AnomalyReport and format with the human formatter
        from prediction_market.reporting.anomaly_report import AnomalyReport
        from prediction_market.reporting.human_formatter import format_report

        # Parse JSON string columns
        def _parse_json_col(val: Any, default: Any) -> Any:
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    return default
            if val is None:
                return default
            return val

        # Fetch the market question for display
        market_question = ""

        async def _get_question() -> str:
            from prediction_market.store.database import get_database

            db = await get_database(config)
            try:
                cursor = await db.execute(
                    "SELECT question FROM markets WHERE id = ?",
                    (row["market_id"],),
                )
                qrow = await cursor.fetchone()
                return qrow[0] if qrow else row["market_id"]
            finally:
                await db.close()

        market_question = _run_async(_get_question())

        report = AnomalyReport(
            id=str(row["id"]),
            agent=row["agent"],
            market_id=row["market_id"],
            market_question=market_question,
            severity=row["severity"],
            anomaly_score=float(row["anomaly_score"] or 0),
            confidence=float(row["confidence"] or 0),
            summary=row["summary"],
            details=_parse_json_col(row.get("details"), {}),
            price_evidence=_parse_json_col(row.get("price_evidence"), {}),
            volume_evidence=_parse_json_col(row.get("volume_evidence"), {}),
            calendar_matches=_parse_json_col(row.get("calendar_matches"), []),
            news_check=_parse_json_col(row.get("news_check"), {}),
        )

        click.echo(format_report(report))
