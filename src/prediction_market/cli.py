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
import tempfile
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
# wallets
# ---------------------------------------------------------------------------


def _abbreviate_wallet(wallet: str) -> str:
    """Abbreviate a wallet address as ``0x1234...abcd`` for compact display."""
    if len(wallet) <= 10:
        return wallet
    return f"{wallet[:6]}...{wallet[-4:]}"


@main.command("wallets")
@click.argument("market_id")
@click.option(
    "--hours",
    type=int,
    default=168,
    show_default=True,
    help="Size of the trailing trading window to profile, in hours.",
)
@click.pass_context
def wallets_cmd(ctx: click.Context, market_id: str, hours: int) -> None:
    """Rank a market's wallets by Phase-2 anomaly-profiler score.

    Reads wallet-attributed trades from the local database (no live API
    calls), summarizes them with ``get_market_wallet_summary``, and scores
    each wallet with ``profile_wallets``. Trades ingested before Phase 2
    have no ``proxy_wallet`` and are therefore excluded from this view.
    """
    config = _load(ctx.obj["config_path"])
    _setup_logging(config.log_level)

    from datetime import datetime, timedelta, timezone

    from prediction_market.analysis.wallet_profiler import profile_wallets
    from prediction_market.store.database import get_database
    from prediction_market.store.queries import get_market_wallet_summary

    now = datetime.now(timezone.utc)
    end = now.strftime("%Y-%m-%d %H:%M:%S")
    start = (now - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

    async def _query() -> list[dict[str, Any]]:
        db = await get_database(config)
        try:
            return await get_market_wallet_summary(db, market_id, start, end)
        finally:
            await db.close()

    summaries = _run_async(_query())

    if not summaries:
        click.echo(
            f"No wallet-attributed trades for market {market_id!r} in the last "
            f"{hours}h. Trades ingested before Phase 2 lack wallet attribution."
        )
        return

    features = profile_wallets(summaries, window_end=end, thresholds=config.thresholds)

    if not features:
        click.echo(
            f"No wallets met the minimum-volume threshold for market {market_id!r} "
            f"in the last {hours}h."
        )
        return

    click.echo(
        f"\n{'Wallet':<20} {'Trades':>7} {'Volume USD':>14} {'Share':>7} "
        f"{'Concentration':>13} {'Fresh':>6} {'Score':>7}"
    )
    click.echo("-" * 90)

    for f in features:
        addr = _abbreviate_wallet(f.wallet)
        vol = f"${f.total_volume_usd:,.0f}"
        share = f"{f.volume_share:.1%}"
        conc = f"{f.directional_concentration:.2f}"
        fresh = "yes" if f.is_fresh else "no"
        score = f"{f.score:.3f}"
        click.echo(
            f"{addr:<20} {f.trade_count:>7} {vol:>14} {share:>7} "
            f"{conc:>13} {fresh:>6} {score:>7}"
        )

    click.echo(f"\nTotal: {len(features)} wallet(s) scored")


# ---------------------------------------------------------------------------
# wallet
# ---------------------------------------------------------------------------


@main.command("wallet")
@click.argument("address")
@click.option(
    "--limit",
    type=int,
    default=50,
    show_default=True,
    help="Maximum number of trades to display.",
)
@click.pass_context
def wallet_cmd(ctx: click.Context, address: str, limit: int) -> None:
    """Show one wallet's cross-market trade history from the local DB.

    This is a local-DB view only (``get_wallet_trades``): it never calls
    the live Data API. For an address's *current* on-chain positions or
    activity, use the Data API's ``get_wallet_positions``/
    ``get_wallet_activity`` helpers directly -- those are reserved for the
    live validation session and Phase 3 investigations, not this command.
    """
    config = _load(ctx.obj["config_path"])
    _setup_logging(config.log_level)

    from prediction_market.store.database import get_database
    from prediction_market.store.queries import get_wallet_trades

    async def _query() -> list[dict[str, Any]]:
        db = await get_database(config)
        try:
            return await get_wallet_trades(db, address, limit=limit)
        finally:
            await db.close()

    rows = _run_async(_query())

    if not rows:
        click.echo(f"No trades found for wallet {address!r} in the local database.")
        return

    click.echo(
        f"\n{'Match Time':<20} {'Market':<24} {'Side':<5} {'Outcome':<8} "
        f"{'Price':>7} {'Volume USD':>12}"
    )
    click.echo("-" * 90)

    total_volume = 0.0
    buy_volume = 0.0
    sell_volume = 0.0

    for r in rows:
        ts = str(r["match_time"])[:19]
        market = str(r["market_id"])[:22]
        side = str(r["side"])
        outcome = str(r["outcome"])[:8]
        price = f"{float(r['price']):.4f}" if r["price"] is not None else "N/A"
        vol = float(r["volume_usd"] or 0.0)
        total_volume += vol
        if side.upper() == "BUY":
            buy_volume += vol
        elif side.upper() == "SELL":
            sell_volume += vol
        click.echo(
            f"{ts:<20} {market:<24} {side:<5} {outcome:<8} {price:>7} ${vol:>10,.2f}"
        )

    click.echo(
        f"\nTotal: {len(rows)} trade(s), volume ${total_volume:,.2f} "
        f"(buy ${buy_volume:,.2f} / sell ${sell_volume:,.2f})"
    )


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


# ---------------------------------------------------------------------------
# simulate — run full simulation analysis on a market
# ---------------------------------------------------------------------------


@main.command()
@click.argument("market_id")
@click.option("--hours", type=int, default=24, show_default=True, help="Hours of history to analyze.")
@click.option("--mc", "mc_sims", type=int, default=10000, show_default=True, help="Monte Carlo simulations.")
@click.option("--particles", type=int, default=2000, show_default=True, help="Particle filter particles.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON instead of Markdown report.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.pass_context
def simulate(
    ctx: click.Context,
    market_id: str,
    hours: int,
    mc_sims: int,
    particles: int,
    as_json: bool,
    seed: int,
) -> None:
    """Run full simulation analysis on a market.

    Executes Monte Carlo, Particle Filter, Importance Sampling, and ABM
    analysis using historical price/volume/orderbook data from the database.

    \b
    Example:
        prediction-market simulate <market-id> --hours 48
        prediction-market simulate <market-id> --json
    """
    config = _load(ctx.obj["config_path"])
    _setup_logging("WARNING")

    import numpy as np

    async def _run_analysis() -> dict[str, Any]:
        from prediction_market.store.database import get_database
        from prediction_market.store import queries as q
        from prediction_market.simulation.integration import analyze_market

        db = await get_database(config)
        try:
            # Fetch price/volume data
            snapshots = await q.get_recent_snapshots(db, market_id, hours=hours)
            if not snapshots:
                click.echo(f"No snapshots found for market {market_id}", err=True)
                raise SystemExit(1)

            prices = np.array([s["price_yes"] for s in snapshots if s.get("price_yes")])
            volumes = np.array([s.get("volume_24hr", 0) or 0 for s in snapshots])

            if len(prices) < 5:
                click.echo(f"Only {len(prices)} price observations — need at least 5", err=True)
                raise SystemExit(1)

            # Fetch orderbook data
            orderbook_rows = await q.get_recent_orderbooks(db, market_id, "", hours=hours)

            # Fetch trades
            trades = await q.get_market_trades(db, market_id, hours=hours)

            # Get market question for display
            market_row = await q._fetch_one_dict(
                db, "SELECT question FROM markets WHERE id = ?", (market_id,)
            )
            market_question = market_row["question"] if market_row else market_id

            result = analyze_market(
                prices=prices,
                volumes=volumes,
                orderbook_rows=orderbook_rows if orderbook_rows else None,
                trades=trades if trades else None,
                market_id=market_id,
                mc_simulations=mc_sims,
                n_particles=particles,
                seed=seed,
            )
            result["market_question"] = market_question
            return result
        finally:
            await db.close()

    result = _run_async(_run_analysis())

    if as_json:
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        from prediction_market.simulation.integration import generate_report
        report = generate_report(result)
        # Prepend market question
        q = result.get("market_question", "")
        if q and q != market_id:
            report = f"> **{q}**\n\n{report}"
        click.echo(report)


# ---------------------------------------------------------------------------
# dashboard — simulation overview of all tracked markets
# ---------------------------------------------------------------------------


@main.command()
@click.option("--hours", type=int, default=24, show_default=True, help="Hours of history.")
@click.option("--top", type=int, default=20, show_default=True, help="Number of markets to show.")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON.")
@click.pass_context
def dashboard(ctx: click.Context, hours: int, top: int, as_json: bool) -> None:
    """Simulation dashboard: scored overview of all tracked markets.

    Runs quick Monte Carlo + Particle Filter analysis on each active
    political market and ranks them by combined simulation score.
    """
    config = _load(ctx.obj["config_path"])
    _setup_logging("WARNING")

    import numpy as np

    async def _run_dashboard() -> list[dict[str, Any]]:
        from prediction_market.store.database import get_database
        from prediction_market.store import queries as q
        from prediction_market.simulation.integration import SimulationEnhancedDetector

        db = await get_database(config)
        try:
            # Get all active political markets
            cursor = await db.execute(
                """
                SELECT m.id, m.question, m.volume, m.liquidity
                FROM markets m
                WHERE m.political_confidence > 0 AND m.active = 1
                ORDER BY m.volume DESC
                LIMIT ?
                """,
                (top * 2,),  # Fetch extra in case some have no data
            )
            rows = await cursor.fetchall()
            columns = [d[0] for d in cursor.description]
            markets = [dict(zip(columns, row)) for row in rows]

            if not markets:
                return []

            detector = SimulationEnhancedDetector(
                mc_simulations=3000,  # Lighter for dashboard
                n_particles=500,
                seed=42,
            )

            results = []
            for mkt in markets:
                mid = mkt["id"]
                snapshots = await q.get_recent_snapshots(db, mid, hours=hours)
                if not snapshots or len(snapshots) < 5:
                    continue

                prices = [s["price_yes"] for s in snapshots if s.get("price_yes")]
                volumes = [s.get("volume_24hr", 0) or 0 for s in snapshots]

                if len(prices) < 5:
                    continue

                # Feed through detector
                for i, p in enumerate(prices):
                    v = float(volumes[i]) if i < len(volumes) else 0.0
                    sig = detector.process_tick(mid, float(p), v)

                results.append({
                    "market_id": mid,
                    "question": mkt["question"][:60],
                    "volume": mkt.get("volume", 0),
                    "current_price": float(prices[-1]),
                    "n_observations": len(prices),
                    "combined_score": sig.combined_sim_score,
                    "mc_anomaly": sig.mc_anomaly_score,
                    "pf_surprise": sig.pf_surprise,
                    "pf_drift": sig.pf_drift_score,
                    "regime": sig.pf_regime,
                    "summary": sig.to_summary(),
                })

                if len(results) >= top:
                    break

            # Sort by combined score descending
            results.sort(key=lambda r: r["combined_score"], reverse=True)
            return results
        finally:
            await db.close()

    results = _run_async(_run_dashboard())

    if not results:
        click.echo("No markets with sufficient data. Run 'backfill' first.")
        return

    if as_json:
        click.echo(json.dumps(results, indent=2, default=str))
        return

    # Table output
    click.echo("\n🎯 Prediction Market Simulation Dashboard")
    click.echo(f"   Analyzing {len(results)} markets with {hours}h of data\n")

    click.echo(
        f"{'#':>3} {'Score':>7} {'Risk':>10} {'Price':>7} {'MC σ':>6} "
        f"{'Drift':>7} {'Regime':>7}  Question"
    )
    click.echo("─" * 100)

    for i, r in enumerate(results, 1):
        score = r["combined_score"]
        if score >= 0.7:
            risk = "🔴 CRIT"
        elif score >= 0.5:
            risk = "🟠 HIGH"
        elif score >= 0.3:
            risk = "🟡 ELEV"
        elif score >= 0.15:
            risk = "🔵 LOW"
        else:
            risk = "🟢 NORM"

        click.echo(
            f"{i:>3} {score:>7.4f} {risk:>10} {r['current_price']:>7.3f} "
            f"{r['mc_anomaly']:>6.1f} {r['pf_drift']:>+7.1f} "
            f"{r['regime']:>7}  {r['question']}"
        )

    click.echo(f"\n   Run 'prediction-market simulate <market-id>' for full analysis")


# ---------------------------------------------------------------------------
# archive-case
# ---------------------------------------------------------------------------


@main.command("archive-case")
@click.argument("slug")
@click.option(
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/cases"),
    show_default=True,
    help="Parent directory under which the case directory is written.",
)
@click.option(
    "--max-trade-pages",
    type=int,
    default=200,
    show_default=True,
    help="Safety cap on Data API trade pages fetched (100 trades/page).",
)
@click.pass_context
def archive_case_cmd(
    ctx: click.Context, slug: str, output_dir: Path, max_trade_pages: int
) -> None:
    """Freeze a live Polymarket market into the on-disk backtest case format.

    Fetches the market's Gamma metadata, YES-token CLOB price history, and
    full Data-API trade tape, derives the replay spine, and writes a case
    directory that ``load_case``/``replay_case`` accept.
    """
    config = _load(ctx.obj["config_path"])
    _setup_logging(config.log_level)

    from prediction_market.backtest.archiver import archive_case, summarize_case
    from prediction_market.backtest.case_format import load_case

    click.echo(f"Archiving case for slug {slug!r}...", err=True)
    case_dir = _run_async(
        archive_case(config, slug, output_dir, max_trade_pages=max_trade_pages)
    )

    case = load_case(case_dir)
    summary = summarize_case(case)

    click.echo(f"Wrote case to {case_dir}")
    click.echo(
        f"snapshots={summary['snapshots']} trades={summary['trades']} "
        f"events={summary['events']}"
    )


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


@main.command("cases")
@click.option(
    "--dir",
    "cases_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/cases"),
    show_default=True,
    help="Directory containing archived case subdirectories.",
)
@click.pass_context
def cases_cmd(ctx: click.Context, cases_dir: Path) -> None:
    """List archived backtest cases.

    Reads each subdirectory of ``--dir`` as a case via ``load_case`` and
    prints its slug, question, snapshot/trade counts, and label status.
    """
    config = _load(ctx.obj["config_path"])
    _setup_logging(config.log_level)

    from prediction_market.backtest.case_format import load_case

    cases_dir = Path(cases_dir)
    if not cases_dir.exists():
        click.echo(f"No cases directory at {cases_dir}.")
        return

    rows = []
    for entry in sorted(cases_dir.iterdir()):
        if not entry.is_dir():
            continue
        try:
            case = load_case(entry)
        except ValueError as e:
            click.echo(f"Skipping {entry.name}: {e}", err=True)
            continue
        rows.append(case)

    if not rows:
        click.echo(f"No archived cases found in {cases_dir}.")
        return

    click.echo(f"\n{'Slug':<30} {'Snap':>6} {'Trades':>7} {'Labeled':>8}  Question")
    click.echo("-" * 110)
    for case in rows:
        slug = case.slug[:28]
        labeled = "yes" if case.label is not None else "no"
        q = case.question[:50]
        click.echo(
            f"{slug:<30} {len(case.snapshots):>6} {len(case.trades):>7} "
            f"{labeled:>8}  {q}"
        )

    click.echo(f"\nTotal: {len(rows)} archived case(s)")


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


@main.command("backtest")
@click.option(
    "--case",
    "case_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to an archived case directory (see 'archive-case'/'cases').",
)
@click.option(
    "--null-runs",
    "null_runs",
    type=int,
    default=0,
    show_default=True,
    help="Replay this many synthetic null controls and report the "
    "detector's false-positive path rate.",
)
@click.option(
    "--seed",
    type=int,
    default=1000,
    show_default=True,
    help="Base seed for synthetic null-control generation.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit the evaluation (and null-control results) as JSON.",
)
@click.pass_context
def backtest_cmd(
    ctx: click.Context,
    case_dir: Path,
    null_runs: int,
    seed: int,
    as_json: bool,
) -> None:
    """Replay a case and score the detector's output against its label.

    Replays ``--case`` through the real point-in-time detector, then
    reports whether the labeled anomaly window was detected, the lead
    time relative to the label's ``event_time``, and every emitted
    report. With ``--null-runs N``, also replays N synthetic
    volatility-matched null markets and reports how often the detector
    fired on them anyway -- the false-positive control.
    """
    config = _load(ctx.obj["config_path"])
    _setup_logging(config.log_level)

    from prediction_market.backtest.case_format import load_case
    from prediction_market.backtest.metrics import evaluate_replay
    from prediction_market.backtest.replay import replay_case
    from prediction_market.backtest.synthetic import estimate_false_positive_rate

    case = load_case(case_dir)

    with tempfile.TemporaryDirectory(prefix="prediction-market-backtest-") as tmpdir:
        run_config = config.model_copy(deep=True)
        run_config.database.path = str(Path(tmpdir) / "replay.db")

        result = _run_async(replay_case(case, run_config))
        evaluation = evaluate_replay(result, case)

        fp_result: dict[str, Any] | None = None
        if null_runs > 0:
            fp_result = _run_async(
                estimate_false_positive_rate(
                    case, run_config, runs=null_runs, base_seed=seed
                )
            )

    if as_json:
        payload = evaluation.to_dict()
        if fp_result is not None:
            payload["null_control"] = fp_result
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo(f"Case: {case.slug}")
    click.echo(f"Question: {case.question}")
    if case.label is not None:
        click.echo(
            f"Labeled window: {case.label.window_start} .. {case.label.window_end} "
            f"(event_time={case.label.event_time})"
        )
    else:
        click.echo("Labeled window: none (unlabeled case)")

    click.echo(f"Detected: {evaluation.detected}")
    lead = evaluation.lead_time_minutes
    lead_str = f"{lead:.1f} min" if lead is not None else "n/a"
    click.echo(f"Lead time: {lead_str}")
    click.echo(f"First hit: {evaluation.first_hit_time or 'n/a'}")
    click.echo(
        f"Hits: {evaluation.hits}  False alarms: {evaluation.false_alarms}  "
        f"Total reports: {evaluation.total_reports}"
    )

    if result.reports:
        click.echo("\nReports:")
        for r in result.reports:
            ts = r.details.get("snapshot_timestamp", "?")
            click.echo(f"  {ts}  {r.severity:<8}  {r.summary}")

    if fp_result is not None:
        click.echo("\nNull control:")
        click.echo(
            f"  runs={fp_result['runs']} "
            f"paths_with_reports={fp_result['paths_with_reports']} "
            f"reports_per_path_mean={fp_result['reports_per_path_mean']:.3f} "
            f"fp_path_rate={fp_result['fp_path_rate']:.3f}"
        )
