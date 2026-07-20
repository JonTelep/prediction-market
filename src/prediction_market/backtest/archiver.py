"""Case archiver: freeze a live Polymarket market into the on-disk case format.

Fetches a market's Gamma metadata, its full CLOB YES-token price history,
and its complete Data-API trade tape, derives the replay spine
(``snapshots.json``), and writes everything through
:func:`~prediction_market.backtest.case_format.save_case`. This mirrors
what the live snapshot loop (see ``store/snapshots.py`` and
``orchestrator.py``) would have recorded over time -- reconstructed here
after the fact from the only data the APIs still serve, rather than
observed in real time. There is no order-book history to reconstruct
(order books are not archived by Polymarket's REST APIs; live capture is
Phase 3 work) and no NO-token price history is fetched -- the replay
engine only ever consumes ``price_no = 1 - price_yes``.

``interval="max"`` on the CLOB ``/prices-history`` call is an assumption
about the live API, not a verified fact: the client's docstring documents
only ``1m``/``1h``/``1d`` examples, and this repository's test suite is
fully mocked, so nothing here has exercised the real endpoint. The live
validation session at the end of this phase is tasked with confirming
``"max"`` actually returns full history and reporting back if it does not.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from prediction_market.backtest.case_format import Case, SnapshotRow, save_case
from prediction_market.config import AppConfig
from prediction_market.data.polymarket.clob_client import ClobClient
from prediction_market.data.polymarket.data_client import DataClient
from prediction_market.data.polymarket.gamma_client import GammaClient
from prediction_market.data.polymarket.models import GammaMarket, Trade

logger = logging.getLogger(__name__)

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
_TRADE_PAGE_LIMIT = 100


async def _resolve_market(gamma: GammaClient, slug: str) -> GammaMarket:
    """Resolve *slug* to a single Gamma market via an exact slug match.

    Queries without the ``closed`` param first, then retries with
    ``closed=True``: the live Gamma ``/markets?slug=`` filter excludes
    closed markets by default (verified 2026-07-20), and most archival
    targets are resolved markets.
    """
    results = await gamma.search_markets(slug)
    if not results:
        results = await gamma.search_markets(slug, closed=True)
    if not results:
        raise ValueError(
            f"No Gamma markets found for slug {slug!r}; check the slug is correct."
        )
    exact = [m for m in results if m.slug == slug]
    if not exact:
        candidates = ", ".join(repr(m.slug) for m in results)
        raise ValueError(
            f"No exact slug match for {slug!r} among Gamma results; "
            f"candidates found: {candidates}"
        )
    return exact[0]


async def _fetch_price_history(
    clob: ClobClient, market: GammaMarket
) -> tuple[list[tuple[int, float]], str | None]:
    """Fetch the YES-token price history for *market*.

    Returns a list of ``(t, p)`` points and, if applicable, a note string
    describing an empty-response or error outcome (``None`` on a normal
    non-empty response).
    """
    yes_token = market.yes_token_id
    if yes_token is None:
        return [], "price_history: empty (market has no CLOB tokens)"

    try:
        result = await clob.get_price_history(yes_token, interval="max", fidelity=10)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        logger.warning(
            "archive_case: price history fetch failed for token %s: HTTP %d",
            yes_token,
            status,
        )
        return [], f"price_history: error: {status}"

    if result.history:
        return [(p.t, p.p) for p in result.history], None

    # Live-API finding (2026-07-20): interval-form queries return an empty
    # history for resolved markets, but an explicit startTs/endTs range
    # still serves the data. Retry with a range spanning the market's
    # lifetime (createdAt .. endDate, padded a day each side) at hourly
    # fidelity -- the cadence the live snapshot loop approximates.
    start_ts = _parse_iso_epoch(market.created_at)
    end_ts = _parse_iso_epoch(market.end_date)
    if start_ts is None or end_ts is None:
        return [], "price_history: empty (and no market dates for range retry)"

    try:
        result = await clob.get_price_history(
            yes_token,
            start_ts=start_ts - 86400,
            end_ts=end_ts + 86400,
            fidelity=60,
        )
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        logger.warning(
            "archive_case: ranged price history fetch failed for token %s: HTTP %d",
            yes_token,
            status,
        )
        return [], f"price_history: error: {status} (startTs/endTs retry)"

    if not result.history:
        return [], "price_history: empty (interval and startTs/endTs both)"

    return (
        [(p.t, p.p) for p in result.history],
        f"price_history: {len(result.history)} points via startTs/endTs range "
        "retry (interval query returned empty for this resolved market)",
    )


def _parse_iso_epoch(raw: str) -> int | None:
    """Parse an ISO-8601 string (Gamma createdAt/endDate) to epoch seconds."""
    if not raw:
        return None
    try:
        return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


async def _fetch_trades(
    data: DataClient, market: GammaMarket, max_trade_pages: int
) -> tuple[list[Trade], str | None]:
    """Fetch the full trade tape for *market*.

    Returns the trades and, if the page cap was hit, a note describing the
    truncation (``None`` otherwise).
    """
    trades = await data.get_all_trades(
        condition_id=market.condition_id,
        max_pages=max_trade_pages,
        limit=_TRADE_PAGE_LIMIT,
    )

    # get_all_trades only stops early when a page comes back short of the
    # limit. If it ran through every one of max_trade_pages iterations
    # without a short page, the total is exactly max_trade_pages * limit
    # and the final page was full -- the tape may continue beyond what we
    # fetched.
    truncated = len(trades) == max_trade_pages * _TRADE_PAGE_LIMIT
    note = None
    if truncated:
        note = (
            f"trades: page cap ({max_trade_pages} pages x {_TRADE_PAGE_LIMIT}) "
            f"reached with a full final page -- tape may be truncated; increase "
            f"--max-trade-pages to fetch more"
        )
        logger.warning(
            "archive_case: trade tape may be truncated at max_trade_pages=%d "
            "(%d trades fetched); the final page came back full",
            max_trade_pages,
            len(trades),
        )
    return trades, note


def _derive_snapshots(
    price_points: list[tuple[int, float]], trades: list[Trade]
) -> list[SnapshotRow]:
    """Derive the replay spine from price points and the trade tape.

    One snapshot row per price point: ``price_yes = p``, ``price_no = 1 -
    p``, and ``volume_total`` the cumulative sum of ``volume_usd`` across
    all trades matched at or before that point's timestamp.
    """
    dated_trades = sorted(
        (
            (dt, t.volume_usd)
            for t in trades
            if (dt := t.match_datetime) is not None
        ),
        key=lambda pair: pair[0],
    )

    sorted_points = sorted(price_points, key=lambda pt: pt[0])

    snapshots: list[SnapshotRow] = []
    trade_idx = 0
    running_volume = 0.0
    for t, p in sorted_points:
        point_dt = datetime.fromtimestamp(t, tz=UTC)
        while trade_idx < len(dated_trades) and dated_trades[trade_idx][0] <= point_dt:
            running_volume += dated_trades[trade_idx][1]
            trade_idx += 1
        snapshots.append(
            SnapshotRow(
                timestamp=point_dt.strftime(_TIMESTAMP_FORMAT),
                price_yes=p,
                price_no=1.0 - p,
                volume_total=running_volume,
            )
        )
    return snapshots


async def archive_case(
    config: AppConfig,
    slug: str,
    output_dir: Path,
    *,
    max_trade_pages: int = 200,
) -> Path:
    """Archive a live Polymarket market into the on-disk case format.

    Fetches the market's Gamma metadata, YES-token CLOB price history, and
    full Data-API trade tape; derives ``snapshots.json`` from the two; and
    writes a case directory (named after *slug*, under *output_dir*) via
    :func:`~prediction_market.backtest.case_format.save_case`. The written
    case has no ``[label]`` section -- labeling is a human judgment applied
    later, not something this function infers.

    Args:
        config: Application configuration (API base URLs, rate limits).
        slug: The market's Gamma slug (exact match required).
        output_dir: Parent directory under which ``<slug>/`` is written.
        max_trade_pages: Safety cap on Data API trade pages (100 trades
            per page); defaults to 200 pages (20,000 trades). If the tape
            is truncated at this cap a WARNING is logged and the manifest
            notes record it -- the cap is never applied silently.

    Returns:
        The path to the written case directory.

    Raises:
        ValueError: If no Gamma market matches *slug* exactly.
    """
    http_client = httpx.AsyncClient(
        timeout=30.0,
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )
    gamma = GammaClient(config, http_client=http_client)
    clob = ClobClient(config, http_client=http_client)
    data = DataClient(config, http_client=http_client)

    try:
        market = await _resolve_market(gamma, slug)
        price_points, price_note = await _fetch_price_history(clob, market)
        trades, trade_note = await _fetch_trades(data, market, max_trade_pages)
    finally:
        await gamma.close()
        await clob.close()
        await data.close()
        await http_client.aclose()

    snapshots = _derive_snapshots(price_points, trades)

    notes_parts: list[str] = []
    if price_note:
        notes_parts.append(price_note)
    else:
        notes_parts.append(f"price_history: {len(price_points)} points")
    if trade_note:
        notes_parts.append(trade_note)
    else:
        notes_parts.append(f"trades: {len(trades)} fetched, no page cap hit")
    notes = "; ".join(notes_parts)

    archived_at = datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)

    case = Case(
        slug=slug,
        market_id=market.id,
        condition_id=market.condition_id,
        question=market.question,
        archived_at=archived_at,
        notes=notes,
        market=market,
        snapshots=snapshots,
        trades=trades,
        events=[],
        label=None,
    )

    case_dir = Path(output_dir) / slug
    save_case(case_dir, case)
    return case_dir


def summarize_case(case: Case) -> dict[str, Any]:
    """Small counts summary of *case*, used by both the CLI and callers."""
    return {
        "slug": case.slug,
        "question": case.question,
        "snapshots": len(case.snapshots),
        "trades": len(case.trades),
        "events": len(case.events),
        "labeled": case.label is not None,
    }
