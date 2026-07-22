"""Point-in-time replay engine.

Drives the real :class:`~prediction_market.agents.info_leak_detector
.InfoLeakDetector` through an archived :class:`~prediction_market.backtest
.case_format.Case` against a throwaway SQLite database. At each step the
detector sees only data timestamped at or before that step -- this module
is the sole place that decides what "point-in-time" means; it must not
reach into detector internals, monkeypatch time, or reimplement scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from prediction_market.agents.info_leak_detector import InfoLeakDetector
from prediction_market.backtest.case_format import Case
from prediction_market.config import AppConfig
from prediction_market.data.external.models import NewsCheckResult
from prediction_market.reporting.anomaly_report import AnomalyReport
from prediction_market.reporting.sink import ReportSink
from prediction_market.store.database import init_database
from prediction_market.store.queries import save_scheduled_events
from prediction_market.store.snapshots import (
    _normalize_match_time,
    save_market,
    save_price_snapshot,
    save_trades_batch,
)

logger = logging.getLogger(__name__)

_POLITICAL_CLASSIFICATION = {"confidence": 1.0, "reasons": ["backtest case"]}


class _ListSink(ReportSink):
    """A :class:`ReportSink` that captures every report in a list.

    Used by the replay engine in place of the live sinks (files, stdout,
    webhooks) -- the harness inspects reports in-process rather than
    dispatching them anywhere.
    """

    def __init__(self) -> None:
        self.reports: list[AnomalyReport] = []

    async def write(self, report: AnomalyReport) -> None:
        self.reports.append(report)


class _NullNewsChecker:
    """A news checker stub that always reports no news found.

    Replay is offline by definition -- there is no live GDELT/NewsAPI
    access during historical replay -- so the news dampener never engages.
    This is a deliberate scope choice, not a placeholder for a future
    implementation: replaying the news dampener is out of scope for this
    engine.
    """

    async def check_news_exists(
        self, keywords: list[str], before_time=None
    ) -> NewsCheckResult:
        return NewsCheckResult(news_found=False)


@dataclass
class ReplayResult:
    """The outcome of replaying one case.

    Attributes:
        reports: Every :class:`AnomalyReport` the detector emitted, in
            emission order.
        steps: Number of snapshot rows replayed.
        case: The case that was replayed.
    """

    reports: list[AnomalyReport] = field(default_factory=list)
    steps: int = 0
    case: Case | None = None


async def replay_case(case: Case, config: AppConfig) -> ReplayResult:
    """Replay *case* through the real :class:`InfoLeakDetector`.

    Strictly point-in-time: at the step for snapshot row *i*, the detector
    can only see trades with ``match_time <= row[i].timestamp`` and the
    snapshot rows up to and including row *i*. Scheduled events are
    inserted up front -- they are public calendar announcements known in
    advance, not market activity, so pre-loading them is not lookahead.

    Args:
        case: The loaded, validated case to replay.
        config: Application config. ``config.database.path`` determines
            where the throwaway SQLite database is created -- callers
            should point this at a temporary path.

    Returns:
        A :class:`ReplayResult` with every emitted report, the step count,
        and the case that was replayed.
    """
    db = await init_database(config)
    try:
        # Force active=1, closed=0 regardless of the archived market's real
        # (likely resolved) state -- get_active_political_markets filters
        # on `active = 1 AND political_confidence > 0`, and a replayed case
        # is by construction a market under surveillance.
        surveilled_market = case.market.model_copy(update={"active": True, "closed": False})
        await save_market(db, surveilled_market, _POLITICAL_CLASSIFICATION)

        if case.events:
            await save_scheduled_events(db, case.events)

        sink = _ListSink()
        detector = InfoLeakDetector(
            config, db, sinks=[sink], news_checker=_NullNewsChecker()
        )

        inserted_trade_ids: set[str] = set()
        for row in case.snapshots:
            await save_price_snapshot(
                db,
                case.market_id,
                price_yes=row.price_yes,
                price_no=row.price_no,
                volume_total=row.volume_total,
                timestamp=row.timestamp,
            )

            due_trades = [
                t
                for t in case.trades
                if t.id not in inserted_trade_ids
                and _normalize_match_time(t.match_time) <= row.timestamp
            ]
            if due_trades:
                await save_trades_batch(db, due_trades, case.market_id)
                inserted_trade_ids.update(t.id for t in due_trades)

            await detector.tick()

        return ReplayResult(reports=sink.reports, steps=len(case.snapshots), case=case)
    finally:
        await db.close()
