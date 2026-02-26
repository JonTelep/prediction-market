"""Output sinks for anomaly reports.

Each sink implements a single ``write`` method that accepts an
:class:`AnomalyReport` and dispatches it to some destination — a file,
stdout, a webhook endpoint, or a combination of all three.
"""

from __future__ import annotations

import abc
import json
import logging
from pathlib import Path
from typing import Any

import httpx

from prediction_market.reporting.anomaly_report import AnomalyReport
from prediction_market.reporting.human_formatter import format_report as format_md
from prediction_market.reporting.json_formatter import format_report as format_json

logger = logging.getLogger(__name__)


class ReportSink(abc.ABC):
    """Abstract base for all report output destinations."""

    @abc.abstractmethod
    async def write(self, report: AnomalyReport) -> None:
        """Persist or dispatch *report* to this sink's destination."""

    async def close(self) -> None:
        """Release resources held by this sink.  Override when needed."""


class FileSink(ReportSink):
    """Write each report as a JSON file *and* a Markdown file.

    Directory layout::

        <output_dir>/
            <report-id>.json
            <report-id>.md
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def write(self, report: AnomalyReport) -> None:
        json_path = self.output_dir / f"{report.id}.json"
        md_path = self.output_dir / f"{report.id}.md"

        json_path.write_text(format_json(report), encoding="utf-8")
        md_path.write_text(format_md(report), encoding="utf-8")

        logger.info("Report %s written to %s", report.id, self.output_dir)


class StdoutSink(ReportSink):
    """Print a report to standard output (Markdown format)."""

    def __init__(self, use_json: bool = False) -> None:
        self._use_json = use_json

    async def write(self, report: AnomalyReport) -> None:
        if self._use_json:
            print(format_json(report))
        else:
            print(format_md(report))


class WebhookSink(ReportSink):
    """POST the report JSON to a remote webhook URL.

    Uses ``httpx.AsyncClient`` with a configurable timeout and
    automatic retry on transient failures.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = url
        self._timeout = timeout
        self._max_retries = max_retries
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._client = http_client or httpx.AsyncClient(timeout=self._timeout)
        self._owns_client = http_client is None

    async def write(self, report: AnomalyReport) -> None:
        payload = report.to_dict()
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = await self._client.post(
                    self.url,
                    content=json.dumps(payload, default=str),
                    headers=self._headers,
                )
                resp.raise_for_status()
                logger.info(
                    "Report %s posted to webhook (attempt %d, status %d)",
                    report.id,
                    attempt,
                    resp.status_code,
                )
                return
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                logger.warning(
                    "Webhook attempt %d/%d failed for report %s: %s",
                    attempt,
                    self._max_retries,
                    report.id,
                    exc,
                )

        logger.error(
            "All %d webhook attempts failed for report %s: %s",
            self._max_retries,
            report.id,
            last_exc,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class CompositeSink(ReportSink):
    """Fan-out: write a report to every child sink.

    Errors in one child do not prevent the others from receiving the
    report — failures are logged and swallowed.
    """

    def __init__(self, sinks: list[ReportSink]) -> None:
        self.sinks = list(sinks)

    async def write(self, report: AnomalyReport) -> None:
        for sink in self.sinks:
            try:
                await sink.write(report)
            except Exception:
                logger.exception(
                    "Sink %s failed for report %s", type(sink).__name__, report.id
                )

    async def close(self) -> None:
        for sink in self.sinks:
            try:
                await sink.close()
            except Exception:
                logger.exception("Error closing sink %s", type(sink).__name__)
