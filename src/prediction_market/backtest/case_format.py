"""On-disk format for archived backtest cases.

A case is a directory containing:

* ``case.toml`` -- manifest (``[case]`` required, ``[label]`` optional).
* ``market.json`` -- one Gamma-style market dict, parseable by
  :class:`~prediction_market.data.polymarket.models.GammaMarket`.
* ``snapshots.json`` -- the replay spine: a time-ascending list of
  ``{"timestamp", "price_yes", "price_no", "volume_total"}`` rows.
* ``trades.json`` -- Data-API-shaped trade dicts, parseable by
  :class:`~prediction_market.data.polymarket.models.Trade`.
* ``events.json`` (optional) -- ``ScheduledEvent``-shaped dicts.

:func:`load_case` validates on load; :func:`save_case` is its round-trip
partner (reused by the case-archiver tooling in a later prompt).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import tomli_w

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py<3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from prediction_market.data.external.models import ScheduledEvent
from prediction_market.data.polymarket.models import GammaMarket, Trade

_EVENT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class SnapshotRow:
    """One row of the replay spine.

    Attributes:
        timestamp: "%Y-%m-%d %H:%M:%S" UTC TEXT timestamp.
        price_yes: YES outcome price (0-1).
        price_no: NO outcome price (0-1).
        volume_total: All-time cumulative trading volume (USD) as of this
            snapshot -- the replay engine derives the per-interval delta
            fed to the volume analyzer from consecutive rows.
    """

    timestamp: str
    price_yes: float
    price_no: float
    volume_total: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "price_yes": self.price_yes,
            "price_no": self.price_no,
            "volume_total": self.volume_total,
        }


@dataclass(frozen=True)
class CaseLabel:
    """The labeled anomaly window for scoring (consumed by evaluation).

    Attributes:
        window_start: "%Y-%m-%d %H:%M:%S" UTC TEXT timestamp.
        window_end: "%Y-%m-%d %H:%M:%S" UTC TEXT timestamp.
        event_time: "%Y-%m-%d %H:%M:%S" UTC TEXT timestamp of the real-world
            event the labeled window anticipates.
    """

    window_start: str
    window_end: str
    event_time: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start,
            "window_end": self.window_end,
            "event_time": self.event_time,
        }


@dataclass
class Case:
    """A fully loaded backtest case.

    Attributes:
        slug: Short case identifier (also the directory name by convention).
        market_id: Polymarket market id -- must match ``market.id``.
        condition_id: Polymarket condition id.
        question: The market question, duplicated from the manifest for
            convenience (also present on ``market``).
        archived_at: "%Y-%m-%d %H:%M:%S" UTC TEXT timestamp of when the
            case was archived.
        notes: Free-text notes about the case.
        market: The parsed Gamma-style market.
        snapshots: Time-ascending replay spine.
        trades: Parsed Data-API trade models.
        events: Parsed scheduled events (empty if ``events.json`` absent).
        label: The labeled anomaly window, or ``None`` if ``[label]`` was
            not present in the manifest.
    """

    slug: str
    market_id: str
    condition_id: str
    question: str
    archived_at: str
    notes: str
    market: GammaMarket
    snapshots: list[SnapshotRow]
    trades: list[Trade]
    events: list[ScheduledEvent] = field(default_factory=list)
    label: CaseLabel | None = None


def _require_file(path: Path, name: str) -> Path:
    p = path / name
    if not p.exists():
        raise ValueError(f"{name}: missing required file at {p}")
    return p


def load_case(path: Path) -> Case:
    """Load and validate a case directory.

    Args:
        path: Path to the case directory.

    Returns:
        A fully parsed and validated :class:`Case`.

    Raises:
        ValueError: If a required file is missing, snapshots are out of
            time order, or the manifest ``market_id`` does not match
            ``market.json``'s ``id`` -- each naming the offending file.
    """
    path = Path(path)

    manifest_path = _require_file(path, "case.toml")
    with manifest_path.open("rb") as f:
        manifest = tomllib.load(f)

    case_section = manifest.get("case")
    if not case_section:
        raise ValueError("case.toml: missing required [case] section")

    market_path = _require_file(path, "market.json")
    market_data = json.loads(market_path.read_text(encoding="utf-8"))
    market = GammaMarket.model_validate(market_data)

    manifest_market_id = case_section.get("market_id", "")
    if manifest_market_id != market.id:
        raise ValueError(
            f"case.toml: manifest market_id {manifest_market_id!r} does not "
            f"match market.json id {market.id!r}"
        )

    snapshots_path = _require_file(path, "snapshots.json")
    snapshots_data = json.loads(snapshots_path.read_text(encoding="utf-8"))
    snapshots = [
        SnapshotRow(
            timestamp=row["timestamp"],
            price_yes=row["price_yes"],
            price_no=row["price_no"],
            volume_total=row["volume_total"],
        )
        for row in snapshots_data
    ]
    for prev, cur in zip(snapshots, snapshots[1:]):
        if cur.timestamp < prev.timestamp:
            raise ValueError(
                "snapshots.json: rows are not in time-ascending order "
                f"({prev.timestamp!r} followed by {cur.timestamp!r})"
            )

    trades_path = _require_file(path, "trades.json")
    trades_data = json.loads(trades_path.read_text(encoding="utf-8"))
    trades = [Trade.model_validate(row) for row in trades_data]

    events: list[ScheduledEvent] = []
    events_path = path / "events.json"
    if events_path.exists():
        events_data = json.loads(events_path.read_text(encoding="utf-8"))
        for row in events_data:
            events.append(
                ScheduledEvent(
                    source=row["source"],
                    event_type=row["event_type"],
                    title=row["title"],
                    description=row.get("description", ""),
                    event_date=datetime.strptime(row["event_date"], _EVENT_DATE_FORMAT),
                    url=row.get("url", ""),
                    keywords=row.get("keywords", []),
                )
            )

    label: CaseLabel | None = None
    label_section = manifest.get("label")
    if label_section:
        label = CaseLabel(
            window_start=label_section["window_start"],
            window_end=label_section["window_end"],
            event_time=label_section["event_time"],
        )

    return Case(
        slug=case_section.get("slug", ""),
        market_id=manifest_market_id,
        condition_id=case_section.get("condition_id", ""),
        question=case_section.get("question", ""),
        archived_at=case_section.get("archived_at", ""),
        notes=case_section.get("notes", ""),
        market=market,
        snapshots=snapshots,
        trades=trades,
        events=events,
        label=label,
    )


def save_case(path: Path, case: Case) -> None:
    """Write *case* to *path* in the on-disk case format.

    The round-trip partner of :func:`load_case`: ``load_case(save_case(p,
    c))`` reproduces *c* field-wise. Creates *path* if it does not exist.

    Args:
        path: Destination directory (created if missing).
        case: The case to serialize.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "case": {
            "slug": case.slug,
            "market_id": case.market_id,
            "condition_id": case.condition_id,
            "question": case.question,
            "archived_at": case.archived_at,
            "notes": case.notes,
        }
    }
    if case.label is not None:
        manifest["label"] = case.label.to_dict()

    (path / "case.toml").write_bytes(tomli_w.dumps(manifest).encode("utf-8"))

    (path / "market.json").write_text(
        json.dumps(case.market.model_dump(by_alias=True), indent=2, default=str),
        encoding="utf-8",
    )

    (path / "snapshots.json").write_text(
        json.dumps([s.to_dict() for s in case.snapshots], indent=2),
        encoding="utf-8",
    )

    (path / "trades.json").write_text(
        json.dumps(
            [t.model_dump(by_alias=True) for t in case.trades], indent=2, default=str
        ),
        encoding="utf-8",
    )

    (path / "events.json").write_text(
        json.dumps(
            [
                {
                    "source": e.source,
                    "event_type": e.event_type,
                    "title": e.title,
                    "description": e.description,
                    "event_date": e.event_date.strftime(_EVENT_DATE_FORMAT),
                    "url": e.url,
                    "keywords": e.keywords,
                }
                for e in case.events
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
