"""Tests for the case archiver (`archive_case`) and its CLI commands.

All HTTP is respx-mocked. respx raises on any unmocked route, so these
tests also prove exactly which Gamma/CLOB/Data endpoints `archive_case`
touches.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import respx
from click.testing import CliRunner

from prediction_market.backtest.archiver import archive_case
from prediction_market.backtest.case_format import load_case
from prediction_market.backtest.replay import replay_case
from prediction_market.cli import main as cli_main
from prediction_market.config import load_config

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

SLUG = "president-sign-infrastructure-bill"


def _epoch(iso: str) -> int:
    return int(
        datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )


def _trade(idx: int, match_time: str, price: float, size: float) -> dict:
    return {
        "id": f"trade-{idx}",
        "takerOrderId": f"order-{idx}",
        "market": "fixture-market-1",
        "assetId": "token-yes-1",
        "side": "BUY",
        "size": str(size),
        "feeRateBps": "0",
        "price": str(price),
        "status": "MATCHED",
        "matchTime": match_time,
        "outcome": "Yes",
        "bucketIndex": "0",
        "owner": "0xowner",
        "proxyWallet": "0xwallet",
        "transactionHash": f"0xtx{idx}",
    }


@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg.database.path = str(tmp_path / "test.db")
    return cfg


@pytest.fixture
def gamma_market_record():
    with open(FIXTURES / "gamma_markets.json") as f:
        markets = json.load(f)
    (record,) = [m for m in markets if m["slug"] == SLUG]
    return record


@pytest.fixture
def price_history_data():
    return {
        "history": [
            {"t": _epoch("2026-03-01T00:00:00Z"), "p": 0.50},
            {"t": _epoch("2026-03-01T01:00:00Z"), "p": 0.55},
            {"t": _epoch("2026-03-01T02:00:00Z"), "p": 0.60},
        ]
    }


@pytest.fixture
def trade_pages():
    # Page 1: exactly 100 trades (the archiver's hardcoded page limit) so
    # pagination continues to a second, short page -- all matched well
    # before any price point, contributing a flat +100 volume baseline.
    page1 = [_trade(i, "2026-02-28T00:00:00Z", 1, 1) for i in range(100)]
    # Page 2: short (< 100), proving pagination stopped here. Straddles
    # the first and second price points.
    page2 = [
        _trade(200, "2026-03-01T00:30:00Z", 2, 10),  # volume 20
        _trade(201, "2026-03-01T01:30:00Z", 4, 5),  # volume 20
    ]
    return page1, page2


def _mock_gamma(gamma_market_record, base_url):
    respx.get(f"{base_url}/markets").mock(
        return_value=httpx.Response(200, json=[gamma_market_record])
    )


@pytest.mark.asyncio
@respx.mock
async def test_archive_case_happy_path(
    config, tmp_path, gamma_market_record, price_history_data, trade_pages
):
    _mock_gamma(gamma_market_record, config.apis.gamma_base_url)
    respx.get(f"{config.apis.clob_base_url}/prices-history").mock(
        return_value=httpx.Response(200, json=price_history_data)
    )
    page1, page2 = trade_pages
    trades_route = respx.get(f"{config.apis.data_base_url}/trades")
    trades_route.side_effect = [
        httpx.Response(200, json=page1),
        httpx.Response(200, json=page2),
    ]

    out_dir = tmp_path / "cases"
    case_dir = await archive_case(config, SLUG, out_dir)

    assert case_dir == out_dir / SLUG

    case = load_case(case_dir)
    assert case.slug == SLUG
    assert case.market_id == gamma_market_record["id"]
    assert case.condition_id == gamma_market_record["conditionId"]
    assert case.label is None
    assert len(case.trades) == 102

    assert len(case.snapshots) == 3
    volumes = [s.volume_total for s in case.snapshots]
    assert volumes == pytest.approx([100.0, 120.0, 140.0])

    prices_yes = [s.price_yes for s in case.snapshots]
    prices_no = [s.price_no for s in case.snapshots]
    assert prices_yes == pytest.approx([0.50, 0.55, 0.60])
    assert prices_no == pytest.approx([0.50, 0.45, 0.40])

    # Smoke: the archived case round-trips into the replay harness without
    # error. No assertions on the emitted report content -- that's
    # test_replay.py's job.
    result = await replay_case(case, config)
    assert result.steps == 3


@pytest.mark.asyncio
@respx.mock
async def test_archive_case_empty_price_history(config, tmp_path, gamma_market_record):
    _mock_gamma(gamma_market_record, config.apis.gamma_base_url)
    respx.get(f"{config.apis.clob_base_url}/prices-history").mock(
        return_value=httpx.Response(200, json={"history": []})
    )
    respx.get(f"{config.apis.data_base_url}/trades").mock(
        return_value=httpx.Response(200, json=[])
    )

    out_dir = tmp_path / "cases"
    case_dir = await archive_case(config, SLUG, out_dir)

    case = load_case(case_dir)
    assert case.snapshots == []
    assert "price_history: empty" in case.notes


@pytest.mark.asyncio
@respx.mock
async def test_archive_case_zero_gamma_matches_raises(config, tmp_path):
    respx.get(f"{config.apis.gamma_base_url}/markets").mock(
        return_value=httpx.Response(200, json=[])
    )

    with pytest.raises(ValueError, match="no-such-slug"):
        await archive_case(config, "no-such-slug", tmp_path / "cases")


@pytest.mark.asyncio
@respx.mock
async def test_archive_case_trade_cap_warning(
    config, tmp_path, gamma_market_record, price_history_data, caplog
):
    _mock_gamma(gamma_market_record, config.apis.gamma_base_url)
    respx.get(f"{config.apis.clob_base_url}/prices-history").mock(
        return_value=httpx.Response(200, json=price_history_data)
    )

    # max_trade_pages=2, with the archiver's fixed page limit of 100:
    # mock exactly 2 full (100-trade) pages so the loop runs to the cap
    # without a short page ever appearing.
    full_page_1 = [_trade(i, "2026-02-28T00:00:00Z", 1, 1) for i in range(100)]
    full_page_2 = [_trade(100 + i, "2026-02-28T00:00:00Z", 1, 1) for i in range(100)]
    trades_route = respx.get(f"{config.apis.data_base_url}/trades")
    trades_route.side_effect = [
        httpx.Response(200, json=full_page_1),
        httpx.Response(200, json=full_page_2),
    ]

    with caplog.at_level(logging.WARNING):
        case_dir = await archive_case(
            config, SLUG, tmp_path / "cases", max_trade_pages=2
        )

    assert any(
        "truncat" in rec.message.lower() or "cap" in rec.message.lower()
        for rec in caplog.records
    )

    case = load_case(case_dir)
    assert len(case.trades) == 200
    assert "page cap" in case.notes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@respx.mock
def test_cli_archive_case_and_cases(
    tmp_path, gamma_market_record, price_history_data, trade_pages, monkeypatch
):
    config = load_config()
    monkeypatch.setattr(
        "prediction_market.cli._load", lambda config_path: _cli_config(tmp_path)
    )

    _mock_gamma(gamma_market_record, config.apis.gamma_base_url)
    respx.get(f"{config.apis.clob_base_url}/prices-history").mock(
        return_value=httpx.Response(200, json=price_history_data)
    )
    page1, page2 = trade_pages
    trades_route = respx.get(f"{config.apis.data_base_url}/trades")
    trades_route.side_effect = [
        httpx.Response(200, json=page1),
        httpx.Response(200, json=page2),
    ]

    out_dir = tmp_path / "cases"
    runner = CliRunner()

    result = runner.invoke(
        cli_main,
        ["archive-case", SLUG, "--output", str(out_dir)],
    )
    assert result.exit_code == 0, result.output
    assert str(out_dir / SLUG) in result.output
    assert "snapshots=3" in result.output
    assert "trades=102" in result.output

    result = runner.invoke(cli_main, ["cases", "--dir", str(out_dir)])
    assert result.exit_code == 0, result.output
    assert SLUG[:28] in result.output
    assert "Total: 1 archived case" in result.output


def _cli_config(tmp_path):
    cfg = load_config()
    cfg.database.path = str(tmp_path / "cli_test.db")
    return cfg
