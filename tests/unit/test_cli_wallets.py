"""CLI tests for the `wallets` and `wallet` subcommands."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from click.testing import CliRunner

from prediction_market.cli import main as cli_main
from prediction_market.config import load_config
from prediction_market.data.polymarket.models import Trade
from prediction_market.store.database import init_database
from prediction_market.store.snapshots import save_trades_batch

MARKET_ID = "wallet-market-1"


def _cli_config(tmp_path):
    cfg = load_config()
    cfg.database.path = str(tmp_path / "cli_wallets.db")
    return cfg


def _trade(id_, wallet, side, size, price, match_time, outcome="Yes"):
    return Trade(
        id=id_,
        assetId="tok1",
        side=side,
        size=str(size),
        price=str(price),
        matchTime=match_time,
        outcome=outcome,
        owner="",
        proxyWallet=wallet,
        transactionHash="",
    )


async def _seed(config, market_id, trades):
    db = await init_database(config)
    try:
        await db.execute(
            "INSERT OR IGNORE INTO markets (id, question, category, volume) "
            "VALUES (?, ?, ?, ?)",
            (market_id, "Will X happen?", "politics", 100000),
        )
        await db.commit()
        await save_trades_batch(db, trades, market_id)
    finally:
        await db.close()


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# ---------------------------------------------------------------------------
# wallets
# ---------------------------------------------------------------------------


def test_wallets_cmd_ranks_by_score(tmp_path, monkeypatch):
    config = _cli_config(tmp_path)
    monkeypatch.setattr("prediction_market.cli._load", lambda config_path: config)

    now = datetime.now(timezone.utc)
    recent = now - timedelta(hours=1)

    # Whale wallet: large one-sided BUY volume, well within window.
    whale = "0x" + "a" * 40
    # Small wallet: modest, balanced volume.
    minnow = "0x" + "b" * 40

    trades = [
        _trade("t-whale-1", whale, "BUY", 10000, 0.60, _iso(recent)),
        _trade("t-whale-2", whale, "BUY", 10000, 0.60, _iso(recent)),
        _trade("t-minnow-1", minnow, "BUY", 1000, 0.50, _iso(recent)),
        _trade("t-minnow-2", minnow, "SELL", 1000, 0.50, _iso(recent)),
    ]
    asyncio.run(_seed(config, MARKET_ID, trades))

    # Hand-computed totals:
    # whale volume = 6000 + 6000 = 12000
    # minnow volume = 500 + 500 = 1000
    # total market volume = 13000
    # whale share = 12000/13000 = 0.9231 -> 92.3%
    whale_volume = 6000.0 + 6000.0
    total_volume = whale_volume + 500.0 + 500.0
    whale_share = whale_volume / total_volume

    runner = CliRunner()
    result = runner.invoke(cli_main, ["wallets", MARKET_ID, "--hours", "168"])

    assert result.exit_code == 0, result.output
    abbreviated = f"{whale[:6]}...{whale[-4:]}"
    assert abbreviated in result.output
    assert f"{whale_share:.1%}" in result.output


def test_wallets_cmd_hours_window_excludes_old_trade(tmp_path, monkeypatch):
    config = _cli_config(tmp_path)
    monkeypatch.setattr("prediction_market.cli._load", lambda config_path: config)

    now = datetime.now(timezone.utc)
    recent = now - timedelta(hours=1)
    old = now - timedelta(hours=500)

    wallet = "0x" + "c" * 40

    trades = [
        _trade("t-recent", wallet, "BUY", 5000, 0.50, _iso(recent)),
        _trade("t-old", wallet, "BUY", 9000, 0.50, _iso(old)),
    ]
    asyncio.run(_seed(config, MARKET_ID, trades))

    runner = CliRunner()
    result = runner.invoke(cli_main, ["wallets", MARKET_ID, "--hours", "168"])

    assert result.exit_code == 0, result.output
    # Only the recent trade's volume (5000 * 0.50 = 2500) should count;
    # the old trade's volume (9000 * 0.50 = 4500) must not appear anywhere
    # in the totals-bearing output.
    assert "$2,500" in result.output
    assert "$4,500" not in result.output


def test_wallets_cmd_empty_db_prints_no_data_line(tmp_path, monkeypatch):
    config = _cli_config(tmp_path)
    monkeypatch.setattr("prediction_market.cli._load", lambda config_path: config)

    async def _init():
        db = await init_database(config)
        await db.close()

    asyncio.run(_init())

    runner = CliRunner()
    result = runner.invoke(cli_main, ["wallets", "no-such-market"])

    assert result.exit_code == 0, result.output
    assert "no wallet-attributed trades" in result.output.lower()
    assert "Phase 2" in result.output


# ---------------------------------------------------------------------------
# wallet
# ---------------------------------------------------------------------------


def test_wallet_cmd_totals(tmp_path, monkeypatch):
    config = _cli_config(tmp_path)
    monkeypatch.setattr("prediction_market.cli._load", lambda config_path: config)

    now = datetime.now(timezone.utc)
    wallet = "0x" + "d" * 40

    trades = [
        _trade("t1", wallet, "BUY", 1000, 0.50, _iso(now - timedelta(hours=3))),
        _trade("t2", wallet, "SELL", 500, 0.40, _iso(now - timedelta(hours=2))),
        _trade("t3", wallet, "BUY", 200, 0.90, _iso(now - timedelta(hours=1))),
    ]
    asyncio.run(_seed(config, MARKET_ID, trades))

    # Hand-computed totals:
    # buy volume = 1000*0.50 + 200*0.90 = 500 + 180 = 680
    # sell volume = 500*0.40 = 200
    # total volume = 880
    buy_volume = 1000 * 0.50 + 200 * 0.90
    sell_volume = 500 * 0.40
    total_volume = buy_volume + sell_volume

    runner = CliRunner()
    result = runner.invoke(cli_main, ["wallet", wallet])

    assert result.exit_code == 0, result.output
    assert "Total: 3 trade(s)" in result.output
    assert f"${total_volume:,.2f}" in result.output
    assert f"${buy_volume:,.2f}" in result.output
    assert f"${sell_volume:,.2f}" in result.output


def test_wallet_cmd_unknown_address_empty_state(tmp_path, monkeypatch):
    config = _cli_config(tmp_path)
    monkeypatch.setattr("prediction_market.cli._load", lambda config_path: config)

    async def _init():
        db = await init_database(config)
        await db.close()

    asyncio.run(_init())

    runner = CliRunner()
    result = runner.invoke(cli_main, ["wallet", "0x" + "9" * 40])

    assert result.exit_code == 0, result.output
    assert "No trades found" in result.output
