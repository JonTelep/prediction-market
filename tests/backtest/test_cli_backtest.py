"""CLI tests for the `backtest` subcommand."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from prediction_market.cli import main as cli_main
from prediction_market.config import load_config

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cases" / "minimal"


def _cli_config(tmp_path):
    cfg = load_config()
    cfg.database.path = str(tmp_path / "cli_backtest.db")
    return cfg


def test_backtest_cli_reports_detected_and_lead_time(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "prediction_market.cli._load", lambda config_path: _cli_config(tmp_path)
    )
    runner = CliRunner()

    result = runner.invoke(cli_main, ["backtest", "--case", str(FIXTURE_DIR)])

    assert result.exit_code == 0, result.output
    assert "Detected:" in result.output
    assert "Lead time:" in result.output
    assert "First hit:" in result.output


def test_backtest_cli_json_parses(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "prediction_market.cli._load", lambda config_path: _cli_config(tmp_path)
    )
    runner = CliRunner()

    result = runner.invoke(
        cli_main, ["backtest", "--case", str(FIXTURE_DIR), "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["case_slug"] == "minimal"
    assert "detected" in payload
    assert "hits" in payload


def test_backtest_cli_with_null_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "prediction_market.cli._load", lambda config_path: _cli_config(tmp_path)
    )
    runner = CliRunner()

    result = runner.invoke(
        cli_main,
        [
            "backtest",
            "--case",
            str(FIXTURE_DIR),
            "--null-runs",
            "2",
            "--seed",
            "1000",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "null_control" in payload
    assert payload["null_control"]["runs"] == 2
