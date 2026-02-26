"""Configuration loading from TOML files and environment variables."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


class DatabaseConfig(BaseModel):
    path: str = "data/prediction_market.db"


class PollingConfig(BaseModel):
    snapshot_interval_seconds: int = 60
    orderbook_interval_seconds: int = 300
    market_discovery_interval_seconds: int = 3600
    event_refresh_interval_seconds: int = 7200


class ThresholdConfig(BaseModel):
    price_zscore: float = 2.5
    volume_zscore: float = 3.0
    combined_score_min: float = 4.0
    event_proximity_hours: int = 24
    event_amplifier: float = 1.5
    news_dampener: float = 0.3
    rolling_window_days: int = 7
    susceptibility_threshold: float = 0.7
    liquidity_drop_pct: float = 0.50
    depth_weight: float = 0.30
    spread_weight: float = 0.25
    concentration_weight: float = 0.25
    imbalance_weight: float = 0.20


class RateLimitConfig(BaseModel):
    gamma_requests_per_window: int = 4000
    gamma_window_seconds: int = 10
    clob_requests_per_window: int = 9000
    clob_window_seconds: int = 10
    data_requests_per_window: int = 1000
    data_window_seconds: int = 10


class ReportingConfig(BaseModel):
    output_dir: str = "data/reports"
    formats: list[str] = Field(default_factory=lambda: ["json", "markdown"])
    webhook_url: str = ""


class WebSocketConfig(BaseModel):
    clob_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    rtds_url: str = "wss://ws-live-data.polymarket.com/ws"
    reconnect_delay_seconds: int = 5
    max_reconnect_attempts: int = 10


class APIsConfig(BaseModel):
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    data_base_url: str = "https://data-api.polymarket.com"
    congress_base_url: str = "https://api.congress.gov/v3"
    courtlistener_base_url: str = "https://www.courtlistener.com/api/rest/v4"
    gdelt_base_url: str = "https://api.gdeltproject.org/api/v2"
    newsapi_base_url: str = "https://newsapi.org/v2"


class AppConfig(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    polling: PollingConfig = Field(default_factory=PollingConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    rate_limits: RateLimitConfig = Field(default_factory=RateLimitConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    apis: APIsConfig = Field(default_factory=APIsConfig)

    # API keys from env vars
    congress_api_key: str = ""
    courtlistener_token: str = ""
    newsapi_key: str = ""
    webhook_url: str = ""
    log_level: str = "INFO"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load configuration from TOML files and environment variables.

    Priority: env vars > custom config > default.toml
    """
    # Load default config
    default_path = _CONFIG_DIR / "default.toml"
    data: dict[str, Any] = {}
    if default_path.exists():
        with open(default_path, "rb") as f:
            data = tomllib.load(f)

    # Load custom config override
    if config_path and config_path.exists():
        with open(config_path, "rb") as f:
            custom = tomllib.load(f)
        data = _deep_merge(data, custom)

    # Apply env var overrides
    env_overrides: dict[str, Any] = {}
    if db_path := os.getenv("DATABASE_PATH"):
        env_overrides.setdefault("database", {})["path"] = db_path

    data = _deep_merge(data, env_overrides)

    # Build config with env var API keys
    config = AppConfig(
        **{k: v for k, v in data.items() if k in AppConfig.model_fields},
        congress_api_key=os.getenv("CONGRESS_API_KEY", ""),
        courtlistener_token=os.getenv("COURTLISTENER_TOKEN", ""),
        newsapi_key=os.getenv("NEWSAPI_KEY", ""),
        webhook_url=os.getenv("WEBHOOK_URL", ""),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
    return config


def load_political_keywords() -> dict[str, Any]:
    """Load political keywords configuration."""
    path = _CONFIG_DIR / "political_keywords.toml"
    if path.exists():
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {"classification": {"political_tags": [], "title_keywords": [], "political_categories": [], "min_volume_usd": 10000}}
