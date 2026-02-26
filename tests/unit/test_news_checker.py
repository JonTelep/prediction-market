"""Unit tests for NewsChecker with mocked HTTP."""

from datetime import datetime, timezone

import httpx
import pytest
import respx

from prediction_market.config import load_config
from prediction_market.data.external.news_checker import NewsChecker


@pytest.fixture
def config():
    cfg = load_config()
    cfg.newsapi_key = "test-key-123"
    return cfg


@pytest.fixture
def config_no_newsapi():
    cfg = load_config()
    cfg.newsapi_key = ""
    return cfg


def _gdelt_response(articles):
    return {"articles": articles}


def _newsapi_response(articles):
    return {"status": "ok", "totalResults": len(articles), "articles": articles}


def _gdelt_article(title, url, seendate):
    return {"title": title, "url": url, "seendate": seendate, "domain": "example.com"}


def _newsapi_article(title, url, published_at):
    return {
        "title": title,
        "url": url,
        "publishedAt": published_at,
        "source": {"name": "TestNews"},
        "description": f"Description for {title}",
    }


@pytest.mark.asyncio
@respx.mock
async def test_news_found_gdelt_only(config_no_newsapi):
    before_time = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)

    respx.get(url__startswith=config_no_newsapi.apis.gdelt_base_url).mock(
        return_value=httpx.Response(
            200,
            json=_gdelt_response([
                _gdelt_article(
                    "Bill passes committee",
                    "https://news.example.com/1",
                    "20260220100000",
                ),
            ]),
        )
    )

    checker = NewsChecker(config_no_newsapi)
    try:
        result = await checker.check_news_exists(
            keywords=["infrastructure", "bill"],
            before_time=before_time,
            window_hours=4,
        )
        assert result.news_found is True
        assert len(result.articles) == 1
        assert result.earliest_article_time is not None
        assert result.earliest_article_time < before_time
    finally:
        await checker.close()


@pytest.mark.asyncio
@respx.mock
async def test_news_found_both_sources(config):
    before_time = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)

    respx.get(url__startswith=config.apis.gdelt_base_url).mock(
        return_value=httpx.Response(
            200,
            json=_gdelt_response([
                _gdelt_article(
                    "Committee vote",
                    "https://news.example.com/1",
                    "20260220100000",
                ),
            ]),
        )
    )
    respx.get(url__startswith=config.apis.newsapi_base_url).mock(
        return_value=httpx.Response(
            200,
            json=_newsapi_response([
                _newsapi_article(
                    "Bill advances",
                    "https://news.example.com/2",
                    "2026-02-20T10:30:00Z",
                ),
            ]),
        )
    )

    checker = NewsChecker(config)
    try:
        result = await checker.check_news_exists(
            keywords=["infrastructure", "bill"],
            before_time=before_time,
        )
        assert result.news_found is True
        assert len(result.articles) == 2
    finally:
        await checker.close()


@pytest.mark.asyncio
@respx.mock
async def test_no_news_found(config):
    before_time = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)

    respx.get(url__startswith=config.apis.gdelt_base_url).mock(
        return_value=httpx.Response(200, json=_gdelt_response([]))
    )
    respx.get(url__startswith=config.apis.newsapi_base_url).mock(
        return_value=httpx.Response(200, json=_newsapi_response([]))
    )

    checker = NewsChecker(config)
    try:
        result = await checker.check_news_exists(
            keywords=["classified", "secret"],
            before_time=before_time,
        )
        assert result.news_found is False
        assert result.articles == []
        assert result.earliest_article_time is None
    finally:
        await checker.close()


@pytest.mark.asyncio
@respx.mock
async def test_empty_keywords(config):
    checker = NewsChecker(config)
    try:
        result = await checker.check_news_exists(
            keywords=[],
            before_time=datetime.now(timezone.utc),
        )
        assert result.news_found is False
        assert result.query_keywords == []
    finally:
        await checker.close()


@pytest.mark.asyncio
@respx.mock
async def test_gdelt_api_error(config):
    before_time = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)

    respx.get(url__startswith=config.apis.gdelt_base_url).mock(
        return_value=httpx.Response(500)
    )
    respx.get(url__startswith=config.apis.newsapi_base_url).mock(
        return_value=httpx.Response(200, json=_newsapi_response([]))
    )

    checker = NewsChecker(config)
    try:
        result = await checker.check_news_exists(
            keywords=["test"],
            before_time=before_time,
        )
        # GDELT error is swallowed; result depends on NewsAPI only
        assert result.news_found is False
    finally:
        await checker.close()


@pytest.mark.asyncio
@respx.mock
async def test_newsapi_rate_limit(config):
    before_time = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)

    respx.get(url__startswith=config.apis.gdelt_base_url).mock(
        return_value=httpx.Response(200, json=_gdelt_response([]))
    )
    respx.get(url__startswith=config.apis.newsapi_base_url).mock(
        return_value=httpx.Response(429)
    )

    checker = NewsChecker(config)
    try:
        result = await checker.check_news_exists(
            keywords=["test"],
            before_time=before_time,
        )
        assert result.news_found is False
    finally:
        await checker.close()


@pytest.mark.asyncio
@respx.mock
async def test_deduplication_by_url(config):
    before_time = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
    same_url = "https://news.example.com/same"

    respx.get(url__startswith=config.apis.gdelt_base_url).mock(
        return_value=httpx.Response(
            200,
            json=_gdelt_response([
                _gdelt_article("Story A", same_url, "20260220100000"),
            ]),
        )
    )
    respx.get(url__startswith=config.apis.newsapi_base_url).mock(
        return_value=httpx.Response(
            200,
            json=_newsapi_response([
                _newsapi_article("Story A", same_url, "2026-02-20T10:00:00Z"),
            ]),
        )
    )

    checker = NewsChecker(config)
    try:
        result = await checker.check_news_exists(
            keywords=["test"],
            before_time=before_time,
        )
        assert result.news_found is True
        assert len(result.articles) == 1  # deduplicated
    finally:
        await checker.close()


@pytest.mark.asyncio
@respx.mock
async def test_articles_after_before_time_filtered(config):
    before_time = datetime(2026, 2, 20, 10, 0, 0, tzinfo=timezone.utc)

    respx.get(url__startswith=config.apis.gdelt_base_url).mock(
        return_value=httpx.Response(
            200,
            json=_gdelt_response([
                # This article is AFTER before_time, should be filtered
                _gdelt_article("Late news", "https://late.com", "20260220120000"),
            ]),
        )
    )
    respx.get(url__startswith=config.apis.newsapi_base_url).mock(
        return_value=httpx.Response(200, json=_newsapi_response([]))
    )

    checker = NewsChecker(config)
    try:
        result = await checker.check_news_exists(
            keywords=["test"],
            before_time=before_time,
        )
        assert result.news_found is False
    finally:
        await checker.close()
