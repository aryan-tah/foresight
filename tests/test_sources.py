from datetime import date

import pytest
import respx

from foresight import sources


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr(sources, "yesterday", lambda: date(2026, 9, 17))


@respx.mock
async def test_wikipedia_gaps_become_zero_view_days():
    respx.get(url__startswith="https://wikimedia.org/api/rest_v1/metrics/pageviews").respond(
        json={"items": [
            {"timestamp": "2026091400", "views": 10},
            {"timestamp": "2026091700", "views": 40},
        ]}
    )
    async with sources.new_client() as c:
        s = await sources.wikipedia_pageviews(c, "Some Article", 90)
    assert s.id == "wiki:Some_Article"
    assert s.values == [10, 0, 0, 40]
    assert s.dates[0] == date(2026, 9, 14) and s.dates[-1] == date(2026, 9, 17)


@respx.mock
async def test_npm_drops_trailing_days_the_registry_has_not_counted_yet():
    respx.get(url__startswith="https://api.npmjs.org/downloads/range").respond(
        json={"downloads": [
            {"day": "2026-09-14", "downloads": 500},
            {"day": "2026-09-15", "downloads": 700},
            {"day": "2026-09-16", "downloads": 0},
            {"day": "2026-09-17", "downloads": 0},
        ]}
    )
    async with sources.new_client() as c:
        s = await sources.package_downloads(c, "npm", "react", 90)
    assert s.values == [500, 700]
    assert s.dates[-1] == date(2026, 9, 15)


@respx.mock
async def test_crypto_ignores_todays_partial_price_and_forward_fills():
    from datetime import datetime, timezone

    ts = lambda y, m, d, h=0: int(datetime(y, m, d, h, tzinfo=timezone.utc).timestamp() * 1000)
    respx.get(url__startswith="https://api.coingecko.com").respond(
        json={"prices": [
            [ts(2026, 9, 15), 100.0],
            [ts(2026, 9, 17), 120.0],
            [ts(2026, 9, 18, 9), 999.0],  # today, still moving
        ]}
    )
    async with sources.new_client() as c:
        s = await sources.crypto_prices(c, "Bitcoin", 365)
    assert s.id == "crypto:bitcoin"
    assert s.values == [100.0, 100.0, 120.0]


@respx.mock
async def test_not_found_is_explained_to_the_agent():
    respx.get(url__startswith="https://wikimedia.org").respond(404)
    async with sources.new_client() as c:
        with pytest.raises(sources.SourceError, match="not found"):
            await sources.wikipedia_pageviews(c, "Nope", 90)


@respx.mock
async def test_weather_trims_the_archives_unfinished_recent_days():
    respx.get(url__startswith="https://geocoding-api.open-meteo.com").respond(
        json={"results": [{"name": "Mumbai", "country_code": "IN", "latitude": 19, "longitude": 72}]}
    )
    respx.get(url__startswith="https://archive-api.open-meteo.com").respond(
        json={"daily": {
            "time": ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17"],
            "temperature_2m_mean": [27.0, 28.0, None, None],
        }}
    )
    async with sources.new_client() as c:
        s = await sources.weather_history(c, "Mumbai", "temperature", 90)
    assert s.values == [27.0, 28.0]
    assert s.unit == "°C"
