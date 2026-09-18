"""Live public data sources. All free, none need an API key.

Each fetcher returns a gap-free daily Series ending yesterday (UTC) at the
latest, so a half-finished "today" never drags a forecast down.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote

import httpx

from .series import Series, daily_range

USER_AGENT = "foresight-agent/0.1 (https://github.com/aryan-tah/foresight)"
MAX_DAYS = 1000

WEATHER_VARIABLES = {
    "temperature": ("temperature_2m_mean", "°C"),
    "precipitation": ("precipitation_sum", "mm"),
    "wind": ("wind_speed_10m_max", "km/h"),
}


class SourceError(Exception):
    """A data source could not answer; the message is written for the LLM."""


def new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=20, follow_redirects=True
    )


def yesterday() -> date:
    return datetime.now(timezone.utc).date() - timedelta(days=1)


async def _get_json(client: httpx.AsyncClient, url: str, what: str, **params):
    try:
        resp = await client.get(url, params=params or None)
    except httpx.HTTPError as e:
        raise SourceError(f"Could not reach {what}: {e!r}") from e
    if resp.status_code == 404:
        raise SourceError(f"{what}: not found. Check the name/spelling.")
    if resp.status_code == 429:
        raise SourceError(f"{what}: rate limited right now. Try another source.")
    if resp.status_code >= 400:
        raise SourceError(f"{what}: HTTP {resp.status_code}")
    return resp.json()


def _daily(points: dict[date, float], end: date, fill_zero: bool) -> tuple[list[date], list[float]]:
    """Make a gap-free daily series. Gaps are zero-filled or forward-filled."""
    points = {d: v for d, v in points.items() if d <= end and v is not None}
    if not points:
        raise SourceError("The source returned no data for that range.")
    d, last = min(points), None
    dates, values = [], []
    while d <= max(points):
        last = points.get(d, 0.0 if fill_zero else last)
        dates.append(d)
        values.append(float(last))
        d += timedelta(days=1)
    return dates, values


async def search_wikipedia(client: httpx.AsyncClient, query: str) -> list[dict]:
    data = await _get_json(
        client,
        "https://en.wikipedia.org/w/api.php",
        "Wikipedia search",
        action="opensearch",
        search=query,
        limit=6,
        format="json",
    )
    titles, descs = data[1], data[2]
    return [{"article": t.replace(" ", "_"), "hint": d} for t, d in zip(titles, descs)]


async def wikipedia_pageviews(client: httpx.AsyncClient, article: str, days: int) -> Series:
    article = article.strip().replace(" ", "_")
    start, end = daily_range(yesterday(), _clamp(days))
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia/all-access/user/{quote(article, safe='')}/daily/"
        f"{start:%Y%m%d}/{end:%Y%m%d}"
    )
    data = await _get_json(client, url, f"Wikipedia pageviews for '{article}'")
    points = {
        datetime.strptime(i["timestamp"][:8], "%Y%m%d").date(): i["views"]
        for i in data["items"]
    }
    dates, values = _daily(points, end, fill_zero=True)
    return Series(
        id=f"wiki:{article}",
        label=f"Wikipedia pageviews — {article.replace('_', ' ')}",
        unit="views/day",
        dates=dates,
        values=values,
        source_url=f"https://en.wikipedia.org/wiki/{quote(article)}",
    )


async def package_downloads(
    client: httpx.AsyncClient, registry: str, package: str, days: int
) -> Series:
    package = package.strip()
    end = yesterday()
    if registry == "npm":
        start, _ = daily_range(end, min(_clamp(days), 540))  # npm caps at 18 months
        data = await _get_json(
            client,
            f"https://api.npmjs.org/downloads/range/{start}:{end}/{quote(package, safe='@')}",
            f"npm downloads for '{package}'",
        )
        points = {date.fromisoformat(i["day"]): i["downloads"] for i in data["downloads"]}
        source = f"https://www.npmjs.com/package/{package}"
    elif registry == "pypi":
        data = await _get_json(
            client,
            f"https://pypistats.org/api/packages/{quote(package.lower())}/overall",
            f"PyPI downloads for '{package}'",
            mirrors="false",
        )
        points = {date.fromisoformat(i["date"]): i["downloads"] for i in data["data"]}
        source = f"https://pypistats.org/packages/{package.lower()}"
    else:
        raise SourceError("registry must be 'npm' or 'pypi'")

    dates, values = _daily(points, end, fill_zero=True)
    # Registries publish a day late: the newest days show 0 until counted.
    while len(values) > 1 and values[-1] == 0:
        dates.pop()
        values.pop()
    return Series(
        id=f"{registry}:{package}",
        label=f"{registry} downloads — {package}",
        unit="downloads/day",
        dates=dates,
        values=values,
        source_url=source,
    )


async def crypto_prices(client: httpx.AsyncClient, coin_id: str, days: int) -> Series:
    coin_id = coin_id.strip().lower()
    data = await _get_json(
        client,
        f"https://api.coingecko.com/api/v3/coins/{quote(coin_id)}/market_chart",
        f"CoinGecko prices for '{coin_id}' (ids look like 'bitcoin', 'ethereum')",
        vs_currency="usd",
        days=min(_clamp(days), 365),  # free tier limit
        interval="daily",
    )
    points = {
        datetime.fromtimestamp(ms / 1000, timezone.utc).date(): price
        for ms, price in data["prices"]
    }
    dates, values = _daily(points, yesterday(), fill_zero=False)
    return Series(
        id=f"crypto:{coin_id}",
        label=f"{coin_id.title()} price",
        unit="USD",
        dates=dates,
        values=values,
        source_url=f"https://www.coingecko.com/en/coins/{coin_id}",
    )


async def weather_history(
    client: httpx.AsyncClient, city: str, variable: str, days: int
) -> Series:
    if variable not in WEATHER_VARIABLES:
        raise SourceError(f"variable must be one of {sorted(WEATHER_VARIABLES)}")
    field, unit = WEATHER_VARIABLES[variable]

    geo = await _get_json(
        client,
        "https://geocoding-api.open-meteo.com/v1/search",
        f"geocoding for '{city}'",
        name=city,
        count=1,
    )
    if not geo.get("results"):
        raise SourceError(f"Could not find a city called '{city}'.")
    place = geo["results"][0]

    start, end = daily_range(yesterday(), _clamp(days))
    data = await _get_json(
        client,
        "https://archive-api.open-meteo.com/v1/archive",
        f"weather archive for '{city}'",
        latitude=place["latitude"],
        longitude=place["longitude"],
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        daily=field,
        timezone="UTC",
    )
    # The archive lags a few days; the newest entries come back as null.
    points = {
        date.fromisoformat(d): v
        for d, v in zip(data["daily"]["time"], data["daily"][field])
    }
    dates, values = _daily(points, end, fill_zero=False)
    name = f"{place['name']}, {place.get('country_code', '')}".strip(", ")
    return Series(
        id=f"weather:{place['name'].lower().replace(' ', '-')}:{variable}",
        label=f"Daily {variable} — {name}",
        unit=unit,
        dates=dates,
        values=values,
        source_url="https://open-meteo.com/",
    )


def _clamp(days: int) -> int:
    return max(60, min(int(days), MAX_DAYS))
