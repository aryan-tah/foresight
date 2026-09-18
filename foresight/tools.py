"""The tools Claude can call, and the dispatcher that runs them.

Tool results are deliberately small (summaries, not raw arrays). The full data
lives in the session's SeriesStore; charts go to the browser as UI events.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import httpx

from . import sources
from .forecaster import MAX_HORIZON, Model, run_forecast
from .series import SeriesStore

_DAYS = {
    "type": "integer",
    "description": "Days of history to fetch (60-1000, default 400). Use 1000 for anything seasonal or yearly.",
}

TOOLS = [
    {
        "name": "search_wikipedia",
        "description": (
            "Find the exact English Wikipedia article title for a topic. Call this "
            "before get_wikipedia_pageviews unless you are certain of the title."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_wikipedia_pageviews",
        "description": (
            "Daily human pageviews of an English Wikipedia article. The best free "
            "proxy for public attention/interest in a person, product, event, "
            "company or idea. Returns a series_id plus summary stats."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "article": {"type": "string", "description": "Exact title, e.g. 'ChatGPT'"},
                "days": _DAYS,
            },
            "required": ["article"],
        },
    },
    {
        "name": "get_package_downloads",
        "description": (
            "Daily download counts for a software package from npm or PyPI. A proxy "
            "for developer adoption of a library, framework or AI SDK. PyPI history "
            "is limited to ~180 days. Returns a series_id plus summary stats."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "registry": {"type": "string", "enum": ["npm", "pypi"]},
                "package": {"type": "string", "description": "e.g. 'react', 'anthropic'"},
                "days": _DAYS,
            },
            "required": ["registry", "package"],
        },
    },
    {
        "name": "get_crypto_prices",
        "description": (
            "Daily USD price of a cryptocurrency (max 365 days). Prices are close to "
            "a random walk: forecasts are weak and you must say so. Returns a "
            "series_id plus summary stats."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "coin_id": {
                    "type": "string",
                    "description": "CoinGecko id: 'bitcoin', 'ethereum', 'solana', 'dogecoin'...",
                },
                "days": _DAYS,
            },
            "required": ["coin_id"],
        },
    },
    {
        "name": "get_weather_history",
        "description": (
            "Daily weather history for a city: mean temperature (°C), precipitation "
            "(mm) or max wind (km/h). Use 700+ days so the model sees a full "
            "seasonal cycle. Returns a series_id plus summary stats."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "variable": {
                    "type": "string",
                    "enum": sorted(sources.WEATHER_VARIABLES),
                },
                "days": _DAYS,
            },
            "required": ["city", "variable"],
        },
    },
    {
        "name": "forecast",
        "description": (
            "Forecast a fetched series with Google's TimesFM 3.0 time-series "
            "foundation model. Draws a chart with uncertainty bands for the user "
            "and returns the median path, p10/p90 band, and a backtest showing how "
            "well the model would have predicted the most recent period versus a "
            "naive baseline. Always use this for numbers about the future; never "
            "estimate them yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series_id": {"type": "string", "description": "From a data tool."},
                "horizon_days": {
                    "type": "integer",
                    "description": f"Days ahead to forecast, 1-{MAX_HORIZON}.",
                },
            },
            "required": ["series_id", "horizon_days"],
        },
    },
]


@dataclass
class ToolContext:
    """Everything a tool call needs; one per chat session."""

    http: httpx.AsyncClient
    model: Model
    store: SeriesStore = field(default_factory=SeriesStore)


@dataclass
class ToolOutcome:
    content: str  # goes back to Claude
    is_error: bool = False
    chart: dict | None = None  # goes to the browser
    note: str = ""  # one-line status for the UI


async def execute(name: str, args: dict, ctx: ToolContext) -> ToolOutcome:
    """Run one tool call. Never raises: failures become error results for Claude."""
    try:
        if not isinstance(args, dict):
            raise ValueError("tool input must be a JSON object")
        return await _HANDLERS[name](args, ctx)
    except KeyError as e:
        if name not in _HANDLERS:
            return ToolOutcome(f"Unknown tool '{name}'.", is_error=True)
        return ToolOutcome(str(e.args[0]), is_error=True, note="failed")
    except (sources.SourceError, ValueError, TypeError) as e:
        return ToolOutcome(str(e), is_error=True, note="failed")


def _fetched(series) -> ToolOutcome:
    return ToolOutcome(
        json.dumps(series.summary()),
        note=f"{len(series.values)} days of data",
    )


async def _search_wikipedia(args, ctx):
    hits = await sources.search_wikipedia(ctx.http, _req(args, "query"))
    return ToolOutcome(json.dumps(hits), note=f"{len(hits)} matches")


async def _wikipedia(args, ctx):
    s = await sources.wikipedia_pageviews(ctx.http, _req(args, "article"), args.get("days", 400))
    return _fetched(ctx.store.add(s))


async def _packages(args, ctx):
    s = await sources.package_downloads(
        ctx.http, _req(args, "registry"), _req(args, "package"), args.get("days", 400)
    )
    return _fetched(ctx.store.add(s))


async def _crypto(args, ctx):
    s = await sources.crypto_prices(ctx.http, _req(args, "coin_id"), args.get("days", 365))
    return _fetched(ctx.store.add(s))


async def _weather(args, ctx):
    s = await sources.weather_history(
        ctx.http, _req(args, "city"), _req(args, "variable"), args.get("days", 730)
    )
    return _fetched(ctx.store.add(s))


async def _forecast(args, ctx):
    series = ctx.store.get(_req(args, "series_id"))
    horizon = int(_req(args, "horizon_days"))
    # TimesFM is CPU-bound; keep the event loop free while it runs.
    result = await asyncio.to_thread(run_forecast, series, horizon, ctx.model)
    return ToolOutcome(
        json.dumps(result.summary()),
        chart=result.chart(),
        note=f"{horizon}-day forecast",
    )


def _req(args: dict, key: str):
    if args.get(key) in (None, ""):
        raise ValueError(f"Missing required parameter '{key}'.")
    return args[key]


_HANDLERS = {
    "search_wikipedia": _search_wikipedia,
    "get_wikipedia_pageviews": _wikipedia,
    "get_package_downloads": _packages,
    "get_crypto_prices": _crypto,
    "get_weather_history": _weather,
    "forecast": _forecast,
}
