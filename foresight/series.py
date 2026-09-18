"""Time series container and the per-conversation store the agent's tools share.

Data tools put a Series in the store and hand Claude only a short summary plus
the series id; the forecast tool looks the full data up by id. That keeps
thousands of raw numbers out of the LLM's context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import mean


@dataclass
class Series:
    id: str
    label: str
    unit: str
    dates: list[date]
    values: list[float]
    source_url: str

    def __post_init__(self) -> None:
        if len(self.dates) != len(self.values):
            raise ValueError("dates and values must be the same length")

    def summary(self) -> dict:
        """Compact description that is safe to put in the LLM's context."""
        v = self.values
        recent, prior = v[-7:], v[-14:-7]
        out = {
            "series_id": self.id,
            "label": self.label,
            "unit": self.unit,
            "points": len(v),
            "start": self.dates[0].isoformat(),
            "end": self.dates[-1].isoformat(),
            "latest": sig(v[-1]),
            "min": sig(min(v)),
            "max": sig(max(v)),
            "mean_last_7": sig(mean(recent)),
            "source": self.source_url,
        }
        if prior and mean(prior):
            out["change_last_7_vs_prior_7_pct"] = round(
                100 * (mean(recent) - mean(prior)) / abs(mean(prior)), 1
            )
        return out


@dataclass
class SeriesStore:
    _items: dict[str, Series] = field(default_factory=dict)

    def add(self, series: Series) -> Series:
        self._items[series.id] = series
        return series

    def get(self, series_id: str) -> Series:
        try:
            return self._items[series_id]
        except KeyError:
            known = ", ".join(self._items) or "none yet"
            raise KeyError(
                f"Unknown series_id '{series_id}'. Fetched so far: {known}."
            ) from None


def sig(x: float, figures: int = 5) -> float:
    """Round to significant figures so 76146.8124 and 0.00001234 both read well."""
    return float(f"{float(x):.{figures}g}")


def daily_range(end: date, days: int) -> tuple[date, date]:
    """Inclusive [start, end] window covering `days` days."""
    return end - timedelta(days=days - 1), end
